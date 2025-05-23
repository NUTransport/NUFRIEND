from util import *
from helper import gurobi_suppress_output, load_conversion_factors, \
    load_fuel_tech_eff_factor, load_railroad_values, elec_rate_state

'''
FACILITY SIZING:
    - create another module/file for this
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.min_cost_flow.html
    - define problem as in notes, add source node s and lowerbounds etc.
    - create component to allow for extraction of data from CCWS for a specified time window
    - must get only components of graph where transportation between selected facilities is feasible (within range)
    - may be a collection of disconnected components: if flow appears between two components, cannot serve it
    - can include nodes in these that are not selected by setting recharge costs at them to be infinite

'''

'''
GRAPH PREPROCESSING
'''


def facility_sizing(G: nx.DiGraph, H: nx.DiGraph, fuel_type: str, D: float, unit_sizing_obj=False, emissions_obj=False,
                    suppress_output=True):
    """
    Size facilities by energy usage over <time_window> period using flows
    :param G: [nx.Digraph] with or without flows routed on it
    :param CCWS_filename: [str] name of file to route flows from
    :param comm_flow: [str]/[list] name(s) of commodity groups to be routed; None => 9 groupings; total is always run;
                                   this means only goods movements of comm_flows are converted to new technology
    :param time_window: [tuple[str]] ('MMDDYYYY', 'MMDDYYYY') => (start, end) inclusive time period for data routing
    :param loc_energy_eff: [float] kWh/ton-mi efficiency of locomotive (energy to wheels on track)
    :param batt_p_loc: [int] battery tender cars per locomotive; min/default is 1 (battery on locomotive)
    :param kwh_p_batt: [float] kWh of energy storage capacity of each battery tender car/locomotive
    :param cost_p_location: [int]/[dict] $/kWh of energy by node (or region); default is 1 for all (same)
    :param reroute: [bool] whether all feasible traffic is rerouted to new technology corridors or not
                           -True: reroute all feasible traffic to battery-electric routes;
                           -False: keep original routing and serve only flows with paths in tech corridors
    :param forced_switch: [bool] whether thru traffic on a new tech. link is made to switch to new tech.
                                 -True: all corridors are 100% homogenous tech corridors
                                 -False: corridors allow for mixt tech. traffic
    :param plot: [bool] plot results with nodes sized according to energy demanded in kWh over the time period
    :param crs: [str] projection code
    :return: [nx.Digraph] with new node attrs.
                            - <facility_size> in kWh of energy delivered
                            - <energy_cost> total cost at node
                          with new edge attrs.
                            - <percentage_by_fuel> [dict] with <fuel> keys and % of tonnage moved by <fuel> as value;
                                baseline <fuel> is assumed 'diesel', can alter this to any other
                          and new graph attrs.
                            - <time_window> of flows routed
                            - <comm_flow> of flows routed
    """

    H = deepcopy(H).to_directed()      # for peak sizing of facilities
    F = deepcopy(H)                    # for avg sizing of facilities

    if unit_sizing_obj:
        cost_p_location = {i: 1 for i in G.nodes}
    else:
        # if <emissions_obj> then cost is in [gCO2/kWh], otherwise, cost is in [$/MWh]
        cost_p_location = elec_rate_state(G, emissions=emissions_obj)

    # if isinstance(cost_p_location, float) or isinstance(cost_p_location, int):
    #     c = cost_p_location
    #     cost_p_location = {i: c for i in G.nodes}

    rr_v = load_railroad_values().loc[G.graph['railroad']]  # railroad energy intensity statistics
    ft_ef = load_fuel_tech_eff_factor().loc[fuel_type]  # fuel tech efficiency factors
    cf = load_conversion_factors()['Value']  # numerical constants for conversion across units
    if fuel_type == 'battery':
        # tonmi2kwh = btu/ton-mi * kWh/btu * <energy_efficiency> * <energy_loss> = kWh/ton-mi- not adjusted by commodity
        tonmi2energy = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kwh']) *
                        (1 / rr_v['Energy correction factor']) * (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))
    elif fuel_type == 'hydrogen':
        # tonmi2kwh = btu/ton-mi * kWh/btu * <energy_efficiency> * <energy_loss> = kWh/ton-mi- not adjusted by commodity
        tonmi2energy = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kgh2']) *
                        (1 / rr_v['Energy correction factor']) * (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))
    # battery locomotive range given from D used to calculate battery locomotive energy capacity
    # loc2kwh = kWh/ton-mi * ton/loc * km * mi/km * loc/batt = kWh/loc
    loc2energy = tonmi2energy * rr_v['ton/loc'] * D * cf['mi/km']

    # 3. create augmented graph for MCNF formulation
    # 3.1. EDGE PEAK FLOW ASSIGNMENT
    key = 'MCNF_peak'
    # 3.1.1. update edge parameters for solution
    for u, v in H.edges():
        if fuel_type == 'battery':
            # lower bound of energy required in MWh
            lb = G.edges[u, v][fuel_type + '_peak_kwh']['TOTAL'] / 1000                             # MWh
            # upper bound of energy allowed to flow, based on average locomotive flows in MWh
            ub = G.edges[u, v][fuel_type + '_avg_loc']['TOTAL'] * loc2energy / 1000    # MWh
        elif fuel_type == 'hydrogen':
            # lower bound of energy required in kgh2
            lb = G.edges[u, v][fuel_type + '_peak_kgh2']['TOTAL']  # kgh2
            # upper bound of energy allowed to flow, based on average locomotive flows in kgh2
            ub = G.edges[u, v][fuel_type + '_avg_loc']['TOTAL'] * loc2energy  # kgh2
        H.edges[u, v][key] = dict(
            c=0,        # cost to flow energy along edge
            x=0,        # energy flow on edge in MWh or kgh2; placeholder for decision variable value
            lb=lb,      # lower bound of energy required in MWh or kgh2
            # upper bound is set as the maximum of (a) average energy flow by locomotives or (b) the lb energy required
            # upper bound is infinite if v is not an enabled facility
            ub=max(ub, lb) if H.nodes[v]['facility'] == 1 else np.inf,
            actual_ub=ub    # store the actual upper bound for analysis
        )

    # 3.1.2. create source with edge attrs. to each node in <H>
    s = 'SOURCE'
    H.add_node(s)
    for i in H.nodes():
        if i == s:
            # skip node <s> (no self-loops)
            H.nodes[i][key] = dict(d=0)
            continue
        # sum of all incident edge lower bounds to i
        H.nodes[i][key] = dict(d=-sum([H.edges[j[0], i][key]['lb'] for j in H.in_edges(i)]))
        # different values if a facility does or does not exist at node i
        H.add_edge(s, i, **{key: dict(c=cost_p_location[i] if H.nodes[i]['facility'] == 1 else 0,
                                      x=0,
                                      lb=0,
                                      ub=np.inf if H.nodes[i]['facility'] == 1 else 0,
                                      actual_ub=np.inf if H.nodes[i]['facility'] == 1 else 0)}
                   )
    # update amount that source must supply
    H.nodes[s][key] = dict(d=sum([H.nodes[i][key]['d'] for i in H if i != s]))

    # 3.1.3. solve MCNF for this graph
    H = min_cost_flow(H, key, suppress_output)

    # 3.1.4. interpret results and store in node and edge attrs. of original graph and plot with nodes by size/legend
    for n in G:
        if fuel_type == 'battery':
            G.nodes[n]['peak'] = {'daily_supply_mwh': 0, 'elec_cost': 0, 'daily_demand_mwh': 0, 'number_loc': 0}
        elif fuel_type == 'hydrogen':
            G.nodes[n]['peak'] = {'daily_supply_kgh2': 0, 'h2_cost': 0, 'daily_demand_kgh2': 0, 'number_loc': 0}

    total_energy = 0
    total_cost = 0
    for _, v in H.out_edges(s):
        if fuel_type == 'battery':
            G.nodes[v]['peak']['daily_supply_mwh'] = H.edges[s, v][key]['x']      # energy consumed by facility at v
            G.nodes[v]['peak']['daily_demand_mwh'] = H.nodes[v][key]['d']      # energy demanded by v
            G.nodes[v]['peak']['energy_transfer'] = 0
            total_energy += H.edges[s, v][key]['x']
            if H.nodes[v]['facility'] == 1:
                # cost of total energy consumed at v
                G.nodes[v]['peak']['elec_cost'] = H.edges[s, v][key]['c'] * H.edges[s, v][key]['x']
                # number of batteries charged at facility v
                if G.nodes[v]['peak']['daily_supply_mwh'] > 0:
                    G.nodes[v]['peak']['number_loc'] = np.ceil(max([G.nodes[v]['peak']['daily_supply_mwh'],
                                                                    -G.nodes[v]['peak']['daily_demand_mwh']]) * 1000 /
                                                               loc2energy)
                else:
                    G.nodes[v]['peak']['energy_transfer'] = 1
                    G.nodes[v]['peak']['number_loc'] = np.ceil(-G.nodes[v]['peak']['daily_demand_mwh'] * 1000 /
                                                               loc2energy)
                total_cost += H.edges[s, v][key]['c'] * H.edges[s, v][key]['x']
        elif fuel_type == 'hydrogen':
            G.nodes[v]['peak']['daily_supply_kgh2'] = H.edges[s, v][key]['x']  # energy consumed by facility at v
            G.nodes[v]['peak']['daily_demand_kgh2'] = H.nodes[v][key]['d']  # energy demanded by v
            G.nodes[v]['peak']['energy_transfer'] = 0
            total_energy += H.edges[s, v][key]['x']
            if H.nodes[v]['facility'] == 1:
                # cost of total energy consumed at v
                G.nodes[v]['peak']['h2_cost'] = H.edges[s, v][key]['c'] * H.edges[s, v][key]['x']
                # number of batteries charged at facility v
                if G.nodes[v]['peak']['daily_supply_kgh2'] > 0:
                    G.nodes[v]['peak']['energy_transfer'] = 0
                    G.nodes[v]['peak']['number_loc'] = np.ceil(G.nodes[v]['peak']['daily_supply_kgh2'] / loc2energy)
                else:
                    G.nodes[v]['peak']['energy_transfer'] = 1
                    G.nodes[v]['peak']['number_loc'] = np.ceil(-G.nodes[v]['peak']['daily_demand_kgh2'] / loc2energy)
                total_cost += H.edges[s, v][key]['c'] * H.edges[s, v][key]['x']

    for u, v in G.edges():
        if H.has_edge(u, v):
            G.edges[u, v][key] = H.edges[u, v][key]
            if G.edges[u, v][key]['actual_ub'] != 0:
                G.edges[u, v][key]['x/actual_ub'] = G.edges[u, v][key]['x'] / G.edges[u, v][key]['actual_ub']
            else:
                G.edges[u, v][key]['x/actual_ub'] = 0
        else:
            G.edges[u, v][key] = {'x': 0, 'c': 0, 'lb': 0, 'ub': 0, 'actual_ub': 0, 'x/actual_ub': 0}

    if fuel_type == 'battery':
        G.graph[key] = dict(total_energy_mwh=total_energy, total_cost=total_cost,
                            total_demand_mwh=- H.nodes[s][key]['d'])
    elif fuel_type == 'hydrogen':
        G.graph[key] = dict(total_energy_kgh2=total_energy, total_cost=total_cost,
                            total_demand_kgh2=- H.nodes[s][key]['d'])

    # 3.2. EDGE AVERAGE FLOW ASSIGNMENT
    key = 'MCNF_avg'
    # 3.2.1. update edge parameters for solution
    for u, v in F.edges():
        if fuel_type == 'battery':
            # lower bound of energy required in MWh
            lb = G.edges[u, v][fuel_type + '_avg_kwh']['TOTAL'] / 1000  # MWh
            # upper bound of energy allowed to flow, based on average locomotive flows in MWh
            ub = G.edges[u, v][fuel_type + '_avg_loc']['TOTAL'] * loc2energy / 1000  # MWh
        elif fuel_type == 'hydrogen':
            # lower bound of energy required in kgh2
            lb = G.edges[u, v][fuel_type + '_avg_kgh2']['TOTAL']    # kgh2
            # upper bound of energy allowed to flow, based on average locomotive flows in kgh2
            ub = G.edges[u, v][fuel_type + '_avg_loc']['TOTAL'] * loc2energy  # kgh2
        F.edges[u, v][key] = dict(
            c=0,  # cost to flow energy along edge
            x=0,  # energy flow on edge in MWh or kgh2; placeholder for decision variable value
            lb=lb,  # lower bound of energy required in MWh or kgh2
            # upper bound is the average energy flow by locomotives or infinite if v is not an enabled facility
            ub=ub if F.nodes[v]['facility'] == 1 else np.inf,
            actual_ub=ub  # store the actual upper bound for analysis
        )

    # 3.2.2. create source with edge attrs. to each node in <F>
    s = 'SOURCE'
    F.add_node(s)
    for i in F.nodes():
        if i == s:
            # skip node <s> (no self-loops)
            F.nodes[i][key] = dict(d=0)
            continue
        # sum of all incident edge lower bounds to i
        F.nodes[i][key] = dict(d=-sum([F.edges[j[0], i][key]['lb'] for j in F.in_edges(i)]))
        # different values if a facility does or does not exist at node i
        F.add_edge(s, i, **{key: dict(c=cost_p_location[i] if F.nodes[i]['facility'] == 1 else 0,
                                      x=0,
                                      lb=0,
                                      ub=H.edges[s, i]['MCNF_peak']['x'] if F.nodes[i]['facility'] == 1 else 0,
                                      actual_ub=H.edges[s, i]['MCNF_peak']['x'] if F.nodes[i]['facility'] == 1 else 0)}
                   )
        # update amount that source must supply
    F.nodes[s][key] = dict(d=sum([F.nodes[i][key]['d'] for i in F if i != s]))

    # 3.2.3. solve MCNF for this graph
    F = min_cost_flow(F, key, suppress_output)

    # 3.2.4. interpret results and store in node and edge attrs. of original graph and plot with nodes by size/legend
    for n in G:
        if fuel_type == 'battery':
            G.nodes[n]['avg'] = {'daily_supply_mwh': 0, 'elec_cost': 0, 'daily_demand_mwh': 0, 'number_loc': 0}
        elif fuel_type == 'hydrogen':
            G.nodes[n]['avg'] = {'daily_supply_kgh2': 0, 'h2_cost': 0, 'daily_demand_kgh2': 0, 'number_loc': 0}

    total_energy = 0
    total_cost = 0
    for _, v in H.out_edges(s):
        if fuel_type == 'battery':
            G.nodes[v]['avg']['daily_supply_mwh'] = F.edges[s, v][key]['x']  # energy consumed by facility at v
            G.nodes[v]['avg']['daily_demand_mwh'] = F.nodes[v][key]['d']  # energy demanded by v
            G.nodes[v]['avg']['energy_transfer'] = 0
            total_energy += F.edges[s, v][key]['x']
            if F.nodes[v]['facility'] == 1:
                # cost of total energy consumed at v
                G.nodes[v]['avg']['elec_cost'] = F.edges[s, v][key]['c'] * F.edges[s, v][key]['x']
                # number of batteries charged at facility v
                if G.nodes[v]['avg']['daily_supply_mwh'] > 0:
                    G.nodes[v]['avg']['number_loc'] = np.ceil(max([G.nodes[v]['avg']['daily_supply_mwh'],
                                                                   -G.nodes[v]['avg']['daily_demand_mwh']]) * 1000 /
                                                              loc2energy)
                else:
                    G.nodes[v]['avg']['energy_transfer'] = 1
                    G.nodes[v]['avg']['number_loc'] = np.ceil(-G.nodes[v]['avg']['daily_demand_mwh'] * 1000 /
                                                              loc2energy)
                total_cost += F.edges[s, v][key]['c'] * F.edges[s, v][key]['x']
        elif fuel_type == 'hydrogen':
            G.nodes[v]['avg']['daily_supply_kgh2'] = F.edges[s, v][key]['x']  # energy consumed by facility at v
            G.nodes[v]['avg']['daily_demand_kgh2'] = F.nodes[v][key]['d']  # energy demanded by v
            G.nodes[v]['avg']['energy_transfer'] = 0
            total_energy += F.edges[s, v][key]['x']
            if F.nodes[v]['facility'] == 1:
                # cost of total energy consumed at v
                G.nodes[v]['avg']['h2_cost'] = F.edges[s, v][key]['c'] * F.edges[s, v][key]['x']
                # number of batteries charged at facility v
                if G.nodes[v]['avg']['daily_supply_kgh2'] > 0:
                    G.nodes[v]['avg']['energy_transfer'] = 0
                    G.nodes[v]['avg']['number_loc'] = np.ceil(G.nodes[v]['avg']['daily_supply_kgh2'] / loc2energy)
                else:
                    G.nodes[v]['avg']['energy_transfer'] = 1
                    G.nodes[v]['avg']['number_loc'] = np.ceil(-G.nodes[v]['avg']['daily_demand_kgh2'] / loc2energy)
                total_cost += F.edges[s, v][key]['c'] * F.edges[s, v][key]['x']

    for u, v in G.edges():
        if F.has_edge(u, v):
            G.edges[u, v][key] = F.edges[u, v][key]
            if G.edges[u, v][key]['actual_ub'] != 0:
                G.edges[u, v][key]['x/actual_ub'] = G.edges[u, v][key]['x'] / G.edges[u, v][key]['actual_ub']
            else:
                G.edges[u, v][key]['x/actual_ub'] = 0
        else:
            G.edges[u, v][key] = {'x': 0, 'c': 0, 'lb': 0, 'ub': 0, 'actual_ub': 0, 'x/actual_ub': 0}

    if fuel_type == 'battery':
        G.graph[key] = dict(total_energy_mwh=total_energy, total_cost=total_cost,
                            total_demand_mwh=- F.nodes[s][key]['d'])
    elif fuel_type == 'hydrogen':
        G.graph[key] = dict(total_energy_kgh2=total_energy, total_cost=total_cost,
                            total_demand_kgh2=- F.nodes[s][key]['d'])

    return G


def facility_sizing_hybrid(G: nx.DiGraph, H: nx.DiGraph, fuel_type: str, D: float, unit_sizing_obj=False,
                           emissions_obj=False, suppress_output=True):
    """
    Size facilities by energy usage over <time_window> period using flows
    :param G: [nx.Digraph] with or without flows routed on it
    :param CCWS_filename: [str] name of file to route flows from
    :param comm_flow: [str]/[list] name(s) of commodity groups to be routed; None => 9 groupings; total is always run;
                                   this means only goods movements of comm_flows are converted to new technology
    :param time_window: [tuple[str]] ('MMDDYYYY', 'MMDDYYYY') => (start, end) inclusive time period for data routing
    :param loc_energy_eff: [float] kWh/ton-mi efficiency of locomotive (energy to wheels on track)
    :param batt_p_loc: [int] battery tender cars per locomotive; min/default is 1 (battery on locomotive)
    :param kwh_p_batt: [float] kWh of energy storage capacity of each battery tender car/locomotive
    :param cost_p_location: [int]/[dict] $/kWh of energy by node (or region); default is 1 for all (same)
    :param reroute: [bool] whether all feasible traffic is rerouted to new technology corridors or not
                           -True: reroute all feasible traffic to battery-electric routes;
                           -False: keep original routing and serve only flows with paths in tech corridors
    :param forced_switch: [bool] whether thru traffic on a new tech. link is made to switch to new tech.
                                 -True: all corridors are 100% homogenous tech corridors
                                 -False: corridors allow for mixt tech. traffic
    :param plot: [bool] plot results with nodes sized according to energy demanded in kWh over the time period
    :param crs: [str] projection code
    :return: [nx.Digraph] with new node attrs.
                            - <facility_size> in kWh of energy delivered
                            - <energy_cost> total cost at node
                          with new edge attrs.
                            - <percentage_by_fuel> [dict] with <fuel> keys and % of tonnage moved by <fuel> as value;
                                baseline <fuel> is assumed 'diesel', can alter this to any other
                          and new graph attrs.
                            - <time_window> of flows routed
                            - <comm_flow> of flows routed
    """

    H = deepcopy(H).to_directed()      # for peak sizing of facilities
    F = deepcopy(H)                    # for avg sizing of facilities

    if unit_sizing_obj:
        cost_p_location = {i: 1 for i in G.nodes}
    else:
        # if <emissions_obj> then cost is in [gCO2/kWh], otherwise, cost is in [$/MWh]
        cost_p_location = elec_rate_state(G, emissions=emissions_obj)

    # if isinstance(cost_p_location, float) or isinstance(cost_p_location, int):
    #     c = cost_p_location
    #     cost_p_location = {i: c for i in G.nodes}

    rr_v = load_railroad_values().loc[G.graph['railroad']]  # railroad energy intensity statistics
    # ft_ef = load_fuel_tech_eff_factor().loc[fuel_type]  # fuel tech efficiency factors
    # cf = load_conversion_factors()['Value']  # numerical constants for conversion across units
    # tonmi2kwh = btu/ton-mi * kWh/btu * <energy_efficiency> * <energy_loss> = kWh/ton-mi- not adjusted by commodity
    # tonmi2energy = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kwh']) *
    #                 (1 / rr_v['Energy correction factor']) * (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))
    # battery locomotive range given from D used to calculate battery locomotive energy capacity
    # loc2kwh = kWh/ton-mi * ton/loc * km * mi/km * loc/batt = kWh/loc
    eff_kwh_p_batt = G.graph['scenario']['eff_kwh_p_batt']
    hybrid_energy_factor = (1 / rr_v['Energy correction factor']) * (1 / rr_v['hybrid energy factor'])
    # loc/kwh
    loc2kwh = hybrid_energy_factor * eff_kwh_p_batt
    # loc2energy = tonmi2energy * rr_v['ton/loc'] * D * cf['mi/km']   # kwh/loc

    fuel_type_batt = fuel_type + '_battery'

    # 3. create augmented graph for MCNF formulation
    # 3.1. EDGE PEAK FLOW ASSIGNMENT
    key = 'MCNF_peak'
    # 3.1.1. update edge parameters for solution
    for u, v in H.edges():
        # lower bound of energy required in MWh
        lb = G.edges[u, v][fuel_type_batt + '_peak_kwh']['TOTAL'] / 1000                             # MWh
        # upper bound of energy allowed to flow, based on peak locomotive flows in MWh
        ub = G.edges[u, v][fuel_type_batt + '_peak_loc']['TOTAL'] * loc2kwh / 1000    # MWh
        H.edges[u, v][key] = dict(
            c=0,        # cost to flow energy along edge
            x=0,        # energy flow on edge in MWh or kgh2; placeholder for decision variable value
            lb=lb,      # lower bound of energy required in MWh or kgh2
            # upper bound is set as the maximum of (a) average energy flow by locomotives or (b) the lb energy required
            # upper bound is infinite if v is not an enabled facility
            ub=max(ub, lb) if H.nodes[v]['facility'] == 1 else np.inf,
            actual_ub=ub    # store the actual upper bound for analysis
        )

    # 3.1.2. create source with edge attrs. to each node in <H>
    s = 'SOURCE'
    H.add_node(s)
    for i in H.nodes():
        if i == s:
            # skip node <s> (no self-loops)
            H.nodes[i][key] = dict(d=0)
            continue
        # sum of all incident edge lower bounds to i
        H.nodes[i][key] = dict(d=-sum([H.edges[j[0], i][key]['lb'] for j in H.in_edges(i)]))
        # different values if a facility does or does not exist at node i
        H.add_edge(s, i, **{key: dict(c=cost_p_location[i] if H.nodes[i]['facility'] == 1 else 0,
                                      x=0,
                                      lb=0,
                                      ub=np.inf if H.nodes[i]['facility'] == 1 else 0,
                                      actual_ub=np.inf if H.nodes[i]['facility'] == 1 else 0)}
                   )
    # update amount that source must supply
    H.nodes[s][key] = dict(d=sum([H.nodes[i][key]['d'] for i in H if i != s]))

    # 3.1.3. solve MCNF for this graph
    H = min_cost_flow_hybrid(H, key, suppress_output)

    # 3.1.4. interpret results and store in node and edge attrs. of original graph and plot with nodes by size/legend
    for n in G:
        G.nodes[n]['peak'] = {'daily_supply_mwh': 0, 'elec_cost': 0, 'daily_demand_mwh': 0, 'number_loc': 0}

    total_energy = 0
    total_cost = 0
    for _, v in H.out_edges(s):
        G.nodes[v]['peak']['daily_supply_mwh'] = H.edges[s, v][key]['x']      # energy consumed by facility at v
        G.nodes[v]['peak']['daily_demand_mwh'] = H.nodes[v][key]['d']      # energy demanded by v
        G.nodes[v]['peak']['energy_transfer'] = 0
        total_energy += H.edges[s, v][key]['x']
        if H.nodes[v]['facility'] == 1:
            # cost of total energy consumed at v
            G.nodes[v]['peak']['elec_cost'] = H.edges[s, v][key]['c'] * H.edges[s, v][key]['x']
            # number of batteries charged at facility v
            if G.nodes[v]['peak']['daily_supply_mwh'] > 0:
                G.nodes[v]['peak']['number_loc'] = np.ceil(max([G.nodes[v]['peak']['daily_supply_mwh'],
                                                                -G.nodes[v]['peak']['daily_demand_mwh']]) * 1000 /
                                                           loc2kwh)
            else:
                G.nodes[v]['peak']['energy_transfer'] = 1
                G.nodes[v]['peak']['number_loc'] = np.ceil(-G.nodes[v]['peak']['daily_demand_mwh'] * 1000 /
                                                           loc2kwh)
            total_cost += H.edges[s, v][key]['c'] * H.edges[s, v][key]['x']

    for u, v in G.edges():
        if H.has_edge(u, v):
            G.edges[u, v][key] = H.edges[u, v][key]
            if G.edges[u, v][key]['actual_ub'] != 0:
                G.edges[u, v][key]['x/actual_ub'] = G.edges[u, v][key]['x'] / G.edges[u, v][key]['actual_ub']
            else:
                G.edges[u, v][key]['x/actual_ub'] = 0
        else:
            G.edges[u, v][key] = {'x': 0, 'c': 0, 'lb': 0, 'ub': 0, 'actual_ub': 0, 'x/actual_ub': 0}

    G.graph[key] = dict(total_energy_mwh=total_energy, total_cost=total_cost,
                        total_demand_mwh=- H.nodes[s][key]['d'])

    # 3.2. EDGE AVERAGE FLOW ASSIGNMENT
    key = 'MCNF_avg'
    # 3.2.1. update edge parameters for solution
    for u, v in F.edges():
        # lower bound of energy required in MWh
        lb = G.edges[u, v][fuel_type_batt + '_avg_kwh']['TOTAL'] / 1000  # MWh
        # upper bound of energy allowed to flow, based on peak locomotive flows in MWh
        ub = G.edges[u, v][fuel_type_batt + '_peak_loc']['TOTAL'] * loc2kwh / 1000  # MWh
        ub = max(lb, ub)
        F.edges[u, v][key] = dict(
            c=0,  # cost to flow energy along edge
            x=0,  # energy flow on edge in MWh or kgh2; placeholder for decision variable value
            lb=lb,  # lower bound of energy required in MWh or kgh2
            # upper bound is the average energy flow by locomotives or infinite if v is not an enabled facility
            ub=ub if F.nodes[v]['facility'] == 1 else np.inf,
            actual_ub=ub  # store the actual upper bound for analysis
        )

    # 3.2.2. create source with edge attrs. to each node in <F>
    s = 'SOURCE'
    F.add_node(s)
    for i in F.nodes():
        if i == s:
            # skip node <s> (no self-loops)
            F.nodes[i][key] = dict(d=0)
            continue
        # sum of all incident edge lower bounds to i
        F.nodes[i][key] = dict(d=-sum([F.edges[j[0], i][key]['lb'] for j in F.in_edges(i)]))
        # different values if a facility does or does not exist at node i
        F.add_edge(s, i, **{key: dict(c=cost_p_location[i] if F.nodes[i]['facility'] == 1 else 0,
                                      x=0,
                                      lb=0,
                                      ub=H.edges[s, i]['MCNF_peak']['x'] if F.nodes[i]['facility'] == 1 else 0,
                                      actual_ub=H.edges[s, i]['MCNF_peak']['x'] if F.nodes[i]['facility'] == 1 else 0)}
                   )
        # update amount that source must supply
    F.nodes[s][key] = dict(d=sum([F.nodes[i][key]['d'] for i in F if i != s]))

    # 3.2.3. solve MCNF for this graph
    F = min_cost_flow_hybrid(F, key, suppress_output)

    # 3.2.4. interpret results and store in node and edge attrs. of original graph and plot with nodes by size/legend
    for n in G:
        G.nodes[n]['avg'] = {'daily_supply_mwh': 0, 'elec_cost': 0, 'daily_demand_mwh': 0, 'number_loc': 0}

    total_energy = 0
    total_cost = 0
    for _, v in H.out_edges(s):
        G.nodes[v]['avg']['daily_supply_mwh'] = F.edges[s, v][key]['x']  # energy consumed by facility at v
        G.nodes[v]['avg']['daily_demand_mwh'] = F.nodes[v][key]['d']  # energy demanded by v
        G.nodes[v]['avg']['energy_transfer'] = 0
        total_energy += F.edges[s, v][key]['x']
        if F.nodes[v]['facility'] == 1:
            # cost of total energy consumed at v
            G.nodes[v]['avg']['elec_cost'] = F.edges[s, v][key]['c'] * F.edges[s, v][key]['x']
            # number of batteries charged at facility v
            if G.nodes[v]['avg']['daily_supply_mwh'] > 0:
                G.nodes[v]['avg']['number_loc'] = np.ceil(max([G.nodes[v]['avg']['daily_supply_mwh'],
                                                               -G.nodes[v]['avg']['daily_demand_mwh']]) * 1000 /
                                                          loc2kwh)
            else:
                G.nodes[v]['avg']['energy_transfer'] = 1
                G.nodes[v]['avg']['number_loc'] = np.ceil(-G.nodes[v]['avg']['daily_demand_mwh'] * 1000 / loc2kwh)
            total_cost += F.edges[s, v][key]['c'] * F.edges[s, v][key]['x']

    for u, v in G.edges():
        if F.has_edge(u, v):
            G.edges[u, v][key] = F.edges[u, v][key]
            if G.edges[u, v][key]['actual_ub'] != 0:
                G.edges[u, v][key]['x/actual_ub'] = G.edges[u, v][key]['x'] / G.edges[u, v][key]['actual_ub']
            else:
                G.edges[u, v][key]['x/actual_ub'] = 0
        else:
            G.edges[u, v][key] = {'x': 0, 'c': 0, 'lb': 0, 'ub': 0, 'actual_ub': 0, 'x/actual_ub': 0}

    G.graph[key] = dict(total_energy_mwh=total_energy, total_cost=total_cost,
                        total_demand_mwh=- F.nodes[s][key]['d'])

    return G


'''
MCNF LP SOLVER
'''


def min_cost_flow(G: nx.DiGraph, key: str, suppress_output=True):
    # solve min cost flow for graph G and return G with solution (flows) as 'x' attribute for each edge

    m = gp.Model('Facility Sizing Problem', env=gurobi_suppress_output(suppress_output))

    edges, costs, lower, upper = gp.multidict({e: [G.edges[e][key]['c'],
                                                   G.edges[e][key]['lb'],
                                                   G.edges[e][key]['ub']] for e in G.edges})

    s = 'SOURCE'
    demand = {i: G.nodes[i][key]['d'] for i in set(G.nodes).difference({s})}
    supply = {s: G.nodes[s][key]['d']}

    x = m.addVars(edges, obj=costs, lb=lower, ub=upper, name='x')

    m.addConstrs((gp.quicksum(x.select(i, '*')) - gp.quicksum(x.select('*', i)) == demand[i] for i in demand.keys()),
                 name='flow balance DEMAND')

    m.addConstr((gp.quicksum(x.select(s, '*')) == - supply[s]), 'flow balance SUPPLY')

    # m.computeIIS()
    # m.write(os.path.join('/Users/adrianhz/Desktop', 'model.ilp'))

    # optimize
    m.update()
    m.optimize()
    # extract solution values
    x_val = m.getAttr('x', x).items()  # get facility size values
    # z_val = m.objval  # get objective fxn value

    for e, v in x_val:
        G.edges[e[0], e[1]][key]['x'] = v

    return G


def min_cost_flow_hybrid(G: nx.DiGraph, key: str, suppress_output=True):
    # solve min cost flow for graph G and return G with solution (flows) as 'x' attribute for each edge

    m = gp.Model('Facility Sizing Problem', env=gurobi_suppress_output(suppress_output))

    edges, costs, lower, upper = gp.multidict({e: [G.edges[e][key]['c'],
                                                   G.edges[e][key]['lb'],
                                                   G.edges[e][key]['ub']] for e in G.edges})

    s = 'SOURCE'
    demand = {i: G.nodes[i][key]['d'] for i in set(G.nodes).difference({s})}
    supply = {s: G.nodes[s][key]['d']}

    # removed upperbound - now unbounded
    # x = m.addVars(edges, obj=costs, lb=lower, name='x')
    x = m.addVars(edges, obj=costs, lb=lower, ub=upper, name='x')

    m.addConstrs((gp.quicksum(x.select(i, '*')) - gp.quicksum(x.select('*', i)) == demand[i] for i in demand.keys()),
                 name='flow balance DEMAND')

    m.addConstr((gp.quicksum(x.select(s, '*')) == - supply[s]), 'flow balance SUPPLY')

    # m.computeIIS()
    # m.write(os.path.join('/Users/adrianhz/Desktop', 'model.ilp'))

    # optimize
    m.update()
    m.optimize()
    # extract solution values
    x_val = m.getAttr('x', x).items()  # get facility size values
    # z_val = m.objval  # get objective fxn value

    for e, v in x_val:
        G.edges[e[0], e[1]][key]['x'] = v

    return G
