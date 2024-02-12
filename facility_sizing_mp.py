from util import *
from helper import gurobi_suppress_output, load_conversion_factors, \
    load_fuel_tech_eff_factor, load_railroad_values, elec_rate_state, elec_rate_state_mp
from network_representation import remove_from_graph

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

RAIL_DATA_DIR = '/Users/maxng/Library/CloudStorage/OneDrive-NorthwesternUniversity/ARPA-E LOCOMOTIVES/Rail Data'
KM2MI = 0.62137119  # [mi/km]


'''
GRAPH PREPROCESSING
'''


def facility_sizing_mp(G: nx.DiGraph, time_horizon: list, fuel_type: str, D: float, unit_sizing_obj=False,
                       emissions_obj=False, suppress_output=True):

    # instantiate storage dicts
    for n in G:
        if fuel_type == 'battery':
            G.nodes[n]['avg'] = {t: {'daily_supply_mwh': 0, 'elec_cost': 0, 'daily_demand_mwh': 0, 'number_loc': 0}
                                 for t in time_horizon}
        elif fuel_type == 'hydrogen':
            G.nodes[n]['avg'] = {t: {'daily_supply_kgh2': 0, 'h2_cost': 0, 'daily_demand_kgh2': 0, 'number_loc': 0}
                                 for t in time_horizon}

    for u, v in G.edges:
        G.edges[u, v]['MCNF_avg'] = {t: {'x': 0, 'c': 0, 'lb': 0, 'ub': 0, 'actual_ub': 0, 'x/actual_ub': 0}
                                     for t in time_horizon}

    G.graph['MCNF_avg'] = {t: dict() for t in time_horizon}

    # run facility sizing LP for each time period
    for t in time_horizon:
        G = facility_sizing_step_mp(G=G, time_step=t, fuel_type=fuel_type, D=D, unit_sizing_obj=unit_sizing_obj,
                                    emissions_obj=emissions_obj, suppress_output=suppress_output)

    return G


def facility_sizing_step_mp(G: nx.DiGraph, time_step: str, fuel_type: str, D: float, unit_sizing_obj=False,
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

    F = deepcopy(selected_subgraph(G, time_step)).to_directed()     # for avg sizing of facilities

    if unit_sizing_obj:
        cost_p_location = {i: 1 for i in G.nodes}
    else:
        # TODO: use updated time- and state-specific emissions costs
        # if <emissions_obj> then cost is in [gCO2/kWh], otherwise, cost is in [$/MWh]
        cost_p_location = elec_rate_state_mp(G, year=time_step)

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

    # 3.2. EDGE AVERAGE FLOW ASSIGNMENT
    key = 'MCNF_avg'
    # 3.2.1. update edge parameters for solution
    for u, v in F.edges():
        if fuel_type == 'battery':
            # lower bound of energy required in MWh
            lb = G.edges[u, v][fuel_type + '_avg_kwh'][time_step]['TOTAL'] / 1000  # MWh
            # upper bound of energy allowed to flow, based on average locomotive flows in MWh
            ub = G.edges[u, v][fuel_type + '_avg_loc'][time_step]['TOTAL'] * loc2energy / 1000  # MWh
        elif fuel_type == 'hydrogen':
            # lower bound of energy required in kgh2
            lb = G.edges[u, v][fuel_type + '_avg_kgh2'][time_step]['TOTAL']    # kgh2
            # upper bound of energy allowed to flow, based on average locomotive flows in kgh2
            ub = G.edges[u, v][fuel_type + '_avg_loc'][time_step]['TOTAL'] * loc2energy  # kgh2
        F.edges[u, v][key] = dict(
            c=0,  # cost to flow energy along edge
            x=0,  # energy flow on edge in MWh or kgh2; placeholder for decision variable value
            lb=lb,  # lower bound of energy required in MWh or kgh2
            # upper bound is the average energy flow by locomotives or infinite if v is not an enabled facility
            ub=ub if F.nodes[v]['facility'][time_step] == 1 else np.inf,
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
        F.add_edge(s, i, **{key: dict(c=cost_p_location[i] if F.nodes[i]['facility'][time_step] == 1 else 0,
                                      x=0,
                                      lb=0,
                                      ub=np.inf if F.nodes[i]['facility'][time_step] == 1 else 0,
                                      actual_ub=np.inf if F.nodes[i]['facility'][time_step] == 1 else 0)}
                   )
        # update amount that source must supply
    F.nodes[s][key] = dict(d=sum([F.nodes[i][key]['d'] for i in F if i != s]))

    # 3.2.3. solve MCNF for this graph
    F = min_cost_flow(F, key, suppress_output)

    # 3.2.4. interpret results and store in node and edge attrs. of original graph and plot with nodes by size/legend
    # for n in G:
    #     if fuel_type == 'battery':
    #         G.nodes[n]['avg'] = {'daily_supply_mwh': 0, 'elec_cost': 0, 'daily_demand_mwh': 0, 'number_loc': 0}
    #     elif fuel_type == 'hydrogen':
    #         G.nodes[n]['avg'] = {'daily_supply_kgh2': 0, 'h2_cost': 0, 'daily_demand_kgh2': 0, 'number_loc': 0}

    total_energy = 0
    total_cost = 0
    for _, v in F.out_edges(s):
        if fuel_type == 'battery':
            G.nodes[v]['avg'][time_step]['daily_supply_mwh'] = F.edges[s, v][key]['x']  # energy consumed by facility at v
            G.nodes[v]['avg'][time_step]['daily_demand_mwh'] = F.nodes[v][key]['d']  # energy demanded by v
            G.nodes[v]['avg'][time_step]['energy_transfer'] = 0
            total_energy += F.edges[s, v][key]['x']
            if F.nodes[v]['facility'][time_step] == 1:
                # cost of total energy consumed at v
                G.nodes[v]['avg'][time_step]['elec_cost'] = F.edges[s, v][key]['c'] * F.edges[s, v][key]['x']
                # number of batteries charged at facility v
                if G.nodes[v]['avg'][time_step]['daily_supply_mwh'] > 0:
                    G.nodes[v]['avg'][time_step]['number_loc'] = np.ceil(max([
                        G.nodes[v]['avg'][time_step]['daily_supply_mwh'],
                        -G.nodes[v]['avg'][time_step]['daily_demand_mwh']]) * 1000 / loc2energy)
                else:
                    G.nodes[v]['avg'][time_step]['energy_transfer'] = 1
                    G.nodes[v]['avg'][time_step]['number_loc'] = np.ceil(
                        -G.nodes[v]['avg'][time_step]['daily_demand_mwh'] * 1000 / loc2energy)
                total_cost += F.edges[s, v][key]['c'] * F.edges[s, v][key]['x']
        elif fuel_type == 'hydrogen':
            G.nodes[v]['avg'][time_step]['daily_supply_kgh2'] = F.edges[s, v][key]['x']  # energy consumed by facility at v
            G.nodes[v]['avg'][time_step]['daily_demand_kgh2'] = F.nodes[v][key]['d']  # energy demanded by v
            G.nodes[v]['avg'][time_step]['energy_transfer'] = 0
            total_energy += F.edges[s, v][key]['x']
            if F.nodes[v]['facility'][time_step] == 1:
                # cost of total energy consumed at v
                G.nodes[v]['avg'][time_step]['h2_cost'] = F.edges[s, v][key]['c'] * F.edges[s, v][key]['x']
                # number of batteries charged at facility v
                if G.nodes[v]['avg'][time_step]['daily_supply_kgh2'] > 0:
                    G.nodes[v]['avg'][time_step]['energy_transfer'] = 0
                    G.nodes[v]['avg'][time_step]['number_loc'] = np.ceil(
                        G.nodes[v]['avg'][time_step]['daily_supply_kgh2'] / loc2energy)
                else:
                    G.nodes[v]['avg'][time_step]['energy_transfer'] = 1
                    G.nodes[v]['avg'][time_step]['number_loc'] = np.ceil(
                        -G.nodes[v]['avg'][time_step]['daily_demand_kgh2'] / loc2energy)
                total_cost += F.edges[s, v][key]['c'] * F.edges[s, v][key]['x']

    for u, v in G.edges():
        if F.has_edge(u, v):
            G.edges[u, v][key][time_step] = F.edges[u, v][key]
            if G.edges[u, v][key][time_step]['actual_ub'] != 0:
                G.edges[u, v][key][time_step]['x/actual_ub'] = (G.edges[u, v][key][time_step]['x'] /
                                                                G.edges[u, v][key][time_step]['actual_ub'])
            else:
                G.edges[u, v][key][time_step]['x/actual_ub'] = 0
        else:
            G.edges[u, v][key][time_step] = {'x': 0, 'c': 0, 'lb': 0, 'ub': 0, 'actual_ub': 0, 'x/actual_ub': 0}

    if fuel_type == 'battery':
        G.graph[key][time_step] = dict(total_energy_mwh=total_energy, total_cost=total_cost,
                                       total_demand_mwh=- F.nodes[s][key]['d'])
    elif fuel_type == 'hydrogen':
        G.graph[key][time_step] = dict(total_energy_kgh2=total_energy, total_cost=total_cost,
                                       total_demand_kgh2=- F.nodes[s][key]['d'])

    return G


def hydrogen_facility_sizing(G: nx.DiGraph, H: nx.DiGraph, fuel_type: str, D: float, cost_p_location: any = 1,
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

    if isinstance(cost_p_location, float) or isinstance(cost_p_location, int):
        c = cost_p_location
        cost_p_location = {i: c for i in G.nodes}

    rr_v = load_railroad_values().loc[G.graph['railroad']]  # railroad energy intensity statistics
    ft_ef = load_fuel_tech_eff_factor().loc[fuel_type]  # fuel tech efficiency factors
    cf = load_conversion_factors()['Value']  # numerical constants for conversion across units
    # tonmi2kgh2 = btu/ton-mi * kgh2/btu * <energy_efficiency> * <energy_loss> = kgh2/ton-mi- not adjusted by commodity
    tonmi2kgh2 = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kgh2']) * (1 / rr_v['Energy correction factor'])
                  * (1 / ft_ef['Efficiency factor']))
    # battery locomotive range given from D used to calculate battery locomotive energy capacity
    # loc2kwh = kWh/ton-mi * ton/loc * km * mi/km * loc/batt = kgh2/loc
    loc2kgh2 = tonmi2kgh2 * rr_v['ton/loc'] * D * cf['mi/km']

    # 3. create augmented graph for MCNF formulation
    # 3.1. EDGE PEAK FLOW ASSIGNMENT
    key = 'MCNF_peak'
    # 3.1.1. update edge parameters for solution
    for u, v in H.edges():
        # lower bound of energy required in kgh2
        lb = G.edges[u, v][fuel_type + '_peak_kgh2']['TOTAL']                             # kgh2
        # upper bound of energy allowed to flow, based on average locomotive flows in kgh2
        ub = G.edges[u, v][fuel_type + '_avg_loc']['TOTAL'] * loc2kgh2    # kgh2
        H.edges[u, v][key] = dict(
            c=0,        # cost to flow energy along edge
            x=0,        # energy flow on edge in kgh2; placeholder for decision variable value
            lb=lb,      # lower bound of energy required in kgh2
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
        G.nodes[n]['peak'] = {'daily_supply_kgh2': 0, 'daily_demand_kgh2': 0, 'number_loc': 0}

    total_energy = 0
    total_cost = 0
    for _, v in H.out_edges(s):
        G.nodes[v]['peak']['daily_supply_kgh2'] = H.edges[s, v][key]['x']      # energy consumed by facility at v
        G.nodes[v]['peak']['daily_demand_kgh2'] = H.nodes[v][key]['d']      # energy demanded by v
        G.nodes[v]['peak']['energy_transfer'] = 0
        total_energy += H.edges[s, v][key]['x']
        if H.nodes[v]['facility'] == 1:
            # number of locomotives filled at facility v
            if G.nodes[v]['peak']['daily_supply_kgh2'] > 0:
                G.nodes[v]['peak']['energy_transfer'] = 0
                G.nodes[v]['peak']['number_loc'] = np.ceil(G.nodes[v]['peak']['daily_supply_kgh2'] / loc2kgh2)
            else:
                G.nodes[v]['peak']['energy_transfer'] = 1
                G.nodes[v]['peak']['number_loc'] = np.ceil(-G.nodes[v]['peak']['daily_demand_kgh2'] / loc2kgh2)

    for u, v in G.edges():
        if H.has_edge(u, v):
            G.edges[u, v][key] = H.edges[u, v][key]
            if G.edges[u, v][key]['actual_ub'] != 0:
                G.edges[u, v][key]['x/actual_ub'] = G.edges[u, v][key]['x'] / G.edges[u, v][key]['actual_ub']
            else:
                G.edges[u, v][key]['x/actual_ub'] = 0
        else:
            G.edges[u, v][key] = {'x': 0, 'c': 0, 'lb': 0, 'ub': 0, 'actual_ub': 0, 'x/actual_ub': 0}

    G.graph[key] = dict(total_energy_mwh=total_energy, total_demand_mwh=- H.nodes[s][key]['d'])

    # 3.2. EDGE AVERAGE FLOW ASSIGNMENT
    key = 'MCNF_avg'
    # 3.2.1. update edge parameters for solution
    for u, v in F.edges():
        # lower bound of energy required in kgh2
        lb = G.edges[u, v][fuel_type + '_avg_kgh2']['TOTAL']  # kgh2
        # upper bound of energy allowed to flow, based on average locomotive flows in kgh2
        ub = G.edges[u, v][fuel_type + '_avg_loc']['TOTAL'] * loc2kgh2  # kgh2
        F.edges[u, v][key] = dict(
            c=0,  # cost to flow energy along edge
            x=0,  # energy flow on edge in kgh2; placeholder for decision variable value
            lb=lb,  # lower bound of energy required in kgh2
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
        G.nodes[n]['avg'] = {'daily_supply_kgh2': 0, 'daily_demand_kgh2': 0, 'number_loc': 0}

    total_energy = 0
    total_cost = 0
    for _, v in H.out_edges(s):
        G.nodes[v]['avg']['daily_supply_kgh2'] = F.edges[s, v][key]['x']  # energy consumed by facility at v
        G.nodes[v]['avg']['daily_demand_kgh2'] = F.nodes[v][key]['d']  # energy demanded by v
        G.nodes[v]['avg']['energy_transfer'] = 0
        total_energy += F.edges[s, v][key]['x']
        if F.nodes[v]['facility'] == 1:
            # number of locomotives filled at facility v
            if G.nodes[v]['avg']['daily_supply_kgh2'] > 0:
                G.nodes[v]['avg']['energy_transfer'] = 0
                G.nodes[v]['avg']['number_loc'] = np.ceil(G.nodes[v]['avg']['daily_supply_kgh2'] / loc2kgh2)
            else:
                G.nodes[v]['avg']['energy_transfer'] = 1
                G.nodes[v]['avg']['number_loc'] = np.ceil(-G.nodes[v]['avg']['daily_demand_kgh2'] / loc2kgh2)

    for u, v in G.edges():
        if F.has_edge(u, v):
            G.edges[u, v][key] = F.edges[u, v][key]
            if G.edges[u, v][key]['actual_ub'] != 0:
                G.edges[u, v][key]['x/actual_ub'] = G.edges[u, v][key]['x'] / G.edges[u, v][key]['actual_ub']
            else:
                G.edges[u, v][key]['x/actual_ub'] = 0
        else:
            G.edges[u, v][key] = {'x': 0, 'c': 0, 'lb': 0, 'ub': 0, 'actual_ub': 0, 'x/actual_ub': 0}

    G.graph[key] = dict(total_energy_mwh=total_energy, total_demand_mwh=- F.nodes[s][key]['d'])

    return G

'''
Problem lies in the tightness of upperbounds, these are not sufficiently large to provide additional energy to nodes 
that do not have facilities located at them, hence infeasibility. One idea is to simplify this graph further and remove
those nodes that are not covered facilities. This may be a problem in the routing module. 
Could make a nodes lb to be the sum of its own plus all subsequent (direction matters) non-facility nodes lb's. 
E.g., suppose i->j->k where i, k selected facilities and j not selected,
 suppose (i,j)_lb = 10, (j,k)_lb = 10 and (i,j)_ub = 12, then there is no way to satisfy the demand bc of the ub 
 tightness. The lb must take into account future links that do not come from a selected facility. Could sum these bounds
 up for all edges leaving a facility node until another facility node is encountered.
 Use the subgraph_from_interchange_nodes_geo method and include a <kwds_to_sum> param. for the attribute names to sum
 for combined edges (lb and ub), to do this with double nest dict structure, 
 use same principal as f = {k: d[k] + e[k] for k in d.keys() if k in e.keys()} ... does this work???

'''

'''
MCNF LP SOLVER
'''


def min_cost_flow(G: nx.DiGraph, key: str, suppress_output=True):
    # solve min cost flow for graph G and return G with solution (flows) as 'x' attribute for each edge

    m = gp.Model('Facility Sizing Problem', env=gurobi_suppress_output(suppress_output))

    e = list(G.edges)[0]
    print([G.edges[e][key]['c'], G.edges[e][key]['lb'], G.edges[e][key]['ub']])

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


'''
HELPER FUNCTIONS
'''


def selected_subgraph(G, time_step):
    path_nodes = {n for n in G if G.nodes[n]['covered'][time_step]}
    edge_set = {(u, v) for u, v in G.edges if G.edges[u, v]['covered'][time_step]}
    edges_to_remove = set(G.edges()).difference(set(edge_set))
    nodes_to_remove = set(G.nodes()).difference(set(path_nodes))
    # graph with feasible subnetwork(s) for coverage of technology
    H = remove_from_graph(G, nodes_to_remove=nodes_to_remove, edges_to_remove=edges_to_remove, connected_only=False)
    H.graph['time_step'] = time_step
    return H

