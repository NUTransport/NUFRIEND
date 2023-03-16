from util import *
# MODULES
from helper import elec_rate_state
from input_output import load_cached_graph, write_scenario_df, load_scenario_df, cache_graph, \
    codify_scenario_output_file, cache_exists
from network_representation import load_simplified_consolidated_graph
from routing import ods_by_perc_ton_mi, route_flows
from facility_deployment import facility_location
from facility_sizing import facility_sizing
from tea import tea_battery_all_facilities, tea_dropin, tea_hydrogen_all_facilities
from lca import lca_battery, lca_dropin, lca_hydrogen
from plotting import plot_scenario


# G, f, l = run_scenario('BNSF', 'battery', perc_ods=0.4, D=600*1.6, plot=True,
#                        load_scenario=False, deployment_table=True)


def run_scenario(rr: str = None, fuel_type: str = None, deployment_perc: float = None,
                 D: float = None, reroute: bool = None, switch_tech: bool = None, max_reroute_inc: float = None,
                 max_util: float = None, station_type: str = None, h2_fuel_type: str = None,
                 clean_energy=False, clean_energy_cost: float = None, emissions_obj=False,
                 CCWS_filename: str = None, perc_ods: float = None, comm_group: str = 'TOTAL',
                 time_window: tuple = None, freq: str = 'M',
                 eff_energy_p_tender: float = None,
                 suppress_output=True, binary_prog=True, select_cycles=False, max_flow=False, budget: int = None,
                 deviation_paths=True, extend_graph=True, od_flow_perc: float = 1,
                 G: nx.DiGraph = None, radius: float = None, intertypes: set = None,
                 scenario_code: str = None, deployment_table=False,
                 plot=True, load_scenario=True, cache_scenario=True, legend_show=True):
    # deployment_table = True means we are creating the deployment table and bypass the deployment_table lookup

    if not scenario_code:
        df_scenario = write_scenario_df(rr, fuel_type, deployment_perc,
                                        D, reroute, switch_tech, max_reroute_inc,
                                        max_util, station_type,
                                        clean_energy, clean_energy_cost, emissions_obj,
                                        CCWS_filename, perc_ods, comm_group,
                                        time_window, freq,
                                        eff_energy_p_tender,
                                        suppress_output, binary_prog,
                                        radius, intertypes, deployment_table)
        scenario_code = df_scenario['scenario_code']
    else:
        df_scenario = load_scenario_df(scenario_code=scenario_code)

    if not deployment_table and load_scenario:
        t0 = time.time()
        if cache_exists(scenario_code):
            G = load_cached_graph(scenario_code)
            print('LOADED GRAPH FROM CACHE:: %s seconds ---' % round(time.time() - t0, 3))
        else:
            G = run_scenario_df(df_scenario=df_scenario, G=G,
                                select_cycles=select_cycles, deviation_paths=deviation_paths,
                                max_flow=max_flow, budget=budget,
                                extend_graph=extend_graph, od_flow_perc=od_flow_perc)
            if cache_scenario:
                t0 = time.time()
                cache_graph(G=G, scenario_code=scenario_code)
                print('CACHE GRAPH:: %s seconds ---' % round(time.time() - t0, 3))
    else:
        if deployment_table:
            df_scenario['perc_ods'] = perc_ods
        G = run_scenario_df(df_scenario=df_scenario, G=G,
                            select_cycles=select_cycles, deviation_paths=deviation_paths,
                            max_flow=max_flow, budget=budget, extend_graph=extend_graph, od_flow_perc=od_flow_perc)

        if cache_scenario:
            t0 = time.time()
            cache_graph(G=G, scenario_code=scenario_code)
            print('CACHE GRAPH:: %s seconds ---' % round(time.time() - t0, 3))

    G = update_graph_values(G=G, fuel_type=df_scenario['fuel_type'], max_util=df_scenario['max_util'],
                            station_type=df_scenario['station_type'],
                            clean_energy=clean_energy, clean_energy_cost=clean_energy_cost,
                            h2_fuel_type=h2_fuel_type)

    if plot:
        t0 = time.time()
        fig, label = plot_scenario(G, fuel_type=df_scenario['fuel_type'],
                                   deployment_perc=df_scenario['deployment_perc'],
                                   comm_group=df_scenario['comm_group'], legend_show=legend_show)
        print('PLOTTING:: %s seconds ---' % round(time.time() - t0, 3))
    else:
        fig = None
        label = None

    return G, fig, label


def update_graph_values(G: nx.DiGraph, fuel_type: str, max_util: float, station_type: str,
                        clean_energy=True, clean_energy_cost: float = None,
                        h2_fuel_type: str = None):
    # valid <h2_fuel_type> values:
    # ['Natural Gas', 'NG with CO2 Sequestration', 'PEM Electrolysis - Solar', 'PEM Electrolysis - Nuclear']

    if not clean_energy:
        return G

    if clean_energy_cost is None:
        clean_energy_cost = 0.0

    # update TEA with premiums on clean energy costs and LCA with emissions cuts
    if fuel_type == 'battery':
        t0 = time.time()
        G = lca_battery(G=G, clean_energy=clean_energy)
        print('LCA2:: ' + str(time.time() - t0))

        t0 = time.time()
        G = tea_battery_all_facilities(G, max_util=max_util, station_type=station_type,
                                       clean_energy_cost=clean_energy_cost)
        print('TEA UPDATE:: ' + str(time.time() - t0))

    elif fuel_type == 'hydrogen':
        if h2_fuel_type is None:
            h2_fuel_type = 'Natural Gas'

        t0 = time.time()
        G = lca_hydrogen(G=G, h2_fuel_type=h2_fuel_type)
        print('LCA UPDATE:: ' + str(time.time() - t0))

        t0 = time.time()
        G = tea_hydrogen_all_facilities(G=G, max_util=max_util, station_type=station_type,
                                        clean_energy_cost=clean_energy_cost)
        print('TEA UPDATE:: ' + str(time.time() - t0))

    G = operations_stats(G)
    return G


def run_scenario_df(df_scenario: pd.DataFrame, G: nx.DiGraph = None,
                    select_cycles=False, deviation_paths=True,
                    max_flow=False, budget: int = None, extend_graph=True, od_flow_perc: float = 1):
    t0_total = time.time()

    if isinstance(df_scenario, pd.DataFrame):
        df_scenario = df_scenario['Value']

    idxs = ['rr', 'fuel_type', 'deployment_perc',
            'D', 'reroute', 'switch_tech', 'max_reroute_inc',
            'max_util', 'station_type',
            'clean_energy', 'clean_energy_cost', 'emissions_obj',
            'CCWS_filename', 'perc_ods', 'comm_group',
            'time_window_start', 'time_window_end', 'freq',
            'eff_energy_p_tender',
            'suppress_output', 'binary_prog',
            'radius', 'intertypes',
            'scenario_code']

    [rr, fuel_type, deployment_perc,
     D, reroute, switch_tech, max_reroute_inc,
     max_util, station_type,
     clean_energy, _, emissions_obj,
     CCWS_filename, perc_ods, comm_group,
     time_window_start, time_window_end, freq,
     eff_energy_p_tender,
     suppress_output, binary_prog,
     radius, intertypes,
     scenario_code] = df_scenario.reindex(idxs)

    # deployment_perc = float(deployment_perc)
    # D = float(D)
    # reroute = int(reroute)
    # switch_tech = int(switch_tech)
    # max_reroute_inc = float(max_reroute_inc)
    # max_util = float(max_util)
    # clean_energy = int(clean_energy)
    # clean_energy_cost = float(clean_energy_cost)
    # emissions_obj = int(emissions_obj)
    # eff_energy_p_tender = float(eff_energy_p_tender)
    # suppress_output = int(suppress_output)
    # binary_prog = int(binary_prog)
    # radius = float(radius)

    time_window = (time_window_start, time_window_end)

    # 0. load railroad network representation as a nx.Graph and a simplify and consolidate network
    if not G:
        G = load_simplified_consolidated_graph(rr, radius=radius, intertypes=intertypes)
    else:
        # get a deep copy of G so that changes made to local G are not made to the original G
        G = deepcopy(G)

    if fuel_type == 'battery':
        G.graph['scenario'] = dict(railroad=rr, range_mi=D * KM2MI, fuel_type=fuel_type,
                                   desired_deployment_perc=deployment_perc, reroute=reroute,
                                   switch_tech=switch_tech,
                                   max_reroute_inc=max_reroute_inc, max_util=max_util, station_type=station_type,
                                   eff_kwh_p_batt=eff_energy_p_tender, scenario_code=scenario_code)
        # 1. load od_flow_dict for ranking OD pairs and choosing top <perc_ods> for flows for facility location
        t0 = time.time()
        if perc_ods is None or perc_ods == 'X':
            perc_ods = deployment_perc_lookup_table(df_scenario=df_scenario, deployment_perc=deployment_perc)
        print('LOOKUP TABLE:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        if select_cycles:
            # select almost all O-D pairs with non-zero flow (leave out the 20% with the lowest flow values; too many)
            ods, od_flows = ods_by_perc_ton_mi(G=G, perc_ods=od_flow_perc, CCWS_filename=CCWS_filename)
        if not select_cycles:
            ods, od_flows = ods_by_perc_ton_mi(G=G, perc_ods=perc_ods, CCWS_filename=CCWS_filename)
        G.graph['framework'] = dict(ods=ods)
        print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 2. locate facilities and extract graph form of this, G, and its induced subgraph, H
        G, H = facility_location(G, D=D, ods=ods, od_flows=od_flows, flow_min=perc_ods, select_cycles=select_cycles,
                                 budget=budget, max_flow=max_flow, extend_graph=extend_graph, od_flow_perc=od_flow_perc,
                                 deviation_paths=deviation_paths,
                                 binary_prog=binary_prog, suppress_output=suppress_output)
        print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))

        # if no facilities are selected
        if G.graph['number_facilities'] == 0:
            return G

        t0 = time.time()
        # 3. reroute flows and get peak and average ton and locomotive flows for each edge
        G, H = route_flows(G=G, fuel_type=fuel_type, H=H, D=D, CCWS_filename=CCWS_filename,
                           time_window=time_window, freq=freq,
                           reroute=reroute, switch_tech=switch_tech, max_reroute_inc=max_reroute_inc)
        print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 4. facility sizing based on peak flows and utilization based on average flows
        # load cost by state dataframe and assign to each node
        # emissions_p_location = elec_rate_state(G, emissions=True, clean_energy=clean_energy,
        #                                        clean_elec_prem_dolkwh=clean_energy_cost)  # [gCO2/kWh]
        # cost_p_location = elec_rate_state(G, clean_energy=clean_energy,
        #                                   clean_elec_prem_dolkwh=clean_energy_cost)  # in [$/MWh]

        G = facility_sizing(G=G, H=H, fuel_type=fuel_type, D=D, emissions_obj=emissions_obj,
                            suppress_output=suppress_output)
        print('FACILITY SIZING:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        actual_dep_perc = G.graph['operations']['deployment_perc'][comm_group]
        G.graph['scenario']['actual_deployment_perc'] = actual_dep_perc
        # 5.1. TEA
        G = tea_battery_all_facilities(G, max_util=max_util, station_type=station_type)
        # baseline and other dropin fuels (easy factor calculation)
        G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
        # G = tea_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
        # G = tea_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
        print('TEA:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 5.2. LCA
        G = lca_battery(G, clean_energy=clean_energy)
        # baseline and other dropin fuels (easy factor calculation)
        G = lca_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
        # G = lca_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
        # G = lca_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
        print('LCA:: %s seconds ---' % round(time.time() - t0, 3))

    elif fuel_type == 'hydrogen':
        G.graph['scenario'] = dict(railroad=rr, range_mi=D * KM2MI, fuel_type=fuel_type,
                                   desired_deployment_perc=deployment_perc, reroute=reroute,
                                   switch_tech=switch_tech,
                                   max_reroute_inc=max_reroute_inc, max_util=max_util, station_type=station_type,
                                   eff_kgh2_p_loc=eff_energy_p_tender, scenario_code=scenario_code)
        # 1. load od_flow_dict for ranking OD pairs and choosing top <perc_ods> for flows for facility location
        t0 = time.time()
        if perc_ods is None or perc_ods == 'X':
            perc_ods = deployment_perc_lookup_table(df_scenario=df_scenario, deployment_perc=deployment_perc)
            # perc_ods = deployment_perc_lookup_table(filename=scenario_filename, deployment_perc=deployment_perc)
        print('LOOKUP TABLE:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        ods = ods_by_perc_ton_mi(G=G, perc_ods=perc_ods, CCWS_filename=CCWS_filename)
        G.graph['framework'] = dict(ods=ods)
        print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 2. locate facilities and extract graph form of this, G, and its induced subgraph, H
        G, H = facility_location(G, D=D, ods=ods, binary_prog=binary_prog, suppress_output=suppress_output)
        print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))

        # if no facilities are selected
        if G.graph['number_facilities'] == 0:
            return G

        t0 = time.time()
        # 3. reroute flows and get peak and average ton and locomotive flows for each edge
        G, H = route_flows(G=G, fuel_type=fuel_type, H=H, D=D, CCWS_filename=CCWS_filename,
                           time_window=time_window, freq=freq,
                           reroute=reroute, switch_tech=switch_tech, max_reroute_inc=max_reroute_inc)
        print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 4. facility sizing based on peak flows and utilization based on average flows
        G = facility_sizing(G=G, H=H, fuel_type=fuel_type, D=D, unit_sizing_obj=True, suppress_output=suppress_output)
        print('FACILITY SIZING:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        actual_dep_perc = G.graph['operations']['deployment_perc'][comm_group]
        G.graph['scenario']['actual_deployment_perc'] = actual_dep_perc
        # 5.1. TEA
        G = tea_hydrogen_all_facilities(G, max_util=max_util, station_type=station_type)
        # baseline and other dropin fuels (easy factor calculation)
        G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
        G = tea_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
        G = tea_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
        print('TEA:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 5.2. LCA
        G = lca_hydrogen(G)
        # baseline and other dropin fuels (easy factor calculation)
        G = lca_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
        G = lca_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
        G = lca_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
        print('LCA:: %s seconds ---' % round(time.time() - t0, 3))

    elif fuel_type == 'e-fuel' or fuel_type == 'biodiesel' or fuel_type == 'diesel':
        if deployment_perc is None:
            deployment_perc = 1

        G.graph['scenario'] = dict(railroad=rr, range_mi=np.nan, fuel_type=fuel_type,
                                   desired_deployment_perc=deployment_perc, reroute=reroute,
                                   switch_tech=switch_tech,
                                   max_reroute_inc=max_reroute_inc, max_util=max_util, station_type=station_type,
                                   scenario_code=scenario_code)

        t0 = time.time()
        # 1. route baseline flows to get average daily ton and locomotive flows for each edge
        G = route_flows(G=G, fuel_type=fuel_type, CCWS_filename=CCWS_filename, time_window=time_window)
        print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 2. TEA
        G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
        G = tea_dropin(G, fuel_type=fuel_type, deployment_perc=deployment_perc)
        print('TEA:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 3. LCA
        G = lca_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
        G = lca_dropin(G, fuel_type=fuel_type, deployment_perc=deployment_perc)
        print('LCA:: %s seconds ---' % round(time.time() - t0, 3))

    G = operations_stats(G)
    print('SCENARIO RUN:: %s seconds ---' % round(time.time() - t0_total, 3))

    return G


def operations_stats(G: nx.DiGraph) -> nx.DiGraph:
    # compute the operational stats of solution in G (many relative to diesel baseline)
    comm_list = list({c for u, v in G.edges for c in G.edges[u, v]['baseline_avg_ton'].keys()})
    if G.graph['scenario']['fuel_type'] == 'battery':
        # G.graph['operations'].update(
        #     dict(
        #         emissions_change=100 * ((G.graph['diesel_LCA']['annual_total_emissions_tonco2'][comm_group] -
        #                                  G.graph['energy_source_LCA']['annual_total_emissions_tonco2']) /
        #                                 G.graph['diesel_LCA']['annual_total_emissions_tonco2'][comm_group]),
        #         cost_avoided_emissions=-1e-3 * ((G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'] -
        #                                          G.graph['diesel_TEA']['total_LCO_tonmi']) /
        #                                         (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'] -
        #                                          G.graph['diesel_LCA']['total_emissions_tonco2_tonmi']))
        #     )
        # )
        G.graph['operations'].update(
            dict(
                emissions_change=dict(zip(
                    comm_list,
                    [100 * (G.graph['diesel_LCA']['annual_total_emissions_tonco2'][c] -
                            G.graph['energy_source_LCA']['annual_total_emissions_tonco2'][c]) /
                     G.graph['diesel_LCA']['annual_total_emissions_tonco2'][c] for c in comm_list])),
                cost_avoided_emissions=dict(zip(
                    comm_list,
                    [-1e-3 * (G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][c] -
                              G.graph['diesel_TEA']['total_LCO_tonmi'][c]) /
                     (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'][c] -
                      G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list]))
            ))
    elif G.graph['scenario']['fuel_type'] == 'hydrogen':
        # G.graph['operations'].update(
        #     dict(
        #         emissions_change=100 * ((G.graph['diesel_LCA']['annual_total_emissions_tonco2'] -
        #                                  G.graph['energy_source_LCA']['annual_total_emissions_tonco2']) /
        #                                 G.graph['diesel_LCA']['annual_total_emissions_tonco2']),
        #         cost_avoided_emissions=-1e-3 * ((G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'] -
        #                                          G.graph['diesel_TEA']['total_LCO_tonmi']) /
        #                                         (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'] -
        #                                          G.graph['diesel_LCA']['total_emissions_tonco2_tonmi']))
        #     )
        # )
        G.graph['operations'].update(
            dict(
                emissions_change=dict(zip(
                    comm_list,
                    [100 * (G.graph['diesel_LCA']['annual_total_emissions_tonco2'][c] -
                            G.graph['energy_source_LCA']['annual_total_emissions_tonco2'][c]) /
                     G.graph['diesel_LCA']['annual_total_emissions_tonco2'][c] for c in comm_list])),
                cost_avoided_emissions=dict(zip(
                    comm_list,
                    [-1e-3 * (G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][c] -
                              G.graph['diesel_TEA']['total_LCO_tonmi'][c]) /
                     (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'][c] -
                      G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list]))
            ))
    elif G.graph['scenario']['fuel_type'] == 'e-fuel' or G.graph['scenario']['fuel_type'] == 'biodiesel':
        # G.graph['operations'].update(
        #     dict(
        #         emissions_change=100 * ((G.graph['diesel_LCA']['annual_total_emissions_tonco2'] -
        #                                  G.graph['energy_source_LCA']['annual_total_emissions_tonco2']) /
        #                                 G.graph['diesel_LCA']['annual_total_emissions_tonco2']),
        #         cost_avoided_emissions=-1e-3 * ((G.graph['energy_source_TEA']['total_LCO_tonmi'] -
        #                                          G.graph['diesel_TEA']['total_LCO_tonmi']) /
        #                                         (G.graph['energy_source_LCA']['total_emissions_tonco2_tonmi'] -
        #                                          G.graph['diesel_LCA']['total_emissions_tonco2_tonmi']))
        #     )
        # )
        G.graph['operations'].update(
            dict(
                emissions_change=dict(zip(
                    comm_list,
                    [100 * (G.graph['diesel_LCA']['annual_total_emissions_tonco2'][c] -
                            G.graph['energy_source_LCA']['annual_total_emissions_tonco2'][c]) /
                     G.graph['diesel_LCA']['annual_total_emissions_tonco2'][c] for c in comm_list])),
                cost_avoided_emissions=dict(zip(
                    comm_list,
                    [-1e-3 * (G.graph['energy_source_TEA']['total_LCO_tonmi'][c] -
                              G.graph['diesel_TEA']['total_LCO_tonmi'][c]) /
                     (G.graph['energy_source_LCA']['total_emissions_tonco2_tonmi'][c] -
                      G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list]))
            ))

    return G


'''
DEPLOYMENT TABLES
'''


def deployment_perc_stats(scenario_code: str):
    # return percentage of ods needed to be routed to produce a desired deployment_perc for given scenario
    # may need more scenario params passed to here i.e., range, reroute, switch_tech, max_reroute_inc
    # store in files for easy access

    filepath = os.path.join(DEP_TAB_O_DIR, scenario_code + '.csv')
    if not os.path.exists(filepath):
        perc_ods = [round(i, 2) for i in np.linspace(0, 1, 51)]
        # perc_ods = [round(i, 2) for i in np.linspace(0, 1, 21)]
        # perc_ods = [round(i, 2) for i in np.linspace(0, 1, 5)]
        # perc_ods = [round(i, 2) for i in np.linspace(0, 1, 101)]
        # perc_ods = [round(i, 3) for i in np.linspace(0, 1, 201)]
        # perc_ods = list(np.linspace(0, .2, 11)) + [round(i, 2) for i in np.linspace(.22, 0.5, 15)] + \
        #            [round(i, 2) for i in np.linspace(0.5, 1, 11)]
        # extract fuel_type key value from codified name
        df_scenario = load_scenario_df(scenario_code=scenario_code)
        # fuel_type = get_val_from_code(scenario_code=scenario_code, key='fuel_type')
        fuel_type = df_scenario['fuel_type']
        G_orig = load_simplified_consolidated_graph(df_scenario['rr'])

        if fuel_type == 'battery':
            cols = ['deployment_perc', 'station_total', 'station_annual_cost', 'battery_annual_cost',
                    'station_LCO_tonmi', 'battery_LCO_tonmi', 'om_LCO_tonmi', 'energy_LCO_tonmi', 'delay_LCO_tonmi',
                    'total_LCO_tonmi', 'total_scenario_LCO_tonmi',
                    'actual_utilization', 'number_chargers', 'number_facilities',
                    'avg_queue_time_p_loc', 'avg_queue_length', 'peak_queue_time_p_loc', 'peak_queue_length',
                    'avg_daily_delay_cost_p_car', 'total_annual_delay_cost', 'battery_emissions_tonco2_tonmi',
                    'avg_emissions_tonco2_tonmi', 'scenario_annual_emissions_tonco2', 'alt_tech_annual_tonmi',
                    'scenario_annual_tonmi', 'emissions_change', 'cost_avoided_emissions_$_kg',
                    'inflation_tonmi', 'inflation_mi', 'baseline_annual_tonmi', 'baseline_annual_emissions_tonco2']
        elif fuel_type == 'hydrogen':
            cols = ['deployment_perc', 'station_total', 'station_annual_cost',
                    'station_LCO_tonmi', 'terminal_LCO_tonmi', 'liquefier_LCO_tonmi', 'fuel_LCO_tonmi',
                    'tender_LCO_tonmi', 'delay_LCO_tonmi', 'total_LCO_tonmi', 'total_scenario_LCO_tonmi',
                    'actual_utilization', 'number_pumps', 'number_facilities',
                    'avg_queue_time_p_loc', 'avg_queue_length', 'peak_queue_time_p_loc', 'peak_queue_length',
                    'avg_daily_delay_cost_p_car', 'total_annual_delay_cost', 'hydrogen_emissions_tonco2_tonmi',
                    'avg_emissions_tonco2_tonmi', 'scenario_annual_emissions_tonco2', 'alt_tech_annual_tonmi',
                    'scenario_annual_tonmi', 'emissions_change', 'cost_avoided_emissions_$_kg',
                    'inflation_tonmi', 'inflation_mi', 'baseline_annual_tonmi', 'baseline_annual_emissions_tonco2']

        df = pd.DataFrame(data=0, columns=cols, index=perc_ods)
        df.index.rename('perc_ods', inplace=True)
        # df.loc[1, 'deployment_perc'] = 100
        for p_od in perc_ods[1:]:
            print(p_od)
            # try:
            G, _, _ = run_scenario(perc_ods=p_od, scenario_code=scenario_code, G=G_orig, deployment_table=True,
                                   cache_scenario=False, plot=False, load_scenario=False)

            if G.graph['number_facilities'] == 0:
                df.loc[p_od, cols] = [np.zeros(len(cols))]
            else:
                if fuel_type == 'battery':
                    df.loc[p_od, cols] = [100 * G.graph['operations']['deployment_perc']['TOTAL'],
                                          G.graph['energy_source_TEA']['station_total'],
                                          G.graph['energy_source_TEA']['station_annual_cost'],
                                          G.graph['energy_source_TEA']['battery_annual_cost']['TOTAL'],
                                          G.graph['energy_source_TEA']['station_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['battery_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['om_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['energy_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['delay_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['total_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['total_scenario_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['actual_utilization'],
                                          G.graph['energy_source_TEA']['number_chargers'],
                                          G.graph['number_facilities'],
                                          G.graph['energy_source_TEA']['avg_queue_time_p_loc'],
                                          G.graph['energy_source_TEA']['avg_queue_length'],
                                          G.graph['energy_source_TEA']['peak_queue_time_p_loc'],
                                          G.graph['energy_source_TEA']['peak_queue_length'],
                                          G.graph['energy_source_TEA']['avg_daily_delay_cost_p_car'],
                                          G.graph['energy_source_TEA']['total_annual_delay_cost'],
                                          G.graph['energy_source_LCA']['battery_emissions_tonco2_tonmi']['TOTAL'],
                                          G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi']['TOTAL'],
                                          G.graph['energy_source_LCA']['annual_total_emissions_tonco2']['TOTAL'],
                                          G.graph['operations']['alt_tech_total_annual_tonmi']['TOTAL'],
                                          G.graph['operations']['scenario_total_annual_tonmi']['TOTAL'],
                                          G.graph['operations']['emissions_change']['TOTAL'],
                                          G.graph['operations']['cost_avoided_emissions']['TOTAL'],
                                          G.graph['operations']['perc_tonmi_inc']['TOTAL'],
                                          G.graph['operations']['perc_mi_inc']['TOTAL'],
                                          G.graph['operations']['baseline_total_annual_tonmi']['TOTAL'],
                                          G.graph['diesel_LCA']['annual_total_emissions_tonco2']['TOTAL']]
                elif fuel_type == 'hydrogen':
                    df.loc[p_od, cols] = [100 * G.graph['operations']['deployment_perc']['TOTAL'],
                                          G.graph['energy_source_TEA']['station_total'],
                                          G.graph['energy_source_TEA']['station_annual_cost'],
                                          G.graph['energy_source_TEA']['station_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['terminal_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['liquefier_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['fuel_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['tender_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['delay_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['total_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['total_scenario_LCO_tonmi']['TOTAL'],
                                          G.graph['energy_source_TEA']['actual_utilization'],
                                          G.graph['energy_source_TEA']['number_pumps'],
                                          G.graph['number_facilities'],
                                          G.graph['energy_source_TEA']['avg_queue_time_p_loc'],
                                          G.graph['energy_source_TEA']['avg_queue_length'],
                                          G.graph['energy_source_TEA']['peak_queue_time_p_loc'],
                                          G.graph['energy_source_TEA']['peak_queue_length'],
                                          G.graph['energy_source_TEA']['avg_daily_delay_cost_p_car'],
                                          G.graph['energy_source_TEA']['total_annual_delay_cost'],
                                          G.graph['energy_source_LCA']['hydrogen_emissions_tonco2_tonmi']['TOTAL'],
                                          G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi']['TOTAL'],
                                          G.graph['energy_source_LCA']['annual_total_emissions_tonco2']['TOTAL'],
                                          G.graph['operations']['alt_tech_total_annual_tonmi']['TOTAL'],
                                          G.graph['operations']['scenario_total_annual_tonmi']['TOTAL'],
                                          G.graph['operations']['emissions_change']['TOTAL'],
                                          G.graph['operations']['cost_avoided_emissions']['TOTAL'],
                                          G.graph['operations']['perc_tonmi_inc']['TOTAL'],
                                          G.graph['operations']['perc_mi_inc']['TOTAL'],
                                          G.graph['operations']['baseline_total_annual_tonmi']['TOTAL'],
                                          G.graph['diesel_LCA']['annual_total_emissions_tonco2']['TOTAL']]

        # for each row, want a one-to-one mapping in increasing order only
        df.drop_duplicates(subset=['deployment_perc'], keep='last', inplace=True)
        df.sort_index(inplace=True)
        df.reset_index(inplace=True)
        df['perc_ods'] = 100 * df['perc_ods']  # convert to percentage
        idxs_keep = []
        for i in range(len(df) - 1):
            if df.loc[i + 1, 'deployment_perc'] > df.loc[i, 'deployment_perc'] > 0:
                idxs_keep.append(i)
        idxs_keep.append(len(df) - 1)
        df = df.loc[idxs_keep]

        df.to_csv(filepath, index=False)

    return df


def deployment_perc_lookup_table(df_scenario: pd.DataFrame, deployment_perc: float):

    scenario_code = codify_scenario_output_file(df_scenario=df_scenario, deployment_table=True)
    filepath = os.path.join(DEP_TAB_O_DIR, scenario_code + '.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, header=0)
    else:
        df_scenario['deployment_perc'] = 0
        df_scenario.reset_index(level=['Keyword']).to_csv(os.path.join(SCENARIO_DIR, scenario_code + '.csv'),
                                                          index=False)
        df = deployment_perc_stats(scenario_code)

    # TODO: check what is going on here, correct min ("interpolation")?
    # keep only these columns of interest
    df = df[['perc_ods', 'deployment_perc']] / 100  # convert from percentages to rates

    if deployment_perc >= df['deployment_perc'].max():
        return df.loc[df.index[-1], 'perc_ods']
    else:
        # dfi = df.copy()
        # dfi.index = dfi['perc_ods']
        return df[df['deployment_perc'] >= deployment_perc]['perc_ods'].min()


def cache_everything():
    RR_list = ['BNSF', 'CN', 'CP', 'CSXT', 'KCS', 'NS', 'UP', 'WCAN', 'EAST', 'USA1']
    range_list = [200, 400, 800, 1000, 1500]
    deploy_pct_list = [0.2, 0.4, 0.6, 0.8, 1]

    for ES_name in ["battery", 'diesel', 'biodiesel', 'e-fuel']:
        # for RR_name in ["BNSF", "UP", "NS", "KCS", "CSXT", "CN", "CP"]:
        for RR_name in RR_list: # cache every RR
            if ES_name == 'battery':
                for RR_range in range_list:
                    for deploy_pct in deploy_pct_list:
                        print(RR_name, ES_name, RR_range, deploy_pct)
                        run_scenario(rr=RR_name, fuel_type=ES_name, comm_group="TOTAL", D=1.6 * RR_range,
                                     deployment_perc=deploy_pct,
                                     plot=False, cache_scenario=True, load_scenario=False)
            elif ES_name == 'diesel':
                print(RR_name, ES_name, 0, 1)
                run_scenario(rr=RR_name, fuel_type=ES_name, comm_group="TOTAL", D=0, deployment_perc=1,
                             plot=False, cache_scenario=True, load_scenario=False)
            else:
                for deploy_pct in deploy_pct_list:
                    print(RR_name, ES_name, 0, deploy_pct)
                    run_scenario(rr=RR_name, fuel_type=ES_name, comm_group="TOTAL", D=0, deployment_perc=deploy_pct,
                                 plot=False, cache_scenario=True, load_scenario=False)
    return ["Cached!"]


'''
PLOT STATS
'''


def plot_stats(scenario_code: str, figshow=False):
    df_scenario = load_scenario_df(scenario_code=scenario_code)
    fuel_type = df_scenario['fuel_type']

    filename = scenario_code + '.csv'
    filepath = os.path.join(DEP_TAB_O_DIR, filename)
    df = pd.read_csv(filepath, header=0)

    df.dropna(inplace=True)
    idxs_keep = []
    for i in df.index:
        if df.loc[i, 'deployment_perc'] > 0:
            idxs_keep.append(i)
    df = df.loc[idxs_keep]
    # df['deployment_perc'] = df['deployment_perc'] * 100
    # df['perc_ods'] = df['perc_ods'] * 100

    [red, blue, grey, purple, yellow, green] = ['tab:red', 'tab:blue', 'tab:gray', 'tab:purple', 'orange', 'green']

    # matplotlib.rcParams.update({'font.size': 7})
    sns.set_theme(palette='Set2', font_scale=0.5)

    # SECTION 1: ALL PLOTS
    fig, ax = plt.subplots(4, 3, figsize=(20, 8))
    ((a00, a01, a02), (a10, a11, a12), (a20, a21, a22), (a30, a31, a32)) = ax
    # fig, ax = plt.subplots(3, 3, figsize=(20, 8))
    # ((a00, a01, a02), (a10, a11, a12), (a20, a21, a22)) = ax

    a00.set_title('Deployment % vs. Top % of OD pairs selected')
    a00.set_xlabel('Top % [ton-mi] of OD pairs selected')
    a00.set_ylabel('Deployment % [ton-mi]')
    a00.plot(df['perc_ods'], df['deployment_perc'], '.-', color=blue)

    a10.set_title('% Emissions Reduction vs. Deployment %')
    a10.set_xlabel('Deployment % [ton-mi]')
    a10.set_ylabel('% Emissions Reduction')
    a10.plot(df['deployment_perc'], df['emissions_change'], '.-', color=blue)

    a20.set_title('% Average Shipment Distance Increase vs. Deployment %')
    a20.set_xlabel('Deployment % [ton-mi]')
    a20.set_ylabel('% Increase')
    a20.plot(df['deployment_perc'], df['inflation_mi'], '-', color=blue, label='Distance')

    a01.set_title('Total Station Capital Cost vs. Deployment %')
    a01.set_xlabel('Deployment % [ton-mi]')
    a01.set_ylabel('Total Station Capital Cost [$M]')
    a01.plot(df['deployment_perc'], df['station_total'] * 1e-6, '.-', label='Station Capital', color=blue)
    a01.plot(df['deployment_perc'], df['station_annual_cost'] * 1e-6, '.-', label='Station Annual', color=purple)
    if fuel_type == 'battery':
        a01.plot(df['deployment_perc'], df['battery_annual_cost'] * 1e-6, '.-',
                 label='Battery Annual', color=yellow)
        a01.legend(bbox_to_anchor=(0.5, 1.02), loc='upper center', fontsize=5, ncol=3)
    else:
        a01.legend(bbox_to_anchor=(0.5, 1.02), loc='upper center', fontsize=5, ncol=2)

    if fuel_type == 'battery':
        a11.set_title('Number of Facilities and Chargers vs. Deployment %')
        a11.set_xlabel('Deployment % [ton-mi]')
        a11.set_ylabel('Number of Chargers', color=blue)
        a11.plot(df['deployment_perc'], df['number_chargers'], '.-', color=blue)
    else:
        a11.set_title('Number of Facilities and Pumps vs. Deployment %')
        a11.set_xlabel('Deployment % [ton-mi]')
        a11.set_ylabel('Number of Pumps', color=blue)
        a11.plot(df['deployment_perc'], df['number_pumps'], '.-', color=blue)
    a11.tick_params(axis='y', labelcolor=blue)
    a11r = a11.twinx()  # second y axes
    a11r.set_ylabel('Number of Facilities', color=red)
    a11r.plot(df['deployment_perc'], df['number_facilities'], '.-', color=red)
    a11r.tick_params(axis='y', labelcolor=red)

    a21.set_title('Station Utilization vs. Deployment %')
    a21.set_xlabel('Deployment % [ton-mi]')
    a21.set_ylabel('Station Utilization [hrs/day]')
    a21.plot(df['deployment_perc'], df['actual_utilization'] * 24, '.-', color=blue)

    a02.set_title('Levelized Cost vs. Deployment %')
    a02.set_xlabel('Deployment % [ton-mi]')
    a02.set_ylabel('Levelized Cost [$/ton-mi]')
    if fuel_type == 'battery':
        a02.plot(df['deployment_perc'], df['station_LCO_tonmi'], '.-', label='Station', color=blue)
        a02.plot(df['deployment_perc'], df['energy_LCO_tonmi'], '.-', label='Energy', color=red)
        a02.plot(df['deployment_perc'], df['battery_LCO_tonmi'], '.-', label='Battery', color=yellow)
        a02.plot(df['deployment_perc'], df['delay_LCO_tonmi'], '.-', label='Delay', color=green)
        a02.plot(df['deployment_perc'], df['total_LCO_tonmi'], '.-', label='Total', color=grey)
        a02.plot(df['deployment_perc'], df['total_scenario_LCO_tonmi'], '.-', color=purple, label='Total Scenario')
        a02.legend(bbox_to_anchor=(0.5, 1.02), loc='upper center', fontsize=5, ncol=6)
    elif fuel_type == 'hydrogen':
        a02.plot(df['deployment_perc'], df['station_LCO_tonmi'] + df['terminal_LCO_tonmi'] + df['liquefier_LCO_tonmi'],
                 '.-', label='Station', color=blue)
        a02.plot(df['deployment_perc'], df['fuel_LCO_tonmi'], '.-', label='Fuel', color=red)
        a02.plot(df['deployment_perc'], df['tender_LCO_tonmi'], '.-', label='Tender', color=yellow)
        a02.plot(df['deployment_perc'], df['delay_LCO_tonmi'], '.-', label='Delay', color=green)
        a02.plot(df['deployment_perc'], df['total_LCO_tonmi'], '.-', label='Total', color=grey)
        a02.plot(df['deployment_perc'], df['total_scenario_LCO_tonmi'], '.-', color=purple, label='Total Scenario')
        a02.legend(bbox_to_anchor=(0.5, 1.02), loc='upper center', fontsize=5, ncol=6)

    a12.set_title('Levelized Emissions vs. Deployment %')
    a12.set_xlabel('Deployment % [ton-mi]')
    a12.set_ylabel('Levelized Emissions [ton-CO_2/ton-mi]')
    if fuel_type == 'battery':
        a12.plot(df['deployment_perc'], df['battery_emissions_tonco2_tonmi'], '.-', color=blue, label='Battery')
    elif fuel_type == 'hydrogen':
        a12.plot(df['deployment_perc'], df['hydrogen_emissions_tonco2_tonmi'], '.-', color=blue, label='Hydrogen')
    a12.plot(df['deployment_perc'], df['avg_emissions_tonco2_tonmi'], '.-', color=red, label='Scenario')
    a12.legend(bbox_to_anchor=(0.5, 1.02), loc='upper center', fontsize=5, ncol=2)

    a22.set_title('Cost of Avoided Emissions vs. Deployment %')
    a22.set_xlabel('Deployment % [ton-mi]')
    a22.set_ylabel('Cost of Avoided Emissions [$/kg CO_2]')
    a22.plot(df['deployment_perc'], df['cost_avoided_emissions_$_kg'], '.-', color=blue)

    a30.set_title('Annual Emissions vs. Deployment %')
    a30.set_xlabel('Deployment % [ton-mi]')
    a30.set_ylabel('Annual Emissions [kt CO_2]')
    a30.plot(df['deployment_perc'], df['scenario_annual_emissions_tonco2'] * 1e-3, '.-', color=blue, label='Scenario')
    a30.plot(df['deployment_perc'], df['baseline_annual_emissions_tonco2'] * 1e-3, '.-', color=red, label='Baseline')
    a30.legend(bbox_to_anchor=(0.5, 1.02), loc='upper center', fontsize=5, ncol=2)

    a31.set_title('Annual Scenario Ton-Miles vs. Deployment %')
    a31.set_xlabel('Deployment % [ton-mi]')
    a31.set_ylabel('Annual Scenario Ton-Miles [ton-mi]')
    a31.plot(df['deployment_perc'], df['scenario_annual_tonmi'] * (1 - df['inflation_tonmi'] / 100), '.-',
             color=blue, label='Scenario-Baseline Eq.')
    a31.plot(df['deployment_perc'], df['scenario_annual_tonmi'], '.-', color=red, label='Scenario Actual')
    a31.plot(df['deployment_perc'], df['baseline_annual_tonmi'], '.-', color=grey, label='Baseline')
    a31.legend(bbox_to_anchor=(0.5, 1.02), loc='upper center', fontsize=5, ncol=3)

    # compute gross emissions reduction amount and total and marginal cost curves
    df['gross_emissions_change_tonco2'] = (df['baseline_annual_emissions_tonco2'] -
                                           df['scenario_annual_emissions_tonco2'])
    df['total_cost_avoided_emissions'] = 1e3 * df['cost_avoided_emissions_$_kg'] * df['gross_emissions_change_tonco2']
    # get only positive values of gross emissions change and cost of avoided emissions for the fitting
    df['gross_emissions_change_tonco2'] = df[(df['gross_emissions_change_tonco2'] > 0) &
                                             (df['cost_avoided_emissions_$_kg'] > 0)]['gross_emissions_change_tonco2']
    df['total_cost_avoided_emissions'] = df[(df['gross_emissions_change_tonco2'] > 0) &
                                            (df['cost_avoided_emissions_$_kg'] > 0)]['total_cost_avoided_emissions']
    dfp = df[['gross_emissions_change_tonco2', 'total_cost_avoided_emissions']].copy().dropna()
    # for a polynomial fit of the form: TC = a * GE^3 + b * GE^2 + c * GE + d
    (d, c, b, a) = np.polynomial.polynomial.polyfit(dfp['gross_emissions_change_tonco2'],
                                                    dfp['total_cost_avoided_emissions'], 3)
    (amcp, bmcp, cmcp) = [a * 3, b * 2, c]
    # df['total_cost_avoided_emissions_poly'] = 1e3 * (a * df['gross_emissions_change_tonco2'] ** 3 +
    #                                                  b * df['gross_emissions_change_tonco2'] ** 2 +
    #                                                  c * df['gross_emissions_change_tonco2'] + d)
    # df['average_cost_avoided_emissions_poly'] = 1e-3 * (df['total_cost_avoided_emissions_poly'] /
    #                                                     df['gross_emissions_change_tonco2'])
    dfp['marginal_cost_avoided_emissions_$_kg_poly'] = 1e-3 * (amcp * dfp['gross_emissions_change_tonco2'] ** 2 +
                                                               bmcp * dfp['gross_emissions_change_tonco2'] + cmcp)
    # for an exponential fit of the form: TC = a * b^GE ==> ln(TC) = ln(a) + ln(b) * GE
    # weight for each value of y be its magnitude for better fit
    (lna, lnb) = np.polynomial.polynomial.polyfit(dfp['gross_emissions_change_tonco2'],
                                                  np.log(dfp['total_cost_avoided_emissions']), 1,
                                                  w=np.sqrt(dfp['total_cost_avoided_emissions']))
    (amce, bmce) = [np.exp(lna) * lnb, np.exp(lnb)]
    # df['total_cost_avoided_emissions_exp'] = 1e3 * (a * df['gross_emissions_change_tonco2'] ** 3 +
    #                                                  b * df['gross_emissions_change_tonco2'] ** 2 +
    #                                                  c * df['gross_emissions_change_tonco2'] + d)
    # df['average_cost_avoided_emissions_exp'] = 1e-3 * (df['total_cost_avoided_emissions_poly'] /
    #                                                     df['gross_emissions_change_tonco2'])
    dfp['marginal_cost_avoided_emissions_$_kg_exp'] = 1e-3 * amce * (bmce ** dfp['gross_emissions_change_tonco2'])

    a32.set_title('Cost of Avoided Emissions vs. Gross Emissions Reduced')
    a32.set_xlabel('Gross Emissions Reduced [kt CO_2]')
    a32.set_ylabel('Cost of Avoided Emissions [$/kg CO_2]')
    a32.plot(df['gross_emissions_change_tonco2'] * 1e-3, df['cost_avoided_emissions_$_kg'],
             'd', color=grey, label='AC')
    # a32.plot(dfp['gross_emissions_change_tonco2'] * 1e-3, dfp['marginal_cost_avoided_emissions_$_kg_poly'],
    #          '+', color=red, markersize=7, label='MC-Poly Fit')
    # a32.plot(dfp['gross_emissions_change_tonco2'] * 1e-3, dfp['marginal_cost_avoided_emissions_$_kg_exp'],
    #          'x', color=blue, markersize=6, label='MC-Exp Fit')
    x = np.linspace(dfp['gross_emissions_change_tonco2'].min(), dfp['gross_emissions_change_tonco2'].max(), 100)
    a32.plot(1e-3 * x, 1e-3 * (a * x ** 2 + b * x + c + d / x), linestyle=(0, (3, 5, 1, 5, 1, 5)),
             color=red, label='AC-Poly Fit')
    a32.plot(1e-3 * x, 1e-3 * (np.exp(lna) * np.exp(lnb) ** x) / x, linestyle=(0, (3, 5, 1, 5, 1, 5)),
             color=blue, label='AC-Exp Fit')
    a32.plot(1e-3 * x, 1e-3 * (amcp * x ** 2 + bmcp * x + cmcp), '-', color=red, label='MC-Poly Fit')
    a32.plot(1e-3 * x, 1e-3 * (amce * bmce ** x), '-', color=blue, label='MC-Exp Fit')
    a32.set_ylim(max(-0.2, a32.get_ylim()[0]), min(1, a32.get_ylim()[1]))
    a32.legend(bbox_to_anchor=(0.5, 1.02), loc='upper center', fontsize=5, ncol=5)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    plt.savefig(os.path.join(DEP_TAB_O_DIR, 'Figures', scenario_code + '.png'), dpi=300, format='png')
    if figshow:
        plt.show()
    else:
        plt.close(fig)

    # matplotlib.rcParams.update({'font.size': 12})
    sns.set_theme(palette='Set2', font_scale=1)

    # SECTION 2: SUMMARY
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    (a00, a01) = ax
    # ((a00, a01), (a10, a11)) = ax
    # fig, ax = plt.subplots(3, 3, figsize=(20, 8))
    # ((a00, a01, a02), (a10, a11, a12), (a20, a21, a22)) = ax

    dp_x = np.array(df['deployment_perc'])
    ec_y = np.array(df['emissions_change'])
    idxs_dominated = []
    for i in range(len(dp_x)):
        for j in range(len(dp_x)):
            if i != j:
                if dp_x[j] >= dp_x[i] and ec_y[j] <= ec_y[i]:
                    idxs_dominated.append(i)
                    break
    idxs_keep = list(set(df.index).difference(set(idxs_dominated)))
    dp_x, ec_y = zip(*sorted(zip(df.loc[idxs_keep, 'deployment_perc'].values,
                                 df.loc[idxs_keep, 'emissions_change'].values)))

    a00.set_title('% Emissions Reduction vs. Deployment %')
    a00.set_xlabel('Deployment % [ton-mi]')
    a00.set_ylabel('% Emissions Reduction')
    a00.plot(dp_x, ec_y, '.-', color=blue)

    if fuel_type == 'battery':
        a01.set_title('Number of Facilities and Chargers vs. Deployment %')
        a01.set_xlabel('Deployment % [ton-mi]')
        a01.set_ylabel('Number of Chargers', color=blue)
        a01.plot(df['deployment_perc'], df['number_chargers'], '.-', color=blue)
    elif fuel_type == 'hydrogen':
        a01.set_title('Number of Facilities and Pumps vs. Deployment %')
        a01.set_xlabel('Deployment % [ton-mi]')
        a01.set_ylabel('Number of Pumps', color=blue)
        a01.plot(df['deployment_perc'], df['number_pumps'], '.-', color=blue)
    a01.tick_params(axis='y', labelcolor=blue)
    a01r = a01.twinx()  # second y axes
    a01r.set_ylabel('Number of Facilities', color=red)
    a01r.plot(df['deployment_perc'], df['number_facilities'], '.-', color=red)
    a01r.tick_params(axis='y', labelcolor=red)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    plt.savefig(os.path.join(DEP_TAB_O_DIR, 'Figures', scenario_code + '_summary.png'), dpi=300, format='png')
    if figshow:
        plt.show()
    else:
        plt.close(fig)

    # SECTION 2a: SUMMARY SINGLE
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    a00 = ax

    dp_x_ec = np.array(df['deployment_perc'])
    ec_y = np.array(df['emissions_change'])
    idxs_dominated = []
    for i in range(len(dp_x_ec)):
        for j in range(len(dp_x_ec)):
            if i != j:
                if dp_x_ec[j] >= dp_x_ec[i] and ec_y[j] <= ec_y[i]:
                    idxs_dominated.append(i)
                    break
    idxs_keep = list(set(df.index).difference(set(idxs_dominated)))
    dp_x_ec, ec_y = zip(*sorted(zip(df.loc[idxs_keep, 'deployment_perc'].values,
                                    df.loc[idxs_keep, 'emissions_change'].values)))

    dp_x_nf = np.array(df['deployment_perc'])
    nf_y = np.array(df['number_facilities'])
    idxs_dominated = []
    for i in range(len(dp_x_nf)):
        for j in range(len(dp_x_nf)):
            if i != j:
                if dp_x_nf[j] >= dp_x_nf[i] and nf_y[j] <= nf_y[i]:
                    idxs_dominated.append(i)
                    break
    idxs_keep = list(set(df.index).difference(set(idxs_dominated)))
    dp_x_nf, nf_y = zip(*sorted(zip(df.loc[idxs_keep, 'deployment_perc'].values,
                                    df.loc[idxs_keep, 'number_facilities'].values)))

    a00.set_title('% Emissions Reduction and Number of Facilities vs. Deployment %')
    a00.set_xlabel('Deployment % [ton-mi]')
    blue = '#4e2a84'
    red = '#d1b558'
    a00.set_ylabel('% Emissions Reduction', color=blue)
    a00.plot(dp_x_ec, ec_y, '.-', color=blue)
    a00.tick_params(axis='y', labelcolor=blue)
    a00r = a00.twinx()  # second y axes
    a00r.set_ylabel('Number of Facilities', color=red)
    a00r.plot(dp_x_nf, nf_y, '.-', color=red)
    a00r.tick_params(axis='y', labelcolor=red)
    a00r.grid(None)

    fig.tight_layout()
    # plt.subplots_adjust(wspace=0.3, hspace=0.35)
    plt.savefig(os.path.join(DEP_TAB_O_DIR, 'Figures', scenario_code + '_summary_single.png'), dpi=300, format='png')
    if figshow:
        plt.show()
    else:
        plt.close(fig)

    # # SECTION 3: fit TC, AC, MC
    # fig = plt.figure(figsize=(12, 7))
    # ax = fig.gca()
    # ax.set_title('Cost of Avoided Emissions vs. Gross Emissions Reduction')
    # ax.set_xlabel('Gross Emissions Reduction [kt CO_2]')
    # ax.set_ylabel('Cost of Avoided Emissions [$/kg CO_2]')
    # ax.plot(df['gross_emissions_change_tonco2'] * 1e-3, df['cost_avoided_emissions_$_kg'], 'd', color=grey, label='AC')
    # ax.plot(dfp['gross_emissions_change_tonco2'] * 1e-3, dfp['marginal_cost_avoided_emissions_$_kg_poly'],
    #         '+', color=red, markersize=7, label='MC-Poly Fit')
    # ax.plot(dfp['gross_emissions_change_tonco2'] * 1e-3, dfp['marginal_cost_avoided_emissions_$_kg_exp'],
    #         'x', color=blue, markersize=6, label='MC-Exp Fit')
    # x = np.linspace(dfp['gross_emissions_change_tonco2'].min(), dfp['gross_emissions_change_tonco2'].max(), 100)
    # ax.plot(1e-3 * x, 1e-3 * (a * x ** 2 + b * x + c + d / x), linestyle=(0, (3, 5, 1, 5, 1, 5)),
    #         color=red, label='AC-Poly Fit')
    # ax.plot(1e-3 * x, 1e-3 * (np.exp(lna) * np.exp(lnb) ** x) / x, linestyle=(0, (3, 5, 1, 5, 1, 5)), color=blue,
    #         label='AC-Exp Fit')
    # ax.plot(1e-3 * x, 1e-3 * (amcp * x ** 2 + bmcp * x + cmcp), '-', color=red, label='MC-Poly Fit')
    # ax.plot(1e-3 * x, 1e-3 * (amce * bmce ** x), '-', color=blue, label='MC-Exp Fit')
    # ax.set_ylim(max(0, ax.get_ylim()[0]), min(1, ax.get_ylim()[1]))
    # ax.legend(bbox_to_anchor=(0.5, 0.98), loc='upper center', fontsize=10, ncol=5)
    # axr = ax.twinx()  # second y axes
    # axr.set_ylabel('Total Cost [M$]', color=yellow)
    # axr.plot(1e-3 * dfp['gross_emissions_change_tonco2'], 1e-6 * dfp['total_cost_avoided_emissions'], '*', color=yellow)
    # axr.plot(1e-3 * x, 1e-6 * (a * x ** 3 + b * x ** 2 + c * x + d), '-', color=yellow)
    # axr.tick_params(axis='y', labelcolor=yellow)

    # axb = ax.twiny()
    # axb.xaxis.set_ticks_position("bottom")
    # axb.xaxis.set_label_position("bottom")
    # axb.spines["bottom"].set_position(("axes", -0.15))
    # axb.set_frame_on(True)
    # axb.patch.set_visible(False)
    # axb.spines[:].set_visible(False)
    # axb.spines['bottom'].set_visible(True)
    # tick_vals = ax.get_xticks()
    # axb.set_xticks(tick_vals)
    # axb.set_xticklabels([str(int(round(100 * x / (1e-3 * df['baseline_annual_emissions_tonco2'].max()))))
    #                      for x in tick_vals])
    # axb.set_xlabel('% Emissions Reduction')

    # fig.tight_layout()
    # fig.subplots_adjust(bottom=0.2)
    # plt.savefig(os.path.join(DEP_TAB_O_DIR, 'Figures', scenario_code + 'AC_MC.png'), dpi=300, format='png')
    # if figshow:
    #     plt.show()
    # else:
    #     plt.close(fig)

    # SECTION 3: clean, numerical fit, TC, AC, MC
    fig = plt.figure(figsize=(12, 7))
    ax = fig.gca()
    ax.set_title('Cost of Avoided Emissions vs. Gross Emissions Reduction')
    ax.set_xlabel('Gross Emissions Reduction [kt CO_2]')
    ax.set_ylabel('Cost of Avoided Emissions [$/kg CO_2]')
    ge_x = np.array(df['gross_emissions_change_tonco2'])
    tc_y = np.array(df['total_cost_avoided_emissions'])
    idxs_dominated = []
    for i in range(len(ge_x)):
        for j in range(len(ge_x)):
            if i != j:
                if ge_x[j] >= ge_x[i] and tc_y[j] <= tc_y[i]:
                    idxs_dominated.append(i)
                    break
    idxs_keep = list(set(df.index).difference(set(idxs_dominated)))
    ge_x, tc_y, ac_y = zip(*sorted(zip(df.loc[idxs_keep, 'gross_emissions_change_tonco2'].values,
                                       df.loc[idxs_keep, 'total_cost_avoided_emissions'].values,
                                       df.loc[idxs_keep, 'cost_avoided_emissions_$_kg'].values)))
    ge_x = 1e-3 * np.array(ge_x)
    tc_y = np.array(tc_y)
    ac_y = np.array(ac_y)
    # ge_x = df.loc[idxs_keep, 'gross_emissions_change_tonco2'].values
    # tc_y = df.loc[idxs_keep, 'total_cost_avoided_emissions'].values
    # ac_y = df.loc[idxs_keep, 'cost_avoided_emissions_$_kg'].values
    #
    mc_y = np.array([])
    gec_x = np.array([])
    for i in range(1, len(tc_y)):
        ge = (ge_x[i] - ge_x[i - 1]) / 2
        mc = 1e-6 * (tc_y[i] - tc_y[i - 1]) / (2 * ge)
        mc_y = np.append(mc_y, mc)
        gec_x = np.append(gec_x, ge + ge_x[i - 1])

    ax.plot(ge_x, ac_y, 'd', color=red, label='AC')
    ax.plot(ge_x, ac_y, '-.', color=red, label='AC')
    ax.plot(gec_x, mc_y, 'd-', color=blue, label='MC')
    # ax.plot(1e-3 * gec_x, mc_y, '-', color=grey, label='MC')
    # ax.plot(dfp['gross_emissions_change_tonco2'] * 1e-3, dfp['marginal_cost_avoided_emissions_$_kg_poly'],
    #         '+', color=red, markersize=7, label='MC-Poly Fit')
    # ax.plot(dfp['gross_emissions_change_tonco2'] * 1e-3, dfp['marginal_cost_avoided_emissions_$_kg_exp'],
    #         'x', color=blue, markersize=6, label='MC-Exp Fit')
    ax.set_ylim(max(0, ax.get_ylim()[0]), min(1, ax.get_ylim()[1]))
    ax.legend(bbox_to_anchor=(0.5, 0.98), loc='upper center', fontsize=10, ncol=5)
    axr = ax.twinx()  # second y axes
    axr.set_ylabel('Total Cost [M$]', color=yellow)
    axr.plot(ge_x, 1e-6 * tc_y, '*-', color=yellow)
    # axr.plot(1e-3 * ge_x, 1e-6 * tc_y, '-', color=yellow)
    axr.tick_params(axis='y', labelcolor=yellow)

    axb = ax.twiny()
    axb.xaxis.set_ticks_position("bottom")
    axb.xaxis.set_label_position("bottom")
    axb.spines["bottom"].set_position(("axes", -0.15))
    axb.set_frame_on(True)
    axb.patch.set_visible(False)
    axb.spines[:].set_visible(False)
    axb.spines['bottom'].set_visible(True)
    tick_vals = ax.get_xticks()
    axb.set_xticks(tick_vals)
    axb.set_xticklabels([str(int(round(100 * x / (1e-3 * df['baseline_annual_emissions_tonco2'].max()))))
                         for x in tick_vals])
    axb.set_xlabel('% Emissions Reduction')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(DEP_TAB_O_DIR, 'Figures', scenario_code + 'diff_AC_MC.png'), dpi=300, format='png')
    if figshow:
        plt.show()
    else:
        plt.close(fig)

    return
