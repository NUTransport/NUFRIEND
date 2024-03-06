from util import *
# MODULES
from helper import regional_alignments
from input_output import load_cached_graph, cache_graph, cache_exists, extract_assert_scenario_inputs
from network_representation import load_simplified_consolidated_graph
from routing import ods_by_perc_ton_mi, route_flows
from facility_deployment import facility_location
from facility_sizing import facility_sizing, facility_sizing_hybrid
from tea import tea_battery_all_facilities, tea_dropin, tea_hydrogen_all_facilities, tea_hybrid
from lca import lca_battery, lca_dropin, lca_hydrogen, lca_hybrid
from plotting import plot_scenario


def run_scenario_file(scenario_code: str, G: nx.DiGraph = None, plot=True, load_scenario=True, cache_scenario=False):

    if load_scenario and cache_exists(scenario_code):
        t0 = time.time()
        G = load_cached_graph(scenario_code)
        print('LOADED GRAPH FROM CACHE:: %s seconds ---' % round(time.time() - t0, 3))

        [rr, fuel_type, deployment_perc, range_km, max_flow, budget, extend_graph, reroute, switch_tech,
         max_reroute_inc, max_util, station_type, h2_fuel_type, clean_energy, clean_energy_cost, emissions_obj,
         eff_energy_p_tender, tender_cost_p_tonmi, diesel_cost_p_gal, comm_group, flow_data_filename,
         suppress_output, legend_show, scenario_code] = extract_assert_scenario_inputs(scenario_code=scenario_code)
    else:
        t0_total = time.time()

        [rr, fuel_type, deployment_perc, range_km, max_flow, budget, extend_graph, reroute, switch_tech,
         max_reroute_inc, max_util, station_type, h2_fuel_type, clean_energy, clean_energy_cost, emissions_obj,
         eff_energy_p_tender, tender_cost_p_tonmi, diesel_cost_p_gal, comm_group, flow_data_filename,
         suppress_output, legend_show, scenario_code] = extract_assert_scenario_inputs(scenario_code=scenario_code)

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

        # 0. load railroad network representation as a nx.Graph and a simplify and consolidate network
        if not G:
            G = load_simplified_consolidated_graph(rr)
        else:
            # get a deep copy of G so that changes made to local G are not made to the original G
            G = deepcopy(G)

        if fuel_type == 'battery':
            G.graph['scenario'] = dict(railroad=rr, range_mi=range_km * KM2MI, fuel_type=fuel_type,
                                       desired_deployment_perc=deployment_perc, reroute=reroute,
                                       switch_tech=switch_tech,
                                       max_reroute_inc=max_reroute_inc, max_util=max_util, station_type=station_type,
                                       eff_kwh_p_batt=eff_energy_p_tender, scenario_code=scenario_code)
            # 1. load od_flow_dict for ranking OD pairs and choosing top <perc_ods> for flows for facility location
            t0 = time.time()
            print('LOOKUP TABLE:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            # select almost all O-D pairs with non-zero flow (leave out the 20% with the lowest flow values; too many)
            ods, od_flows = ods_by_perc_ton_mi(G=G, flow_data_filename=flow_data_filename)

            G.graph['framework'] = dict(ods=ods)
            print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            # 2. locate facilities and extract graph form of this, G, and its induced subgraph, H
            G, H = facility_location(G, D=range_km, ods=ods, od_flows=od_flows, flow_min=deployment_perc, budget=budget,
                                     max_flow=max_flow, extend_graph=extend_graph, suppress_output=suppress_output)
            print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))

            # if no facilities are selected
            if G.graph['number_facilities'] == 0:
                return G

            t0 = time.time()
            # 3. reroute flows and get peak and average ton and locomotive flows for each edge
            G, H = route_flows(G=G, fuel_type=fuel_type, flow_data_filename=flow_data_filename, H=H, D=range_km,
                               reroute=reroute, switch_tech=switch_tech, max_reroute_inc=max_reroute_inc)
            print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            # 4. facility sizing based on peak flows and utilization based on average flows
            # load cost by state dataframe and assign to each node
            # emissions_p_location = elec_rate_state(G, emissions=True, clean_energy=clean_energy,
            #                                        clean_elec_prem_dolkwh=clean_energy_cost)  # [gCO2/kWh]
            # cost_p_location = elec_rate_state(G, clean_energy=clean_energy,
            #                                   clean_elec_prem_dolkwh=clean_energy_cost)  # in [$/MWh]

            G = facility_sizing(G=G, H=H, fuel_type=fuel_type, D=range_km, emissions_obj=emissions_obj,
                                suppress_output=suppress_output)
            print('FACILITY SIZING:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            actual_dep_perc = G.graph['operations']['deployment_perc'][comm_group]
            G.graph['scenario']['actual_deployment_perc'] = actual_dep_perc
            # 5.1. TEA
            G = tea_battery_all_facilities(G, max_util=max_util,
                                           clean_energy_cost=clean_energy_cost if clean_energy else None,
                                           tender_cost_p_tonmi=tender_cost_p_tonmi, diesel_cost_p_gal=diesel_cost_p_gal)
            # baseline and other dropin fuels (easy factor calculation)
            G = tea_dropin(G=G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
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
            G.graph['scenario'] = dict(railroad=rr, range_mi=range_km * KM2MI, fuel_type=fuel_type,
                                       desired_deployment_perc=deployment_perc, reroute=reroute,
                                       switch_tech=switch_tech,
                                       max_reroute_inc=max_reroute_inc, max_util=max_util, station_type=station_type,
                                       eff_kgh2_p_loc=eff_energy_p_tender, scenario_code=scenario_code)
            # 1. load od_flow_dict for ranking OD pairs and choosing top <perc_ods> for flows for facility location
            t0 = time.time()
            print('LOOKUP TABLE:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            ods, od_flows = ods_by_perc_ton_mi(G=G, flow_data_filename=flow_data_filename)
            G.graph['framework'] = dict(ods=ods)
            print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            # 2. locate facilities and extract graph form of this, G, and its induced subgraph, H
            G, H = facility_location(G, D=range_km, ods=ods, od_flows=od_flows, flow_min=deployment_perc, budget=budget,
                                     max_flow=max_flow, extend_graph=extend_graph, suppress_output=suppress_output)

            print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))

            # if no facilities are selected
            if G.graph['number_facilities'] == 0:
                return G

            t0 = time.time()
            # 3. reroute flows and get peak and average ton and locomotive flows for each edge
            G, H = route_flows(G=G, fuel_type=fuel_type, flow_data_filename=flow_data_filename, H=H, D=range_km,
                               reroute=reroute, switch_tech=switch_tech, max_reroute_inc=max_reroute_inc)
            print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            # 4. facility sizing based on peak flows and utilization based on average flows
            G = facility_sizing(G=G, H=H, fuel_type=fuel_type, D=range_km, unit_sizing_obj=True,
                                suppress_output=suppress_output)
            print('FACILITY SIZING:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            actual_dep_perc = G.graph['operations']['deployment_perc'][comm_group]
            G.graph['scenario']['actual_deployment_perc'] = actual_dep_perc
            # 5.1. TEA
            G = tea_hydrogen_all_facilities(G, max_util=max_util, station_type=station_type,
                                            clean_energy_cost=clean_energy_cost if clean_energy else None,
                                            tender_cost_p_tonmi=tender_cost_p_tonmi,
                                            diesel_cost_p_gal=diesel_cost_p_gal)
            # baseline and other dropin fuels (easy factor calculation)
            G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
            G = tea_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
            G = tea_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
            print('TEA:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            # 5.2. LCA
            G = lca_hydrogen(G, h2_fuel_type=h2_fuel_type)
            # baseline and other dropin fuels (easy factor calculation)
            G = lca_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
            G = lca_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
            G = lca_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
            print('LCA:: %s seconds ---' % round(time.time() - t0, 3))

        elif 'hybrid' in fuel_type:
            # add in regional alignment information to G (based on A-STEP simulation tool)
            G = regional_alignments(G)

            G.graph['scenario'] = dict(railroad=rr, range_mi=range_km * KM2MI, fuel_type=fuel_type,
                                       desired_deployment_perc=deployment_perc, reroute=reroute,
                                       switch_tech=switch_tech,
                                       max_reroute_inc=max_reroute_inc, max_util=max_util, station_type=station_type,
                                       eff_kwh_p_batt=eff_energy_p_tender, scenario_code=scenario_code)
            # 1. load od_flow_dict for ranking OD pairs and choosing top <perc_ods> for flows for facility location
            t0 = time.time()
            # if perc_ods is None or perc_ods == 'X':
            #     perc_ods = deployment_perc_lookup_table(df_scenario=df_scenario, deployment_perc=deployment_perc)
            print('LOOKUP TABLE:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            # select almost all O-D pairs with non-zero flow (leave out the 20% with the lowest flow values; too many)
            ods, od_flows = ods_by_perc_ton_mi(G=G, flow_data_filename=flow_data_filename)
            G.graph['framework'] = dict(ods=ods)
            print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            # 2. locate facilities and extract graph form of this, G, and its induced subgraph, H
            G, H = facility_location(G, D=range_km, ods=ods, od_flows=od_flows, flow_min=deployment_perc, budget=budget,
                                     max_flow=max_flow, extend_graph=extend_graph, suppress_output=suppress_output)
            print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))

            # if no facilities are selected
            if G.graph['number_facilities'] == 0:
                return G

            t0 = time.time()
            # 3. reroute flows and get peak and average ton and locomotive flows for each edge
            G, H = route_flows(G=G, fuel_type=fuel_type, flow_data_filename=flow_data_filename, H=G, D=range_km,
                               reroute=reroute, switch_tech=switch_tech, max_reroute_inc=max_reroute_inc)
            print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            # 4. facility sizing based on peak flows and utilization based on average flows
            # load cost by state dataframe and assign to each node
            # emissions_p_location = elec_rate_state(G, emissions=True, clean_energy=clean_energy,
            #                                        clean_elec_prem_dolkwh=clean_energy_cost)  # [gCO2/kWh]
            # cost_p_location = elec_rate_state(G, clean_energy=clean_energy,
            #                                   clean_elec_prem_dolkwh=clean_energy_cost)  # in [$/MWh]

            G = facility_sizing_hybrid(G=G, H=H, fuel_type=fuel_type, D=range_km, emissions_obj=emissions_obj,
                                       suppress_output=suppress_output)
            print('FACILITY SIZING:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            actual_dep_perc = G.graph['operations']['deployment_perc'][comm_group]
            G.graph['scenario']['actual_deployment_perc'] = actual_dep_perc
            # 5.1. TEA
            G = tea_hybrid(G, max_util=max_util, station_type=station_type,
                           clean_energy_cost=clean_energy_cost if clean_energy else None,
                           tender_cost_p_tonmi=tender_cost_p_tonmi, diesel_cost_p_gal=diesel_cost_p_gal)
            # baseline and other dropin fuels (easy factor calculation)
            G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
            # G = tea_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
            # G = tea_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
            print('TEA:: %s seconds ---' % round(time.time() - t0, 3))

            t0 = time.time()
            # 5.2. LCA
            G = lca_hybrid(G, clean_energy=clean_energy)
            # baseline and other dropin fuels (easy factor calculation)
            G = lca_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type)
            # G = lca_dropin(G, fuel_type='biodiesel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
            # G = lca_dropin(G, fuel_type='e-fuel', deployment_perc=actual_dep_perc, scenario_fuel_type=fuel_type)
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
            G = route_flows(G=G, fuel_type=fuel_type, flow_data_filename=flow_data_filename)
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

        # G = update_graph_values(G=G, fuel_type=fuel_type, max_util=max_util, station_type=station_type,
        #                         clean_energy=clean_energy, clean_energy_cost=clean_energy_cost,
        #                         h2_fuel_type=h2_fuel_type, tender_cost_p_tonmi=tender_cost_p_tonmi,
        #                         diesel_cost_p_gal=diesel_cost_p_gal)
        print('SCENARIO RUN:: %s seconds ---' % round(time.time() - t0_total, 3))

        if cache_scenario:
            t0 = time.time()
            cache_graph(G=G, scenario_code=scenario_code)
            print('CACHE GRAPH:: %s seconds ---' % round(time.time() - t0, 3))

    if plot:
        t0 = time.time()
        fig, label = plot_scenario(G, fuel_type=fuel_type, deployment_perc=deployment_perc, comm_group=comm_group,
                                   legend_show=legend_show)
        print('PLOTTING:: %s seconds ---' % round(time.time() - t0, 3))
    else:
        fig = None
        label = None

    return G, fig, label


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
                      G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list])),
                cost_avoided_emissions_no_delay=dict(zip(
                    comm_list,
                    [-1e-3 * (G.graph['energy_source_TEA']['total_scenario_nodelay_LCO_tonmi'][c] -
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
                      G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list])),
                cost_avoided_emissions_no_delay=dict(zip(
                    comm_list,
                    [-1e-3 * (G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][c] -
                              G.graph['diesel_TEA']['total_LCO_tonmi'][c]) /
                     (G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'][c] -
                      G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list]))
            ))
    elif 'hybrid' in G.graph['scenario']['fuel_type']:
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
                      G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][c]) for c in comm_list])),
                cost_avoided_emissions_no_delay=dict(zip(
                    comm_list,
                    [-1e-3 * (G.graph['energy_source_TEA']['total_scenario_nodelay_LCO_tonmi'][c] -
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


def update_graph_values(G: nx.DiGraph, fuel_type: str, max_util: float, station_type: str,
                        clean_energy=True, clean_energy_cost: float = None,
                        h2_fuel_type: str = None, tender_cost_p_tonmi: float = None,
                        diesel_cost_p_gal: float = None) -> nx.DiGraph:
    # valid <h2_fuel_type> values:
    # ['Natural Gas', 'NG with CO2 Sequestration', 'PEM Electrolysis - Solar', 'PEM Electrolysis - Nuclear']

    if 'hybrid' in fuel_type:
        return G

    if not clean_energy and tender_cost_p_tonmi is None and diesel_cost_p_gal is None:
        return G

    if clean_energy_cost is None:
        clean_energy_cost = 0.0

    # update TEA with premiums on clean energy costs and LCA with emissions cuts
    if fuel_type == 'battery':
        t0 = time.time()
        G = lca_battery(G=G, clean_energy=clean_energy)
        print('LCA2:: ' + str(time.time() - t0))

        t0 = time.time()
        G = tea_battery_all_facilities(G, max_util=max_util, clean_energy_cost=clean_energy_cost,
                                       tender_cost_p_tonmi=tender_cost_p_tonmi, diesel_cost_p_gal=diesel_cost_p_gal)
        G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type,
                       diesel_cost_p_gal=diesel_cost_p_gal)
        print('TEA UPDATE:: ' + str(time.time() - t0))
    elif 'hybrid' in fuel_type:
        t0 = time.time()
        G = lca_hybrid(G=G, clean_energy=clean_energy)
        print('LCA2:: ' + str(time.time() - t0))

        t0 = time.time()
        G = tea_hybrid(G, max_util=max_util, station_type=station_type,
                       clean_energy_cost=clean_energy_cost, tender_cost_p_tonmi=tender_cost_p_tonmi,
                       diesel_cost_p_gal=diesel_cost_p_gal)
        G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type,
                       diesel_cost_p_gal=diesel_cost_p_gal)
        print('TEA UPDATE:: ' + str(time.time() - t0))
    elif fuel_type == 'hydrogen':
        if h2_fuel_type is None:
            h2_fuel_type = 'Natural Gas'

        t0 = time.time()
        G = lca_hydrogen(G=G, h2_fuel_type=h2_fuel_type)
        print('LCA UPDATE:: ' + str(time.time() - t0))

        t0 = time.time()
        G = tea_hydrogen_all_facilities(G=G, max_util=max_util, station_type=station_type,
                                        clean_energy_cost=clean_energy_cost, diesel_cost_p_gal=diesel_cost_p_gal)
        G = tea_dropin(G, fuel_type='diesel', deployment_perc=1, scenario_fuel_type=fuel_type,
                       diesel_cost_p_gal=diesel_cost_p_gal)
        print('TEA UPDATE:: ' + str(time.time() - t0))

    G = operations_stats(G)
    return G
