from util import *
# MODULES
from network_representation import load_simplified_consolidated_graph
from facility_rollout_mp import facility_location_mp
from facility_sizing_mp import facility_sizing_mp
from tea_mp import tea_battery_mp, tea_diesel_mp, tea_hydrogen_mp
from lca_mp import lca_battery_mp, lca_diesel_mp, lca_hydrogen_mp
from routing_mp import route_flows_mp, od_flows_ton_mi_mp
from plotting_mp import plot_dynamic_network_results_mp
from input_output import extract_assert_scenario_mp_inputs


def run_mp_scenario_file(scenario_code: str, G: nx.DiGraph = None, plot=True):

    [rr, fuel_type, range_km, max_flow,
     time_horizon, deployment_percs, budgets, discount_rates,
     facility_costs, fixed_facilities, barred_facilities,
     max_util, station_type, h2_fuel_type, clean_energy, clean_energy_cost, emissions_obj,
     eff_energy_p_tender, tender_cost_p_tonmi, diesel_cost_p_gal, flow_data_filename,
     suppress_output, opt_tol, scenario_code] = extract_assert_scenario_mp_inputs(scenario_code=scenario_code)

    if not G:
        G = load_simplified_consolidated_graph(rr)

    if fuel_type == 'battery':
        G.graph['scenario'] = dict(railroad=rr, range_mi=range_km * KM2MI, range_km=range_km, fuel_type=fuel_type,
                                   eff_kwh_p_batt=eff_energy_p_tender)
        t0 = time.time()
        ods, od_flows_ton_mi, od_flows_ton = od_flows_ton_mi_mp(G=G, flow_data_filename=flow_data_filename,
                                                                time_horizon=time_horizon)

        print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 2. locate facilities and extract graph form of this, G, and its induced subgraph, H
        G = facility_location_mp(G, range_km=range_km, time_horizon=time_horizon, od_flows_ton_mi=od_flows_ton_mi,
                                 facility_costs=facility_costs, max_flow=max_flow,
                                 deployment_percs=deployment_percs,
                                 budgets=budgets, discount_rates=discount_rates,
                                 fixed_facilities=fixed_facilities, barred_facilities=barred_facilities,
                                 suppress_output=suppress_output, opt_tol=opt_tol)
        # update ODs considered (as some paths are infeasible due to insufficient range)
        ods = G.graph['framework']['ods']

        print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 3. reroute flows and get average ton and locomotive flows for each edge
        G = route_flows_mp(G=G, range_km=range_km, flow_data_filename=flow_data_filename, time_horizon=time_horizon,
                           od_list=ods, fuel_type=fuel_type)
        print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 4. size facilities based on required energy
        G = facility_sizing_mp(G=G, time_horizon=time_horizon, fuel_type=fuel_type, range_km=range_km,
                               emissions_obj=emissions_obj, suppress_output=suppress_output)
        print('FACILITY SIZING:: {v0} seconds ---'.format(v0=round(time.time() - t0, 3)))

        t0 = time.time()
        # 5. LCA for each time period
        G = lca_battery_mp(G=G, time_horizon=time_horizon, clean_energy=clean_energy)
        G = lca_diesel_mp(G=G, time_horizon=time_horizon)
        print('LCA:: {v0} seconds ---'.format(v0=round(time.time() - t0, 3)))

        t0 = time.time()
        # 6. TEA for each time period
        G = tea_battery_mp(G=G, time_horizon=time_horizon, max_util=max_util,
                           clean_energy_cost=clean_energy_cost if clean_energy else None,
                           tender_cost_p_tonmi=tender_cost_p_tonmi, diesel_cost_p_gal=diesel_cost_p_gal)
        G = tea_diesel_mp(G=G, time_horizon=time_horizon)
        print('TEA:: {v0} seconds ---'.format(v0=round(time.time() - t0, 3)))

        # update stats
        G = operations_stats_mp(G=G, time_horizon=time_horizon)

        t0 = time.time()
        fig = None
        if plot:
            fig = plot_dynamic_network_results_mp(G, time_horizon=time_horizon, fuel_type=fuel_type,
                                                  additional_plots=True, max_flow=max_flow,
                                                  time_step_label=time_horizon, title=scenario_code)
        print('PLOTTING:: %s seconds ---' % round(time.time() - t0, 3))
    elif fuel_type == 'hydrogen':
        G.graph['scenario'] = dict(railroad=rr, range_mi=range_km * KM2MI, range_km=range_km, fuel_type=fuel_type,
                                   eff_kwh_p_batt=eff_energy_p_tender)
        t0 = time.time()
        ods, od_flows_ton_mi, od_flows_ton = od_flows_ton_mi_mp(G=G, flow_data_filename=flow_data_filename,
                                                                time_horizon=time_horizon)

        print('OD LIST:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 2. locate facilities and extract graph form of this, G, and its induced subgraph, H
        G = facility_location_mp(G, range_km=range_km, time_horizon=time_horizon, od_flows_ton_mi=od_flows_ton_mi,
                                 facility_costs=facility_costs, max_flow=max_flow,
                                 deployment_percs=deployment_percs,
                                 budgets=budgets, discount_rates=discount_rates,
                                 fixed_facilities=fixed_facilities, barred_facilities=barred_facilities,
                                 suppress_output=suppress_output, opt_tol=opt_tol)
        # update ODs considered (as some paths are infeasible due to insufficient range)
        ods = G.graph['framework']['ods']

        print('FACILITY LOCATION:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 3. reroute flows and get average ton and locomotive flows for each edge
        G = route_flows_mp(G=G, range_km=range_km, flow_data_filename=flow_data_filename, time_horizon=time_horizon,
                           od_list=ods, fuel_type=fuel_type)
        print('FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

        t0 = time.time()
        # 4. size facilities based on required energy
        G = facility_sizing_mp(G=G, time_horizon=time_horizon, fuel_type=fuel_type, range_km=range_km,
                               emissions_obj=emissions_obj, suppress_output=suppress_output)
        print('FACILITY SIZING:: {v0} seconds ---'.format(v0=round(time.time() - t0, 3)))

        t0 = time.time()
        # 5. LCA for each time period
        G = lca_hydrogen_mp(G=G, time_horizon=time_horizon, h2_fuel_type=h2_fuel_type, clean_energy=clean_energy)
        G = lca_diesel_mp(G=G, time_horizon=time_horizon)
        print('LCA:: {v0} seconds ---'.format(v0=round(time.time() - t0, 3)))

        # return G, None

        t0 = time.time()
        # 6. TEA for each time period
        G = tea_hydrogen_mp(G=G, time_horizon=time_horizon, max_util=max_util, station_type=station_type,
                            clean_energy_cost=clean_energy_cost if clean_energy else None,
                            tender_cost_p_tonmi=tender_cost_p_tonmi, diesel_cost_p_gal=diesel_cost_p_gal)
        G = tea_diesel_mp(G=G, time_horizon=time_horizon)
        print('TEA:: {v0} seconds ---'.format(v0=round(time.time() - t0, 3)))
        # return G, None

        # update stats
        G = operations_stats_mp(G=G, time_horizon=time_horizon)

        t0 = time.time()
        fig = None
        if plot:
            fig = plot_dynamic_network_results_mp(G, time_horizon=time_horizon, fuel_type=fuel_type,
                                                  additional_plots=True, max_flow=max_flow,
                                                  time_step_label=time_horizon, title=scenario_code)
        print('PLOTTING:: %s seconds ---' % round(time.time() - t0, 3))

    return G, fig


def operations_stats_mp(G: nx.DiGraph, time_horizon: list) -> nx.DiGraph:
    # compute the operational stats of solution in G (many relative to diesel baseline)

    comm_list = list(G.graph['operations']['baseline_total_annual_tonmi'][time_horizon[0]].keys())

    if G.graph['scenario']['fuel_type'] == 'battery':
        G.graph['operations'].update(dict(
            emissions_change={t: {c: 100 * (G.graph['diesel_LCA'][t]['annual_total_emissions_tonco2'][c] -
                                            G.graph['energy_source_LCA'][t]['annual_total_emissions_tonco2'][c]) /
                                     G.graph['diesel_LCA'][t]['annual_total_emissions_tonco2'][c]
                                  for c in comm_list} for t in time_horizon},
            cost_avoided_emissions={t: {c: 1e-3 * (G.graph['energy_source_TEA'][t]['total_scenario_LCO_tonmi'][c] -
                                                   G.graph['diesel_TEA'][t]['total_LCO_tonmi'][c]) /
                                           (G.graph['diesel_LCA'][t]['total_emissions_tonco2_tonmi'][c] -
                                            G.graph['energy_source_LCA'][t]['avg_emissions_tonco2_tonmi'][c])
                                        for c in comm_list} for t in time_horizon},
            cost_avoided_emissions_no_delay={t: {c: 1e-3 *
                                                    (G.graph['energy_source_TEA'][t]
                                                     ['total_scenario_nodelay_LCO_tonmi'][c] -
                                                     G.graph['diesel_TEA'][t]['total_LCO_tonmi'][c]) /
                                                    (G.graph['diesel_LCA'][t]['total_emissions_tonco2_tonmi'][c] -
                                                     G.graph['energy_source_LCA'][t]['avg_emissions_tonco2_tonmi'][c])
                                                 for c in comm_list} for t in time_horizon}
            ))
    elif G.graph['scenario']['fuel_type'] == 'hydrogen':
        G.graph['operations'].update(dict(
            emissions_change={t: {c: 100 * (G.graph['diesel_LCA'][t]['annual_total_emissions_tonco2'][c] -
                                            G.graph['energy_source_LCA'][t]['annual_total_emissions_tonco2'][c]) /
                                     G.graph['diesel_LCA'][t]['annual_total_emissions_tonco2'][c]
                                  for c in comm_list} for t in time_horizon},
            cost_avoided_emissions={t: {c: 1e-3 * (G.graph['energy_source_TEA'][t]['total_scenario_LCO_tonmi'][c] -
                                                   G.graph['diesel_TEA'][t]['total_LCO_tonmi'][c]) /
                                           (G.graph['diesel_LCA'][t]['total_emissions_tonco2_tonmi'][c] -
                                            G.graph['energy_source_LCA'][t]['avg_emissions_tonco2_tonmi'][c])
                                        for c in comm_list} for t in time_horizon},
            cost_avoided_emissions_no_delay={t: {c: 1e-3 *
                                                    (G.graph['energy_source_TEA'][t]
                                                     ['total_scenario_nodelay_LCO_tonmi'][c] -
                                                     G.graph['diesel_TEA'][t]['total_LCO_tonmi'][c]) /
                                                    (G.graph['diesel_LCA'][t]['total_emissions_tonco2_tonmi'][c] -
                                                     G.graph['energy_source_LCA'][t]['avg_emissions_tonco2_tonmi'][
                                                         c])
                                                 for c in comm_list} for t in time_horizon}
        ))

    return G
