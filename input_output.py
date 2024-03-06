import numpy as np
import pandas as pd

from util import *
from network_representation import gdfs_from_graph, digraph_from_gdfs, load_simplified_consolidated_graph

# SCENARIO


def load_scenario_df(scenario_code: str):
    filename = scenario_code + '.csv'
    df_scenario = pd.read_csv(os.path.join(SCENARIO_DIR, filename), header=0, index_col='Keyword')['Value']
    # convert str values to float where possible
    df_numeric = pd.to_numeric(df_scenario, errors='coerce')
    # keep original data for any string that was converted to NaN by <pd.to_numeric>
    idxs = list(df_numeric.dropna().index)
    df_scenario.loc[idxs] = df_numeric.loc[idxs]

    return df_scenario


def load_scenario_mp_df(scenario_code: str):
    filename = scenario_code + '.csv'
    df_scenario = pd.read_csv(os.path.join(SCENARIO_DIR, filename), header=0, index_col='Keyword')
    # convert str values to int/float where possible
    df_scenario.loc['time_horizon'] = df_scenario.loc['time_horizon'].apply(lambda x: str(int(x)))
    df_scenario.loc['deployment_percs'] = df_scenario.loc['deployment_percs'].apply(lambda x: float(x))
    df_scenario.loc['budgets'] = df_scenario.loc['budgets'].apply(lambda x: float(x))
    df_scenario.loc['discount_rates'] = df_scenario.loc['discount_rates'].apply(lambda x: float(x))
    int_vals = ['max_flow', 'clean_energy', 'emissions_obj', 'suppress_output']
    for int_val in int_vals:
        df_scenario.loc[int_val, 'Value0'] = int(df_scenario.loc[int_val, 'Value0'])
    float_vals = ['range_km', 'max_util', 'clean_energy_cost', 'eff_energy_p_tender',
                  'tender_cost_p_tonmi', 'diesel_cost_p_gal', 'opt_tol']
    for float_val in float_vals:
        df_scenario.loc[float_val, 'Value0'] = float(df_scenario.loc[float_val, 'Value0'])

    return df_scenario


def extract_assert_scenario_inputs(scenario_code: str):

    idxs = ['rr', 'fuel_type', 'deployment_perc', 'range_km', 'max_flow', 'budget', 'extend_graph', 'reroute',
            'switch_tech', 'max_reroute_inc', 'max_util', 'station_type', 'h2_fuel_type', 'clean_energy',
            'clean_energy_cost', 'emissions_obj', 'eff_energy_p_tender', 'tender_cost_p_tonmi', 'diesel_cost_p_gal',
            'comm_group', 'flow_data_filename', 'year', 'suppress_output', 'legend_show', 'scenario_code']

    df_scenario = load_scenario_df(scenario_code=scenario_code)

    [rr, fuel_type, deployment_perc, range_km, max_flow, budget, extend_graph, reroute, switch_tech, max_reroute_inc,
     max_util, station_type, h2_fuel_type, clean_energy, clean_energy_cost, emissions_obj, eff_energy_p_tender,
     tender_cost_p_tonmi, diesel_cost_p_gal, comm_group, flow_data_filename, year, suppress_output, legend_show,
     scenario_code] = df_scenario.reindex(idxs)

    # fuel_types that require infrastructure to be located; need specific set of parameters to be provided
    infra_fuel_set = {'battery', 'hydrogen', 'hybrid2:1', 'hybrid1:1'}

    # verify validity of inputs
    # rr: railroad selection
    assert rr in {'BNSF', 'UP', 'NS', 'CSXT', 'KCS', 'CN', 'CP', 'USA1', 'WCAN', 'EAST'}, \
        'Provide a railroad selection <rr>.'
    # fuel_type
    assert fuel_type in {'battery', 'hydrogen', 'diesel', 'e-fuel', 'biodiesel', 'hybrid2:1', 'hybrid1:1'}, \
        'Provide an energy source selection <fuel_type>.'
    # deployment_perc
    assert isinstance(deployment_perc, float) or isinstance(deployment_perc, int) and 0 <= deployment_perc <= 1, \
        'Provide a valid input for <deployment_perc>.'
    assert 0 <= deployment_perc <= 1, 'Provide a valid value range for <deployment_perc>.'
    # range_km
    if fuel_type in infra_fuel_set:
        assert isinstance(range_km, float) or isinstance(range_km, int), \
            'Provide a valid input for energy source <range_km>.'
        assert range_km > 0, 'Provide a positive value for <range_km>.'
    # max_flow
    if fuel_type in infra_fuel_set:
        assert isinstance(max_flow, bool) or max_flow in {0, 1}, \
            'Provide a valid input for the <max_flow> facility location parameter.'
    # budget
    assert isinstance(budget, float) or isinstance(budget, int) or budget is None, \
        'Provide a valid input for the <budget> facility location parameter.'
    # extend_graph
    if fuel_type in infra_fuel_set:
        assert isinstance(extend_graph, bool) or extend_graph in {0, 1}, \
            'Provide a valid input for the <extend_graph> network construction parameter.'
    # reroute
    if fuel_type in infra_fuel_set:
        assert isinstance(reroute, bool) or reroute in {0, 1} or reroute is None, \
            'Provide a valid input for the <reroute> routing parameter.'
    # switch_tech
    if fuel_type in infra_fuel_set:
        assert isinstance(switch_tech, bool) or switch_tech in {0, 1} or switch_tech is None, \
            'Provide a valid input for the <switch_tech> technology switching parameter.'
    # max_reroute_inc
    if fuel_type in infra_fuel_set:
        assert isinstance(max_reroute_inc, float) or isinstance(max_reroute_inc, int), \
            'Provide a valid input for the <max_reroute_inc> maximum rerouting distance increase parameter.'
        assert max_reroute_inc >= 0, 'Values for <max_reroute_inc> parameter must be non-negative.'
    # max_util
    if fuel_type in infra_fuel_set:
        assert isinstance(max_util, float) or isinstance(max_util, int) and 0 <= max_util <= 1, \
            'Provide a valid input for the <max_util> maximum station utilization.'
    # station_type
    if fuel_type in infra_fuel_set:
        if fuel_type == 'battery' or 'hybrid' in fuel_type:
            assert station_type == '3MW', \
                'A <station_type> of ' + station_type + ' is not adequate for the <fuel_type> selected.'
        else:
            assert station_type in {'Cryo-pump', '700 via pump', '350 via pump'}, \
                'A <station_type> of ' + station_type + ' is not adequate for the <fuel_type> selected.'
    # h2_fuel_type
    if fuel_type == 'hydrogen':
        assert h2_fuel_type in {'Natural Gas', 'NG with CO2 Sequestration',
                                'PEM Electrolysis - Solar', 'PEM Electrolysis - Nuclear'}, \
            'Provide a valid input for the <h2_fuel_type> hydrogen fuel pathway parameter.'
    # clean_energy
    if fuel_type in infra_fuel_set:
        assert isinstance(clean_energy, bool) or clean_energy in {0, 1}, \
            'Provide a valid input for <clean_energy> selection parameter.'
        clean_energy = bool(clean_energy)
    # clean_energy_cost
    if fuel_type in infra_fuel_set:
        assert isinstance(clean_energy_cost, float) or isinstance(clean_energy_cost, int), \
            'Provide a valid input for the cost of clean energy <clean_energy_cost>.'
        if np.isnan(clean_energy_cost):
            clean_energy_cost = None
    # emissions_obj
    if fuel_type == 'battery':
        assert isinstance(emissions_obj, bool) or emissions_obj in {0, 1}, \
            'Provide a valid input for the <emissions_obj> facility sizing parameter.'
    # eff_energy_p_tender
    assert eff_energy_p_tender is None or (isinstance(eff_energy_p_tender, float) or
                                           isinstance(eff_energy_p_tender, int) and eff_energy_p_tender > 0), \
        'Provide a valid input for the <eff_energy_p_tender> effective energy storage per tender car.'
    if eff_energy_p_tender is None and fuel_type == 'hydrogen':
        eff_energy_p_tender = 4000  # kgh2/tender car
    elif eff_energy_p_tender is None and fuel_type in infra_fuel_set:
        eff_energy_p_tender = 10000  # kWh/tender car
    # tender_cost_p_tonmi
    if fuel_type in infra_fuel_set:
        assert tender_cost_p_tonmi is None or (isinstance(tender_cost_p_tonmi, float) or
                                               isinstance(tender_cost_p_tonmi, int) and tender_cost_p_tonmi >= 0), \
            'Provide a valid input for <tender_cost_p_tonmi> tender car amortized cost.'
        if np.isnan(tender_cost_p_tonmi):
            tender_cost_p_tonmi = None
    # diesel_cost_p_gal
    assert diesel_cost_p_gal is None or (isinstance(diesel_cost_p_gal, float) or isinstance(diesel_cost_p_gal, int)
                                         and diesel_cost_p_gal >= 0), \
        'Provide a valid input for <diesel_cost_p_gal> diesel fuel cost per gallon.'
    if np.isnan(diesel_cost_p_gal):
        diesel_cost_p_gal = None
    # comm_group
    assert isinstance(comm_group, str) and \
           comm_group in {'OTHER', 'TOTAL', 'MO_VEH', 'AG_FOOD', 'IM', 'COAL',
                          'CHEM_PET', 'NONMET_PRO', 'FOR_PROD', 'MET_ORE'},\
        'Provide a valid commodity group.'
    # flow_data_filename
    assert os.path.exists(os.path.join(FLOW_DIR, flow_data_filename)), 'File does not exist.'
    # year
    assert isinstance(year, float) or isinstance(year, int), 'Provide a valid input for <year> for flow data.'
    # suppress_output
    assert isinstance(suppress_output, bool) or suppress_output in {0, 1}, \
        'Provide a valid input for <suppress_output> optimization solver log parameter.'
    suppress_output = bool(suppress_output)
    # legend_show
    assert isinstance(legend_show, bool) or legend_show in {0, 1}, \
        'Provide a valid input for <legend_show> plotting legend switch parameter.'
    legend_show = bool(legend_show)
    # scenario_code
    assert isinstance(scenario_code, str), \
        'Provide a valid input for the <scenario_code> input and output file naming code.'
    assert os.path.exists(os.path.join(SCENARIO_DIR, scenario_code + '.csv')), \
        'Scenario parameter filename does not match <scenario_code> provided.'

    return [rr, fuel_type, deployment_perc, range_km, max_flow, budget, extend_graph, reroute, switch_tech,
            max_reroute_inc, max_util, station_type, h2_fuel_type, clean_energy, clean_energy_cost, emissions_obj,
            eff_energy_p_tender, tender_cost_p_tonmi, diesel_cost_p_gal, comm_group, flow_data_filename,
            suppress_output, legend_show, scenario_code]


def extract_assert_scenario_mp_inputs(scenario_code: str):
    idxs = ['rr', 'fuel_type', 'range_km', 'max_flow',
            'time_horizon', 'deployment_percs', 'budgets', 'discount_rates', 'facility_info_filename',
            'max_util', 'station_type', 'h2_fuel_type', 'clean_energy', 'clean_energy_cost', 'emissions_obj',
            'eff_energy_p_tender', 'tender_cost_p_tonmi', 'diesel_cost_p_gal', 'flow_data_filename',
            'suppress_output', 'opt_tol', 'scenario_code']

    df_scenario = load_scenario_mp_df(scenario_code=scenario_code)

    time_inputs = list(df_scenario.columns)
    for i, c in enumerate(time_inputs):
        assert c == 'Value' + str(i), 'Provide valid column names for time-dependent inputs. ' \
                                      'Must be in format of [<Value0>, <Value1>, ..., <ValueT>].'

    [rr, fuel_type, range_km, max_flow,
     _, _, _, _, facility_info_filename,
     max_util, station_type, h2_fuel_type, clean_energy, clean_energy_cost, emissions_obj,
     eff_energy_p_tender, tender_cost_p_tonmi, diesel_cost_p_gal, flow_data_filename,
     suppress_output, opt_tol, scenario_code] = df_scenario.reindex(idxs)['Value0']

    # verify validity of inputs
    # rr: railroad selection
    assert rr in {'BNSF', 'UP', 'NS', 'CSXT', 'KCS', 'CN', 'CP', 'USA1', 'WCAN', 'EAST'}, \
        'Provide a railroad selection <rr>.'
    # fuel_type
    assert fuel_type in {'battery', 'hydrogen'}, 'Provide an energy source selection <fuel_type>. ' \
                                                 'Only <battery> or <hydrogen> supported.'
    # range_km
    assert isinstance(range_km, str) or isinstance(range_km, float) or isinstance(range_km, int), \
        'Provide a valid input for energy source <range_km>.'
    if isinstance(range_km, str):
        range_km = float(range_km)
    assert range_km > 0, 'Provide a positive value for <range_km>.'
    # max_flow
    assert isinstance(max_flow, bool) or max_flow in {0, 1}, \
        'Provide a valid input for the <max_flow> facility location parameter.'
    max_flow = bool(max_flow)
    # time_horizon
    time_horizon = list(df_scenario.loc['time_horizon', time_inputs])
    for ti, t in enumerate(time_horizon):
        assert isinstance(t, int) or isinstance(t, float) or isinstance(t, str), \
            'Provide a valid input type for <time_horizon>.'
        if isinstance(t, int) or isinstance(t, float):
            time_horizon[ti] = str(int(t))
    # deployment_percs
    deployment_percs = list(df_scenario.loc['deployment_percs', time_inputs])
    for di, d in enumerate(deployment_percs):
        assert isinstance(d, float) or isinstance(d, int) and 0 <= d <= 1, \
            'Provide a valid input for <deployment_percs>.'
        if np.isnan(d):
            if di != 0:
                deployment_percs[di] = deployment_percs[di - 1]
            else:
                deployment_percs[di] = 0
    assert deployment_percs == sorted(deployment_percs), 'Provide a valid ordering of <deployment_percs>.'
    if not max_flow:
        assert deployment_percs[-1] > 0, 'Provide a non-zero value for final input of <deployment_percs>.'
    deployment_percs = {t: deployment_percs[ti] for ti, t in enumerate(time_horizon)}
    # budget
    budgets = list(df_scenario.loc['budgets', time_inputs])
    for bi, b in enumerate(budgets):
        assert isinstance(b, float) or isinstance(b, int), \
            'Provide a valid input for the <budget> facility location parameter.'
        if np.isnan(b):
            if bi != 0:
                budgets[bi] = budgets[bi - 1]
            else:
                budgets[bi] = None
    if max_flow:
        assert not all([b is None for b in budgets]), 'Provide a non-empty value for inputs of <budgets>.'
        assert budgets[-1] > 0, 'Provide a non-zero value for final input of <budgets>.'
    assert budgets == sorted(budgets), 'Provide a valid ordering of <budgets>.'
    budgets = {t: budgets[ti] for ti, t in enumerate(time_horizon)}
    # discount_rates
    discount_rates = list(df_scenario.loc['discount_rates', time_inputs])
    for di, d in enumerate(discount_rates):
        assert isinstance(d, float) or isinstance(d, int) and 0 <= d, \
            'Provide a valid input for <discount_rates>.'
        if np.isnan(d):
            if di != 0:
                discount_rates[di] = discount_rates[di - 1]
            else:
                discount_rates[di] = 1
    assert not all(d == 0 for d in discount_rates), \
        'Provide a non-zero value for at least one input of <discount_rates>.'
    discount_rates = {t: discount_rates[ti] for ti, t in enumerate(time_horizon)}
    # facility_info_filename
    assert isinstance(facility_info_filename, str) or np.isnan(facility_info_filename),\
        'Provide a valid input for <facility_file_name>.'
    if not isinstance(facility_info_filename, str):
        facility_info_filename = None
    facility_costs, fixed_facilities, barred_facilities = load_facility_info(facility_info_filename=
                                                                             facility_info_filename, rr=rr,
                                                                             time_horizon=time_horizon)
    # max_util
    assert isinstance(max_util, float) or isinstance(max_util, int) and 0 <= max_util <= 1, \
        'Provide a valid input for the <max_util> maximum station utilization.'
    # station_type
    if fuel_type == 'battery':
        assert station_type == '3MW', \
            'A <station_type> of ' + station_type + ' is not adequate for the <fuel_type> selected.'
    else:
        assert station_type in {'Cryo-pump', '700 via pump', '350 via pump'}, \
            'A <station_type> of ' + station_type + ' is not adequate for the <fuel_type> selected.'
    # h2_fuel_type
    if fuel_type == 'hydrogen':
        assert h2_fuel_type in {'Natural Gas', 'NG with CO2 Sequestration',
                                'PEM Electrolysis - Solar', 'PEM Electrolysis - Nuclear'}, \
            'Provide a valid input for the <h2_fuel_type> hydrogen fuel pathway parameter.'
    # clean_energy
    assert isinstance(clean_energy, bool) or clean_energy in {0, 1}, \
        'Provide a valid input for <clean_energy> selection parameter.'
    clean_energy = bool(clean_energy)
    # clean_energy_cost
    assert isinstance(clean_energy_cost, float) or isinstance(clean_energy_cost, int), \
        'Provide a valid input for the cost of clean energy <clean_energy_cost>.'
    if np.isnan(clean_energy_cost):
        clean_energy_cost = None
    # emissions_obj
    if fuel_type == 'battery':
        assert isinstance(emissions_obj, bool) or emissions_obj in {0, 1}, \
            'Provide a valid input for the <emissions_obj> facility sizing parameter.'
    # eff_energy_p_tender
    assert eff_energy_p_tender is None or (isinstance(eff_energy_p_tender, float) or
                                           isinstance(eff_energy_p_tender, int) and eff_energy_p_tender > 0), \
        'Provide a valid input for the <eff_energy_p_tender> effective energy storage per tender car.'
    if eff_energy_p_tender is None and fuel_type == 'hydrogen':
        eff_energy_p_tender = 4000  # kgh2/tender car
    elif eff_energy_p_tender is None and fuel_type == 'battery':
        eff_energy_p_tender = 10000  # kWh/tender car
    # tender_cost_p_tonmi
    assert tender_cost_p_tonmi is None or (isinstance(tender_cost_p_tonmi, float) or
                                           isinstance(tender_cost_p_tonmi, int) and tender_cost_p_tonmi >= 0), \
        'Provide a valid input for <tender_cost_p_tonmi> tender car amortized cost.'
    if np.isnan(tender_cost_p_tonmi):
        tender_cost_p_tonmi = None
    # diesel_cost_p_gal
    assert (isinstance(diesel_cost_p_gal, float) or isinstance(diesel_cost_p_gal, int) and diesel_cost_p_gal >= 0), \
        'Provide a valid input for <diesel_cost_p_gal> diesel fuel cost per gallon.'
    if np.isnan(diesel_cost_p_gal):
        diesel_cost_p_gal = None
    # flow_data_filename
    assert os.path.exists(os.path.join(FLOW_DIR, flow_data_filename)), 'File does not exist.'
    # suppress_output
    assert isinstance(suppress_output, bool) or suppress_output in {0, 1}, \
        'Provide a valid input for <suppress_output> optimization solver log parameter.'
    suppress_output = bool(suppress_output)
    # opt_tol
    assert np.isnan(opt_tol) or isinstance(opt_tol, float) or isinstance(opt_tol, int), \
        'Provide a valid input for <opt_tol>.'
    if np.isnan(opt_tol) or opt_tol <= 0:
        opt_tol = None
    # scenario_code
    assert isinstance(scenario_code, str), \
        'Provide a valid input for the <scenario_code> input and output file naming code.'
    assert os.path.exists(os.path.join(SCENARIO_DIR, scenario_code + '.csv')), \
        'Scenario parameter filename does not match <scenario_code> provided.'

    return [rr, fuel_type, range_km, max_flow,
            time_horizon, deployment_percs, budgets, discount_rates,
            facility_costs, fixed_facilities, barred_facilities,
            max_util, station_type, h2_fuel_type, clean_energy, clean_energy_cost, emissions_obj,
            eff_energy_p_tender, tender_cost_p_tonmi, diesel_cost_p_gal, flow_data_filename,
            suppress_output, opt_tol, scenario_code]


def load_facility_info(facility_info_filename: str, rr: str, time_horizon):

    G = load_simplified_consolidated_graph(rr=rr)
    node_list = list(G.nodes)
    facility_costs = {(n, t): 1 for t in time_horizon for n in node_list}
    fixed_facilities = {t: [] for t in time_horizon}
    barred_facilities = {t: [] for t in time_horizon}

    if facility_info_filename and os.path.exists(os.path.join(FACILITY_DIR, facility_info_filename)):
        df = pd.read_csv(os.path.join(FACILITY_DIR, facility_info_filename), header=0, index_col='nodeid')
        assert len(df.columns) / 3 == len(time_horizon), \
            'Provide the correct number of time-dependent columns in <facility_info_filename>.'

        for ti, t in enumerate(time_horizon):
            tidx = str(ti)
            for n in df.index:
                if n not in node_list:
                    continue
                facility_costs[(n, t)] = float(df.loc[n, 'cost' + tidx])
                if int(df.loc[n, 'fixed' + tidx]) == 1:
                    fixed_facilities[t].append(n)
                if int(df.loc[n, 'barred' + tidx]) == 1:
                    barred_facilities[t].append(n)
                assert not (int(df.loc[n, 'fixed' + tidx]) == 1 and int(df.loc[n, 'barred' + tidx]) == 1), \
                    'A single facility {n} cannot be both fixed and barred at the same time step {t}.'.format(n=n, t=ti)

    return facility_costs, fixed_facilities, barred_facilities


def load_dict_from_json(filepath: str):
    # store dict <d> as json file in filepath
    with open(filepath, 'r') as fr:
        return json.load(fr)


def dict_to_json(d: dict, filepath: str):
    # store dict <d> as json file in filepath
    with open(filepath, 'w') as fw:
        json.dump(d, fw)


# CACHE/LOAD GRAPH


def cache_metrics(G: nx.DiGraph, scenario_code: str):
    metrics = G.graph.copy()
    metrics['crs'] = str(G.graph['crs'])  # replace 'crs' val from crs type to str as crs is not JSON serializable

    filepath = os.path.join(MET_O_DIR, scenario_code + '.json')
    with open(filepath, 'w') as fp:
        json.dump(metrics, fp)
    fp.close()


def cache_graph(G: nx.DiGraph, scenario_code: str, cache_nodes=True, cache_edges=True):
    filepath_nodes = os.path.join(NODE_O_DIR, scenario_code + '_nodes.json')
    filepath_edges = os.path.join(EDGE_O_DIR, scenario_code + '_edges.json')
    # filepath_nodes_geo = os.path.join(NODE_O_DIR, codified_name + '_nodes.geojson')
    # filepath_edges_geo = os.path.join(EDGE_O_DIR, codified_name + '_edges.geojson')

    gdf_nodes, gdf_edges = gdfs_from_graph(G, nodes=cache_nodes, edges=cache_edges)

    cache_metrics(G=G, scenario_code=scenario_code)

    if cache_nodes:
        gdf_nodes.reset_index(inplace=True, names='supernodeid')
        df_nodes = pd.DataFrame(gdf_nodes.drop(columns=['geometry']))
        df_nodes.to_json(filepath_nodes, index=True)

    if cache_edges:
        gdf_edges.reset_index(inplace=True)
        df_edges = pd.DataFrame(gdf_edges.drop(columns=['geometry']))
        df_edges.to_json(filepath_edges, index=True)


def load_scenario_metrics(scenario_code: str):
    filepath = os.path.join(MET_O_DIR, scenario_code + '.json')
    if os.path.exists(filepath):
        fp = open(filepath, 'r')
        metrics = json.load(fp)
        fp.close()
        return metrics
    else:
        return None


def cache_exists(scenario_code: str) -> bool:
    # return if cached scenario exists
    filepath_nodes = os.path.join(NODE_O_DIR, scenario_code + '_nodes.json')
    filepath_edges = os.path.join(EDGE_O_DIR, scenario_code + '_edges.json')

    return os.path.exists(filepath_nodes) and os.path.exists(filepath_edges)


def load_cached_graph(scenario_code: str):
    filepath_nodes = os.path.join(NODE_O_DIR, scenario_code + '_nodes.json')
    filepath_edges = os.path.join(EDGE_O_DIR, scenario_code + '_edges.json')

    df_scenario = load_scenario_df(scenario_code=scenario_code)

    if os.path.exists(filepath_nodes):
        # load railroad geometry data
        gdf_nodes, gdf_edges = gdfs_from_graph(load_simplified_consolidated_graph(df_scenario.loc['rr']))
        for u, v in gdf_edges.index:
            gdf_edges.loc[(v, u), 'geometry'] = gdf_edges.loc[(u, v), 'geometry']

        # load scenario edge data
        df_nodes = pd.read_json(filepath_nodes)
        df_nodes.index = df_nodes['supernodeid']
        gdf_nodes.drop(columns=set(gdf_nodes.columns).difference({'geometry'}), inplace=True)
        gdf_nodes = pd.concat([df_nodes, gdf_nodes], axis='columns')

    else:
        return None

    if os.path.exists(filepath_edges):
        df_edges = pd.read_json(filepath_edges)
        df_edges = df_edges.groupby(by=['u', 'v'], as_index=True).first()
        gdf_edges.drop(columns=set(gdf_edges.columns).difference({'geometry'}), inplace=True)
        gdf_edges = pd.concat([df_edges, gdf_edges], axis='columns')

    else:
        return None

    return digraph_from_gdfs(gdf_nodes, gdf_edges, load_scenario_metrics(scenario_code))
