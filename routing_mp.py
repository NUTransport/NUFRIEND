from util import *
# MODULES
from helper import splc_to_node, node_to_edge_path, \
    load_comm_energy_ratios, load_conversion_factors, load_fuel_tech_eff_factor, \
    load_railroad_values, extract_rr, load_flow_data_df_csv, load_flow_data_date_df_csv
from input_output import load_dict_from_json, dict_to_json

'''
ROUTING METHODS
'''


def route_flows_mp(G: nx.DiGraph, D: float, time_horizon: list, od_flows: dict, fuel_type: str):
    edge_list = list(G.edges)
    edge_list.extend([(v, u) for u, v in edge_list])  # both directions
    od_list = list(od_flows.keys())
    t0 = time.time()
    pli_mat = path_link_incidence_mat_mp(G=G, od_list=od_list, edge_list=edge_list)
    print('PLI MATRIX:: {v0} seconds'.format(v0=time.time() - t0))
    # need: G.edges[u, v][<time_period>][<key> = 'baseline_avg_ton', ..., ][<comm> = 'COAL', ..., 'TOTAL'] = <val>
    #       for <key> = 'baseline_avg_ton', 'baseline_avg_loc', 'baseline_avg_gal',
    #               'support_diesel_avg_ton', 'support_diesel_avg_loc', 'support_diesel_avg_gal',
    #               'battery_avg_ton', 'battery_avg_loc', 'battery_avg_kwh',
    #               'hydrogen_avg_ton', 'hydrogen_avg_loc', 'hydrogen_avg_kgh2'

    # od_comm_flows = {<time_period>: {<comm>: np.array([od_flows vals in order of od_list])
    #                                  for <comm> in <comm_groups>} for <time_period> in <time_horizon>}
    od_comm_flows = ods_ton_comm_forecast(G=G, od_list=od_list)
    # comm group list; does not include 'TOTAL'

    rr = G.graph['railroad']
    # lookup dataframes for constants
    rr_v = load_railroad_values().loc[rr]  # railroad energy intensity statistics
    cf = load_conversion_factors()['Value']  # numerical constants for conversion across units
    # arrays ordered in same order as <comm_list> and stored as np arrays for vectorization
    rr_tc = rr_v['ton/car']  # tons/car
    comm_er = load_comm_energy_ratios()['Weighted ratio']  # commodity energy ratios (indexed by <comm_group>)
    # car2loc = loc/train * train/car- not adjusted by commodity
    car2loc = rr_v['loc/train'] * (1 / rr_v['car/train'])

    # alt. tech.
    ft_ef = load_fuel_tech_eff_factor().loc[fuel_type]  # fuel tech efficiency factors
    if fuel_type == 'battery':
        tonmi2energy = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kwh']) *
                        (1 / rr_v['Energy correction factor']) * (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))
    elif fuel_type == 'hydrogen':
        tonmi2energy = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kgh2']) *
                        (1 / rr_v['Energy correction factor']) * (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))
    loc2energy = tonmi2energy * rr_v['ton/loc'] * D * cf['mi/km']
    # tonmi2gal = btu/ton-mi * gal/btu * <energy_correction> = gal/ton-mi- not adjusted by commodity
    tonmi2gal = rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/gal']) * (1 / rr_v['Energy correction factor'])
    # tonmi2loc = kWh/ton-mi * loc/kWh = loc/tonmi - not adjusted by commodity
    tonmi2loc = tonmi2energy * (1 / loc2energy)

    comm_list = list(od_comm_flows[time_horizon[0]].keys()) + ['TOTAL']
    for e in edge_list:
        # instantiate dict storage objects
        G.edges[e]['baseline_avg_ton'] = {t: {c: 0 for c in comm_list} for t in time_horizon}
        G.edges[e]['baseline_avg_loc'] = {t: {c: 0 for c in comm_list} for t in time_horizon}
        G.edges[e]['baseline_avg_gal'] = {t: {c: 0 for c in comm_list} for t in time_horizon}
        G.edges[e][fuel_type + '_avg_ton'] = {t: {c: 0 for c in comm_list} for t in time_horizon}
        G.edges[e][fuel_type + '_avg_loc'] = {t: {c: 0 for c in comm_list} for t in time_horizon}
        if fuel_type == 'battery':
            G.edges[e][fuel_type + '_avg_kwh'] = {t: {c: 0 for c in comm_list} for t in time_horizon}
        elif fuel_type == 'hydrogen':
            G.edges[e][fuel_type + '_avg_kgh2'] = {t: {c: 0 for c in comm_list} for t in time_horizon}
        G.edges[e]['support_diesel_avg_ton'] = {t: {c: 0 for c in comm_list} for t in time_horizon}
        G.edges[e]['support_diesel_avg_loc'] = {t: {c: 0 for c in comm_list} for t in time_horizon}
        G.edges[e]['support_diesel_avg_gal'] = {t: {c: 0 for c in comm_list} for t in time_horizon}

    # flow assignment by <time_period> and <comm_group>
    for t in time_horizon:
        t1 = time.time()
        selected_ods = G.graph['framework']['selected_ods'][t]
        selected_ods = selected_ods.union({(d, o) for o, d in selected_ods})
        for c in comm_list[:-1]:
            edge_tons = pli_mat @ od_comm_flows[t][c]
            # Alt. Tech.
            # determine which OD pairs are captured - set flows of all those not captured to 0
            od_flows_capt = np.array([f if od_list[od_idx] in selected_ods else 0
                                      for od_idx, f in enumerate(od_comm_flows[t][c])])
            edge_tons_alt_tech = pli_mat @ od_flows_capt
            # Support Diesel
            # complement of captured flows, i.e., baseline - alt. tech.
            od_flows_not_capt = np.array([0 if od_list[od_idx] in selected_ods else f
                                          for od_idx, f in enumerate(od_comm_flows[t][c])])
            edge_tons_sd = pli_mat @ od_flows_not_capt
            for e_idx, e_ton in enumerate(edge_tons):
                if e_ton == 0:
                    continue
                e = edge_list[e_idx]
                mi = G.edges[e]['miles']
                # Baseline:
                G.edges[e]['baseline_avg_ton'][t][c] = e_ton
                # loc = loc/car * <commodity_car/ton> * ton
                G.edges[e]['baseline_avg_loc'][t][c] = np.ceil(car2loc * (1 / rr_tc) * e_ton)
                # gal = gal/ton-mi * <commodity_factor> * ton-mi
                G.edges[e]['baseline_avg_gal'][t][c] = tonmi2gal * comm_er.loc[c] * e_ton * mi
                # incrementally sum all comm_groups up to get values for 'TOTAL'
                G.edges[e]['baseline_avg_ton'][t]['TOTAL'] += e_ton
                G.edges[e]['baseline_avg_loc'][t]['TOTAL'] += np.ceil(car2loc * (1 / rr_tc) * e_ton)
                G.edges[e]['baseline_avg_gal'][t]['TOTAL'] += tonmi2gal * comm_er.loc[c] * e_ton * mi
                # ----------
                # Alt. Tech.
                # determine which OD pairs are captured - set flows of all those not captured to 0
                e_ton_alt_tech = edge_tons_alt_tech[e_idx]  # edit
                G.edges[e][fuel_type + '_avg_ton'][t][c] = e_ton_alt_tech
                # loc = loc/car * <commodity_car/ton> * ton
                G.edges[e][fuel_type + '_avg_loc'][t][c] = np.ceil(tonmi2loc * comm_er.loc[c] * e_ton_alt_tech * mi)
                # incrementally sum all comm_groups up to get values for 'TOTAL'
                G.edges[e][fuel_type + '_avg_ton'][t]['TOTAL'] += e_ton_alt_tech
                G.edges[e][fuel_type + '_avg_loc'][t]['TOTAL'] += np.ceil(tonmi2loc * comm_er.loc[c] *
                                                                          e_ton_alt_tech * mi)
                if fuel_type == 'battery':
                    # kwh = kwh/ton-mi * <commodity_factor> * ton-mi
                    G.edges[e][fuel_type + '_avg_kwh'][t][c] = tonmi2energy * comm_er.loc[c] * e_ton_alt_tech * mi
                    # incrementally sum all comm_groups up to get values for 'TOTAL'
                    G.edges[e][fuel_type + '_avg_kwh'][t]['TOTAL'] += tonmi2energy * comm_er.loc[
                        c] * e_ton_alt_tech * mi
                elif fuel_type == 'hydrogen':
                    # kgh2 = kgh2/ton-mi * <commodity_factor> * ton-mi
                    G.edges[e][fuel_type + '_avg_kgh2'][t][c] = tonmi2energy * comm_er.loc[c] * e_ton_alt_tech * mi
                    # incrementally sum all comm_groups up to get values for 'TOTAL'
                    G.edges[e][fuel_type + '_avg_kgh2'][t]['TOTAL'] += (tonmi2energy * comm_er.loc[c] *
                                                                        e_ton_alt_tech * mi)
                # --------------
                # Support Diesel
                # complement of captured flows, i.e., baseline - alt. tech.
                e_ton_sd = edge_tons_sd[e_idx]
                G.edges[e]['support_diesel_avg_ton'][t][c] = e_ton_sd
                # loc = loc/car * <commodity_car/ton> * ton
                G.edges[e]['support_diesel_avg_loc'][t][c] = np.ceil(car2loc * (1 / rr_tc) * e_ton_sd)
                # gal = gal/ton-mi * <commodity_factor> * ton-mi
                G.edges[e]['support_diesel_avg_gal'][t][c] = tonmi2gal * comm_er.loc[c] * e_ton_sd * mi
                # incrementally sum all comm_groups up to get values for 'TOTAL'
                G.edges[e]['support_diesel_avg_ton'][t]['TOTAL'] += e_ton_sd
                G.edges[e]['support_diesel_avg_loc'][t]['TOTAL'] += np.ceil(car2loc * (1 / rr_tc) * e_ton_sd)
                G.edges[e]['support_diesel_avg_gal'][t]['TOTAL'] += tonmi2gal * comm_er.loc[c] * e_ton_sd * mi
        print('\t EDGE ASSIGNMENT {v0}:: {v1} seconds'.format(v0=t, v1=time.time() - t1))

    # all values here are annual
    G.graph['operations'] = dict(
        baseline_total_annual_tonmi={
            t: {c: sum(G.edges[e]['baseline_avg_ton'][t][c] * G.edges[e]['miles'] for e in edge_list)
                for c in comm_list} for t in time_horizon},
        baseline_total_annual_gal={t: {c: sum(G.edges[e]['baseline_avg_gal'][t][c] for e in edge_list)
                                       for c in comm_list} for t in time_horizon},
        alt_tech_total_annual_tonmi={t: {c: sum(G.edges[e][fuel_type + '_avg_ton'][t][c] * G.edges[e]['miles']
                                                for e in edge_list) for c in comm_list} for t in time_horizon},
        support_diesel_total_annual_tonmi={t: {c: sum(G.edges[e]['support_diesel_avg_ton'][t][c] * G.edges[e]['miles']
                                                      for e in edge_list) for c in comm_list} for t in time_horizon},
        support_diesel_total_annual_gal={t: {c: sum(G.edges[e]['support_diesel_avg_gal'][t][c] for e in edge_list)
                                             for c in comm_list} for t in time_horizon},
    )

    G.graph['operations'].update(dict(
        deployment_perc={t: {c: (G.graph['operations']['alt_tech_total_annual_tonmi'][t][c] /
                                 G.graph['operations']['baseline_total_annual_tonmi'][t][c])
                             for c in comm_list} for t in time_horizon},
        # baseline_total_annual_tonmi={t: {c: 365 * G.graph['operations']['baseline_total_tonmi'][t][c]
        #                                  for c in comm_list} for t in time_horizon},
        # baseline_total_annual_gal={t: {c: 365 * G.graph['operations']['baseline_total_gal'][t][c]
        #                                for c in comm_list} for t in time_horizon},
        # alt_tech_total_annual_tonmi={t: {c: 365 * G.graph['operations']['alt_tech_total_tonmi'][t][c]
        #                                  for c in comm_list} for t in time_horizon},
        # support_diesel_total_annual_tonmi={t: {c: 365 * G.graph['operations']['support_diesel_total_tonmi'][t][c]
        #                                        for c in comm_list} for t in time_horizon},
        # support_diesel_total_annual_gal={t: {c: 365 * G.graph['operations']['support_diesel_total_gal'][t][c]
        #                                      for c in comm_list} for t in time_horizon},

    ))

    if fuel_type == 'battery':
        G.graph['operations'].update(dict(
            alt_tech_total_annual_kwh={t: {c: sum(G.edges[e][fuel_type + '_avg_kwh'][t][c] for e in edge_list)
                                           for c in comm_list} for t in time_horizon},
            # alt_tech_total_annual_kwh={t: {c: 365 * sum(G.edges[e][fuel_type + '_avg_kwh'][t][c] for e in edge_list)
            #                                for c in comm_list} for t in time_horizon},
            eff_kwh_p_loc=loc2energy,
            listed_kwh_p_loc=loc2energy * (1 / ft_ef['Effective capacity'])
        ))
    elif fuel_type == 'hydrogen':
        G.graph['operations'].update(dict(
            alt_tech_total_annual_kgh2={t: {c: sum(G.edges[e][fuel_type + '_avg_kgh2'][t][c] for e in edge_list)
                                            for c in comm_list} for t in time_horizon},
            # alt_tech_total_annual_kgh2={t: {c: 365 * sum(G.edges[e][fuel_type + '_avg_kgh2'][t][c] for e in edge_list)
            #                                 for c in comm_list} for t in time_horizon},
            eff_kgh2_p_loc=loc2energy,
            listed_kgh2_p_loc=loc2energy * (1 / ft_ef['Effective capacity'])
        ))

    return G


def path_link_incidence_mat_mp(G: nx.DiGraph, od_list: list, edge_list: list):
    # od_flows is time-indexed as well now; want union of all ods with >0 flow
    # od_list = list(od_flows.keys())

    edge_idx_dict = {v: i for i, v in enumerate(edge_list)}

    # precompute shortest path for all OD pairs (since Dijkstra's finds one-to-all shortest paths)
    sp_dict = dict(nx.all_pairs_dijkstra_path(G, weight='km'))
    pli_data = []
    pli_rows = []
    pli_cols = []

    for od_idx, (o, d) in enumerate(od_list):
        path_edges = node_to_edge_path(sp_dict[o][d])
        pli_data.extend([1 for _ in range(len(path_edges))])
        pli_rows.extend([edge_idx_dict[e] for e in path_edges])
        pli_cols.extend([od_idx for _ in range(len(path_edges))])

    pli_mat = csr_matrix((pli_data, (pli_rows, pli_cols)), shape=(len(edge_list), len(od_list)))

    return pli_mat


'''
DEPLOYMENT PERCENTGE METHODS
'''


def ods_by_perc_ton_mi(G: nx.DiGraph, perc_ods: float, CCWS_filename: str = None, time_window: tuple = None,
                       od_flows_truncate=False):
    """

    Parameters
    ----------
    G
    perc_ods
    CCWS_filename
    time_window

    Returns
    -------

    """
    # return O-D pairs in CCWS tha provide ton flows >= <perc_ods> * total CCWS ton flows
    # od_flows is average daily ton-miles

    if CCWS_filename is None:
        CCWS_filename = FILES[2019]

    # load dict that maps SPLC codes to node_ids in G
    splc_node_dict = splc_to_node(G)
    # load grouped OD flow data
    # TODO: fix to be for time-dependent data
    flow_df = RR_SPLC_comm_grouping(filename=CCWS_filename, time_window=time_window)
    # filter out specific railroad
    rr = G.graph['railroad']
    flow_df = extract_rr(flow_df, rr)
    # only index needed is the OD pair
    flow_df.reset_index(level='Commodity Group Name', inplace=True)
    # filter out OD pairs that are not in the splc_node_dict keys
    splc_set = set(splc_node_dict.keys())
    remove_idxs = list({i for i in flow_df.index.unique() if i[1:7] not in splc_set or i[7:] not in splc_set})
    flow_df.drop(index=remove_idxs, inplace=True)

    # outcomment to have O-D flows counted bidirectionally
    # od_keys = set()
    # rev_od_strs = dict()
    # for od_str in flow_df.index:
    #     rev_od_str = od_str[0] + od_str[7:] + od_str[1:7]
    #     if rev_od_str not in od_keys:
    #         rev_od_strs[od_str] = rev_od_str
    #         od_keys.add(od_str)
    # flow_df.rename(index=rev_od_strs, inplace=True)

    # assign each SPLC OD to its respective nodeid in G
    flow_df.reset_index(level='Origin-Destination SPLC', inplace=True)
    flow_df['Origin-Destination nodeid'] = flow_df['Origin-Destination SPLC'].apply(lambda x:
                                                                                    (splc_node_dict[x[1:7]],
                                                                                     splc_node_dict[x[7:]]))

    flow_df = flow_df.groupby(by=['Origin-Destination nodeid']).sum(numeric_only=True)[['Expanded Tons']]
    flow_df['Origin-Destination nodeid'] = flow_df.index
    tons = flow_df['Expanded Tons'].to_dict()
    # load from json or compute if does not exist
    filepath_sp_dict = os.path.join(NX_DIR, rr + '_SP_dict_miles.json')
    if os.path.exists(filepath_sp_dict):
        miles = load_dict_from_json(filepath_sp_dict)
    else:
        miles = dict(nx.all_pairs_bellman_ford_path_length(G=G, weight='miles'))
        dict_to_json(miles, filepath_sp_dict)
    flow_df['Expanded Ton-Miles Routed'] = flow_df['Origin-Destination nodeid'].apply(lambda x:
                                                                                      tons[x] * miles[x[0]][x[1]])
    flow_df.drop(columns=['Origin-Destination nodeid', 'Expanded Tons'], inplace=True)
    # group by OD pair nodeid, summing all commodity groupings for the total ton-mile values (over all commodities)
    # keep only dataframe with ton-miles sum
    # flow_df = flow_df.groupby(by=['Origin-Destination nodeid']).sum(numeric_only=True)[['Expanded Ton-Miles Routed']]
    # sort OD pairs by ton-miles in descending order
    flow_df.sort_values(by='Expanded Ton-Miles Routed', ascending=False, inplace=True)

    # write to csv
    # flow_df_nodeids = flow_df
    # flow_df_nodeids['O-D nodeid'] = flow_df_nodeids.index
    # flow_df_nodeids['O'] = flow_df_nodeids['O-D nodeid'].apply(lambda x: x[0])
    # flow_df_nodeids['D'] = flow_df_nodeids['O-D nodeid'].apply(lambda x: x[1])
    # flow_df_nodeids.to_csv('/Users/adrianhz/Desktop/OD_flows_USA1.csv')

    # compute cumulative percentage of the ton-miles
    flow_df['Cumulative Percent Ton-Miles'] = flow_df.cumsum() / flow_df.sum()
    # select the subset of OD pairs that provides a cumulative percentage of ton-miles >= <perc_ods>
    m = flow_df[flow_df['Cumulative Percent Ton-Miles'] >= perc_ods]['Cumulative Percent Ton-Miles'].min()
    if m is np.NAN:
        m = 1
    ods = flow_df[flow_df['Cumulative Percent Ton-Miles'] <= m].index
    ods = [(o, d) for o, d in ods if o != d]
    # convert OD pair strings into node_id pair tuples
    # get O-D flows for all O-D pairs as a dict
    od_flows = flow_df['Expanded Ton-Miles Routed'].to_dict()
    if od_flows_truncate:
        od_flows = {od: od_flows[od] for od in ods}

    return ods, od_flows


def ods_by_perc_ton_mi_forecast(G: nx.DiGraph, perc_ods: float, od_flows_truncate=False):
    """

    Parameters
    ----------
    G
    perc_ods
    CCWS_filename
    time_window

    Returns
    -------

    """
    # return O-D pairs in CCWS tha provide ton flows >= <perc_ods> * total CCWS ton flows
    # od_flows is average daily ton-miles

    # load dict that maps SPLC codes to node_ids in G
    splc_node_dict = splc_to_node(G)
    # load grouped OD flow data
    # flow_df = RR_SPLC_comm_grouping(filename=CCWS_filename, time_window=time_window)
    flow_df = pd.read_csv(os.path.join(FLOW_DIR, 'faf5_high/WB2019_summary_class1_SPLC_forecast_2050.csv'),
                          header=0, index_col=['Railroad', 'Origin-Destination SPLC', 'Commodity Group Name'])

    # filter out specific railroad
    rr = G.graph['railroad']
    if 'KCS_ex' in rr:
        # for KCS example network
        flow_df = extract_rr(flow_df, 'KCS', forecast=True)
    else:
        flow_df = extract_rr(flow_df, rr, forecast=True)
    # only index needed is the OD pair
    flow_df.reset_index(level='Commodity Group Name', inplace=True)
    # filter out OD pairs that are not in the splc_node_dict keys
    splc_set = set(splc_node_dict.keys())
    remove_idxs = list({i for i in flow_df.index.unique() if i[1:7] not in splc_set or i[7:] not in splc_set})
    flow_df.drop(index=remove_idxs, inplace=True)

    # outcomment to have O-D flows counted bidirectionally
    # od_keys = set()
    # rev_od_strs = dict()
    # for od_str in flow_df.index:
    #     rev_od_str = od_str[0] + od_str[7:] + od_str[1:7]
    #     if rev_od_str not in od_keys:
    #         rev_od_strs[od_str] = rev_od_str
    #         od_keys.add(od_str)
    # flow_df.rename(index=rev_od_strs, inplace=True)

    # assign each SPLC OD to its respective nodeid in G
    flow_df.reset_index(level='Origin-Destination SPLC', inplace=True)
    flow_df['Origin-Destination nodeid'] = flow_df['Origin-Destination SPLC'].apply(lambda x:
                                                                                    (splc_node_dict[x[1:7]],
                                                                                     splc_node_dict[x[7:]]))

    years = ['', '2020 ', '2023 ', '2025 ', '2030 ', '2035 ', '2040 ', '2045 ', '2050 ']
    cols_to_keep = [y + 'Expanded Tons' for y in years]
    flow_df['Origin-Destination nodeid comb'] = flow_df['Origin-Destination nodeid'].apply(lambda x: x[0] + x[1])
    comb_od_nodeid_dict = {flow_df.loc[i, 'Origin-Destination nodeid comb']:
                               flow_df.loc[i, 'Origin-Destination nodeid'] for i in flow_df.index}
    # od_nodeid_comb_dict = {(o, d): o + d for o, d in flow_df.index}
    flow_df = flow_df.groupby(by=['Origin-Destination nodeid comb']).sum(numeric_only=True)[cols_to_keep]
    flow_df['Origin-Destination nodeid comb'] = flow_df.index
    # tons = flow_df['Expanded Tons'].to_dict()
    # load from json or compute if does not exist
    filepath_sp_dict = os.path.join(NX_DIR, rr + '_SP_dict_miles.json')
    if os.path.exists(filepath_sp_dict):
        miles = load_dict_from_json(filepath_sp_dict)
    else:
        miles = dict(nx.all_pairs_bellman_ford_path_length(G=G, weight='miles'))
        dict_to_json(miles, filepath_sp_dict)
    for y in years:
        flow_df[y + 'Expanded Ton-Miles Routed'] = flow_df['Origin-Destination nodeid comb'].apply(
            lambda x: flow_df.loc[x, y + 'Expanded Tons'] * miles[comb_od_nodeid_dict[x][0]][comb_od_nodeid_dict[x][1]])

    flow_df.drop(columns=['Origin-Destination nodeid comb'] + cols_to_keep, inplace=True)
    # group by OD pair nodeid, summing all commodity groupings for the total ton-mile values (over all commodities)
    # keep only dataframe with ton-miles sum
    # flow_df = flow_df.groupby(by=['Origin-Destination nodeid']).sum(numeric_only=True)[['Expanded Ton-Miles Routed']]
    # sort OD pairs by ton-miles in descending order
    flow_df.sort_values(by='Expanded Ton-Miles Routed', ascending=False, inplace=True)

    # write to csv
    # flow_df_nodeids = flow_df
    # flow_df_nodeids['O-D nodeid'] = flow_df_nodeids.index
    # flow_df_nodeids['O'] = flow_df_nodeids['O-D nodeid'].apply(lambda x: x[0])
    # flow_df_nodeids['D'] = flow_df_nodeids['O-D nodeid'].apply(lambda x: x[1])
    # flow_df_nodeids.to_csv('/Users/adrianhz/Desktop/OD_flows_USA1.csv')

    # compute cumulative percentage of the ton-miles
    flow_df['Cumulative Percent Ton-Miles'] = (flow_df['Expanded Ton-Miles Routed'].cumsum() /
                                               flow_df['Expanded Ton-Miles Routed'].sum())
    # select the subset of OD pairs that provides a cumulative percentage of ton-miles >= <perc_ods>
    m = flow_df[flow_df['Cumulative Percent Ton-Miles'] >= perc_ods]['Cumulative Percent Ton-Miles'].min()
    if m is np.NAN:
        m = 1
    ods = flow_df[flow_df['Cumulative Percent Ton-Miles'] <= m].index
    # convert OD pair strings into node_id pair tuples
    # get O-D flows for all O-D pairs as a dict
    if od_flows_truncate:
        flow_df = flow_df.loc[ods]

    ods = [comb_od_nodeid_dict[od] for od in ods]
    ods = [(o, d) for o, d in ods if o != d]

    flow_df.rename(index=comb_od_nodeid_dict, inplace=True)
    flow_df.fillna(0, inplace=True)
    year_mapper = {'2019': '', '2020': '2020 ', '2023': '2023 ', '2025': '2025 ', '2030': '2030 ', '2035': '2035 ',
                   '2040': '2040 ', '2045': '2045 ', '2050': '2050 '}
    od_flows = {y: [] for y in year_mapper.keys()}
    for y, y_name in year_mapper.items():
        od_flows[y] = flow_df[y_name + 'Expanded Ton-Miles Routed'].to_dict()

    od_flows_reorder = {od: {y: od_flows[y][od] if y in od_flows.keys() and od in od_flows[y].keys() else 0
                             for y in year_mapper.keys()} for od in ods}

    return ods, od_flows_reorder


def ods_ton_comm_forecast(G: nx.DiGraph, od_list: list):
    """

    Parameters
    ----------
    G
    perc_ods
    CCWS_filename
    time_window

    Returns
    -------

    """
    # return O-D pairs in CCWS tha provide ton flows >= <perc_ods> * total CCWS ton flows
    # od_flows is average daily ton-miles

    # load dict that maps SPLC codes to node_ids in G
    splc_node_dict = splc_to_node(G)
    # load grouped OD flow data
    # flow_df = RR_SPLC_comm_grouping(filename=CCWS_filename, time_window=time_window)
    flow_df = pd.read_csv(os.path.join(FLOW_DIR, 'faf5_high/WB2019_summary_class1_SPLC_forecast_2050.csv'),
                          header=0, index_col=['Railroad', 'Origin-Destination SPLC', 'Commodity Group Name'])

    # filter out specific railroad
    rr = G.graph['railroad']
    flow_df = extract_rr(flow_df, rr, forecast=True)
    # only index needed is the OD pair
    flow_df.reset_index(level='Commodity Group Name', inplace=True)
    # filter out OD pairs that are not in the splc_node_dict keys
    splc_set = set(splc_node_dict.keys())
    remove_idxs = list({i for i in flow_df.index.unique() if i[1:7] not in splc_set or i[7:] not in splc_set})
    flow_df.drop(index=remove_idxs, inplace=True)

    # outcomment to have O-D flows counted bidirectionally
    # od_keys = set()
    # rev_od_strs = dict()
    # for od_str in flow_df.index:
    #     rev_od_str = od_str[0] + od_str[7:] + od_str[1:7]
    #     if rev_od_str not in od_keys:
    #         rev_od_strs[od_str] = rev_od_str
    #         od_keys.add(od_str)
    # flow_df.rename(index=rev_od_strs, inplace=True)

    # assign each SPLC OD to its respective nodeid in G
    flow_df.reset_index(level='Origin-Destination SPLC', inplace=True)
    flow_df['Origin-Destination nodeid'] = flow_df['Origin-Destination SPLC'].apply(lambda x:
                                                                                    (splc_node_dict[x[1:7]],
                                                                                     splc_node_dict[x[7:]]))

    years = ['', '2020 ', '2023 ', '2025 ', '2030 ', '2035 ', '2040 ', '2045 ', '2050 ']
    cols_to_keep = [y + 'Expanded Tons' for y in years]
    flow_df['Origin-Destination nodeid comb'] = flow_df['Origin-Destination nodeid'].apply(lambda x: x[0] + x[1])
    comb_od_nodeid_dict = {flow_df.loc[i, 'Origin-Destination nodeid comb']:
                               flow_df.loc[i, 'Origin-Destination nodeid'] for i in flow_df.index}
    flow_df = flow_df.groupby(by=['Origin-Destination nodeid comb',
                                  'Commodity Group Name']).sum(numeric_only=True)[cols_to_keep]

    cols_to_drop = set(flow_df.columns).difference(set(['Origin-Destination nodeid comb',
                                                        'Commodity Group Name'] + cols_to_keep))
    flow_df.drop(columns=cols_to_drop, inplace=True)

    od_list_idx = {v: i for i, v in enumerate(od_list)}
    comm_groups = list(flow_df.index.levels[1])
    year_mapper = {'': '2019', '2020 ': '2020', '2023 ': '2023', '2025 ': '2025', '2030 ': '2030', '2035 ': '2035',
                   '2040 ': '2040', '2045 ': '2045', '2050 ': '2050'}
    od_comm_flows = {year_mapper[y]: {c: np.zeros((len(od_list),)) for c in comm_groups} for y in years}
    for od_comb, comm in flow_df.index:
        od = comb_od_nodeid_dict[od_comb]
        if od in od_list:
            for y in years:
                od_comm_flows[year_mapper[y]][comm][od_list_idx[od]] = flow_df.loc[(od_comb, comm), y + 'Expanded Tons']

    return od_comm_flows
