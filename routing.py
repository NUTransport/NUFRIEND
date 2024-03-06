from util import *
# MODULES
from helper import splc_to_node, mmddyyyy_to_datetime, node_to_edge_path, \
    load_comm_energy_ratios, load_conversion_factors, load_fuel_tech_eff_factor, \
    load_railroad_values, extract_rr, load_hybrid_energy_intensity_values, \
    load_flow_data_df_csv, load_flow_data_date_df_csv
from input_output import load_dict_from_json, dict_to_json

'''
ROUTING METHODS
'''


def route_flows(G: nx.DiGraph, fuel_type: str, flow_data_filename: str, H: nx.DiGraph = None, D: float = None,
                reroute=True, switch_tech=False, max_reroute_inc: float = None, year: int = None, freq: str = None):

    if not H:
        return route_baseline_flows(G=G, flow_data_filename=flow_data_filename)
    else:
        if 'hybrid' in fuel_type:
            return route_peak_avg_flows_hybrid(G=G, H=H, fuel_type=fuel_type,
                                               flow_data_filename=flow_data_filename,
                                               reroute=reroute, switch_tech=switch_tech,
                                               max_reroute_inc=max_reroute_inc)
        else:
            return route_peak_avg_flows(G=G, H=H, fuel_type=fuel_type, D=D, flow_data_filename=flow_data_filename,
                                        year=year, freq=freq,
                                        reroute=reroute, switch_tech=switch_tech, max_reroute_inc=max_reroute_inc)


def path_link_incidence_mat(G: nx.DiGraph, od_list: list, H: nx.DiGraph = None,
                            reroute=True, switch_tech=False, max_reroute_inc: float = None):
    edges_G = list(G.edges)
    edge_idx_dict = {edges_G[i]: i for i in range(len(edges_G))}
    # if H is None, return path-link IM for just baseline case for G
    if not H:
        # precompute shortest path for all listed origins in od_list (since Dijkstra's finds one-to-all shortest paths)
        baseline_sp = dict(nx.all_pairs_dijkstra_path(G, weight='km'))
        pli_mat = np.zeros((len(edges_G), len(od_list)))
        # for each OD pair, find the path b/w them and update the relevant links as being activated
        for i in range(len(od_list)):
            o, d = od_list[i]
            # get baseline path edges between o and d on G
            path_edges_0 = node_to_edge_path(baseline_sp[o][d])
            # get indices corresponding to edges on path b/w o and d on G
            path_edges_0_idxs = [edge_idx_dict[e] for e in path_edges_0]
            # update these link-od matrix entries to be 1 (these links exist on the path between o and d)
            pli_mat[path_edges_0_idxs, i] = 1

    # if H is passed through, return path-link IM for (0) baseline G, (1) alt.tech. H, (2) support diesel H
    #  based on the routing specifications provided in <reroute>, <switch_tech>, and <max_reroute_inc>
    else:
        # precompute shortest path for all pairs (since Dijkstra's finds one-to-all shortest paths)
        baseline_sp = dict(nx.all_pairs_dijkstra_path(G, weight='km'))
        alt_tech_sp = dict(nx.all_pairs_dijkstra_path(H, weight='km'))
        # initialize path-link incidence matrices: will have 3 (one for each kind of graph)
        pli_mat = np.zeros((3, len(edges_G), len(od_list)))
        # for each OD pair, find the path b/w them and update the relevant links as being activated
        for i in range(len(od_list)):
            o, d = od_list[i]
            # (0) Baseline network
            # get baseline path edges for trip b/w o and d on G
            path_edges_0 = node_to_edge_path(baseline_sp[o][d])
            # get indices corresponding to edges on path b/w o and d on G
            path_edges_0_idxs = [edge_idx_dict[e] for e in path_edges_0]
            # update these link-od matrix entries to be 1 (these links exist on the path between o and d)
            pli_mat[0, path_edges_0_idxs, i] = 1
            # (1) Alt. Tech. and (2) Support Diesel networks
            # if rerouting is allowed
            if reroute:
                # get new technology path edges for trip b/w o and d on H
                if o in alt_tech_sp.keys() and d in alt_tech_sp[o].keys():
                    # if o and d are connected in the alt. tech. network, return shortest path between them
                    path_edges_1 = node_to_edge_path(alt_tech_sp[o][d])
                else:
                    # otherwise, path between them is empty (does not exist)
                    path_edges_1 = []
                path_miles_1 = sum([G.edges[u, v]['miles'] for u, v in path_edges_1])
                # calculate baseline path distance
                path_miles_0 = sum([G.edges[u, v]['miles'] for u, v in path_edges_0])
                # if feasible path does not exist on H for rerouting
                if not path_edges_1 or path_miles_1 > path_miles_0 * (1 + max_reroute_inc):
                    # flow stays on shortest baseline route (on G)
                    path_edges_2 = path_edges_0
                    path_edges_1 = []
                    # if we must switch to alternative tech when feasible (i.e., an alt. tech. enabled link mid-route)
                    if switch_tech:
                        # assign baseline flow to the alt. tech. edges enabled with the technology
                        path_edges_1 = [(u, v) for u, v in path_edges_2 if H.has_edge(u, v)]
                        # assign baseline flow to remaining edges not covered by alt. tech. for support diesel
                        path_edges_2 = [(u, v) for u, v in path_edges_2 if (u, v) not in path_edges_1]
                # if feasible path exists on H, the support diesel network does not flow between this OD
                else:
                    # path fully served by alt. tech.; support diesel path not necessary
                    path_edges_2 = []
            # if rerouting not allowed but we must switch technology to alt. tech. mid-route when a link is enabled
            elif switch_tech:
                # assign baseline flow to the edges of H enabled with technology
                path_edges_1 = [(u, v) for u, v in path_edges_0 if H.has_edge(u, v)]
                # assign baseline flow to remaining edges not covered by alt. tech. for support diesel
                path_edges_2 = [(u, v) for u, v in path_edges_0 if (u, v) not in path_edges_1]
            # if rerouting not allowed and switching technology not required mid-route
            else:
                # assign baseline flow to alt. tech. if all the edges in the baseline path are alt. tech. enabled
                path_edges_1 = path_edges_0
                # assign baseline flow to remaining edges not covered by alt. tech. for support diesel
                path_edges_2 = []
                # if new path is not continuously served by the alt. tech.
                # if not all([1 if H.has_edge(u, v) else 0 for u, v in path_edges_1]):
                if not all([H.has_edge(u, v) for u, v in path_edges_1]):
                    path_edges_1 = []
                    path_edges_2 = path_edges_0

            # (1) get indices corresponding to edges on path b/w o and d on AT
            path_edges_1_idxs = [edge_idx_dict[e] for e in path_edges_1]
            # update these link-od matrix entries to be 1 (these links exist on the path between o and d)
            pli_mat[1, path_edges_1_idxs, i] = 1
            # (2) get indices corresponding to edges on path b/w o and d on SD
            path_edges_2_idxs = [edge_idx_dict[e] for e in path_edges_2]
            # update these link-od matrix entries to be 1 (these links exist on the path between o and d)
            pli_mat[2, path_edges_2_idxs, i] = 1

    return pli_mat


def route_baseline_flows(G: nx.DiGraph, flow_data_filename: str):
    G = G.copy().to_directed()

    # route average flows for given <time_window>

    # load flow data file; index is (<railroad>, <OD SPLC>, <commodity>)
    rr = G.graph['railroad']
    flow_df = load_flow_data_df_csv(flow_data_filename, rr=rr)

    # get set of SPLC codes and a dict to map to nodes in G
    splc_node_dict = splc_to_node(G)
    splc_set = set(splc_node_dict.keys())
    # list of all od pairs in dataset that exist in G in str format "'000000DDDDDD"
    od_str_list = list({od_str for od_str, _ in flow_df.index if od_str[1:7] in splc_set and od_str[7:] in splc_set})
    # list of all od pairs in dataset that exist in G
    od_list = [(splc_node_dict[od_str[1:7]], splc_node_dict[od_str[7:]]) for od_str in od_str_list]
    # get list of edges in G
    edges_G = list(G.edges)
    # get path-link incidence matrix for G for the selected list of OD pairs
    pli_mat = path_link_incidence_mat(G=G, od_list=od_list)
    # get list of all commodity groupings
    comm_list = list({c[1] for c in flow_df.index}) + ['TOTAL']
    comm_idx_dict = {comm_list[i]: i for i in range(len(comm_list))}
    # initialize set of vectors containing commodity group flows by OD pair
    f = np.zeros((len(comm_list), len(od_list), 1))
    # for each commodity group
    for c in comm_list[:-1]:
        c_idx = comm_idx_dict[c]
        flow_df_c = flow_df.reset_index(level='Origin-Destination SPLC').loc[c]
        if not isinstance(flow_df_c, pd.Series):
            flow_df_c = flow_df_c.reset_index()
            flow_df_c.index = flow_df_c['Origin-Destination SPLC']
            # assign the tons of flow for this commodity group to the respective index
            f[c_idx, :, 0] = [flow_df_c.loc[od_str, 'Tons'] if od_str in flow_df_c.index else 0
                              for od_str in od_str_list]
            # NEW
            # for od_str_idx, od_str in enumerate(od_str_list):
            #     f[c_idx, od_str_idx, 0] = flow_df_c.loc[od_str, 'Tons'] if od_str in flow_df_c.index else 0
        else:
            # assign the tons of flow for this commodity group to the respective index
            f[c_idx, :, 0] = [flow_df_c['Tons'] if od_str == flow_df_c['Origin-Destination SPLC'] else 0
                              for od_str in od_str_list]
        # increment the TOTAL comm group sum
        f[comm_idx_dict['TOTAL'], :, 0] += f[c_idx, :, 0]

    x = np.zeros((len(comm_list), len(edges_G), 1))
    for c in comm_list:
        c_idx = comm_idx_dict[c]
        x[c_idx, :, :] = np.dot(pli_mat, f[c_idx, :, :])

    # lookup dataframes for constants
    rr_v = load_railroad_values().loc[rr]  # railroad energy intensity statistics
    cf = load_conversion_factors()['Value']  # numerical constants for conversion across units
    # arrays ordered in same order as <comm_list> and stored as np arrays for vectorization
    rr_tc = rr_v['ton/car']  # tons/car
    comm_er = load_comm_energy_ratios()['Weighted ratio'][comm_list[:-1]].to_numpy()  # commodity energy ratios

    # tonmi2kwh = btu/ton-mi * kWh/btu * <energy_efficiency> * <energy_loss> = kWh/ton-mi- not adjusted by commodity
    # tonmi2kwh = (rr_ei['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kwh']) *
    #              (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))
    # # battery locomotive range given from D used to calculate battery locomotive energy capacity
    # # batt2kwh = kWh/ton-mi * ton/loc * km * mi/km * loc/batt = kWh/batt
    # batt2kwh = tonmi2kwh * cf['ton/loc'] * D * cf['mi/km']
    # tonmi2gal = btu/ton-mi * gal/btu * <energy_correction> = gal/ton-mi- not adjusted by commodity
    tonmi2gal = rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/gal']) * (1 / rr_v['Energy correction factor'])
    # tonmi2loc = kWh/ton-mi * batt/kWh * loc/batt = loc/tonmi- not adjusted by commodity
    # tonmi2loc = tonmi2kwh * (1 / batt2kwh)
    # car2loc = loc/train * train/car- not adjusted by commodity
    car2loc = rr_v['loc/train'] * (1 / rr_v['car/train'])

    for i in range(len(edges_G)):
        u, v = edges_G[i]
        mi = G.edges[u, v]['miles']
        # tons extracted from link flow assignment vector x
        G.edges[u, v]['baseline_avg_ton'] = dict(zip(comm_list, x[:, i, 0]))
        # loc = loc/car * <commodity_car/ton> * ton
        G.edges[u, v]['baseline_avg_loc'] = dict(zip(comm_list[:-1], np.ceil(car2loc * (1 / rr_tc) * x[:-1, i, 0])))
        # gal = gal/ton-mi * <commodity_factor> * ton-mi
        G.edges[u, v]['baseline_avg_gal'] = dict(zip(comm_list[:-1], tonmi2gal * comm_er * x[:-1, i, 0] * mi))
        # sum 'TOTAL' values for locomotive and energy flow
        G.edges[u, v]['baseline_avg_loc']['TOTAL'] = sum(G.edges[u, v]['baseline_avg_loc'].values())
        G.edges[u, v]['baseline_avg_gal']['TOTAL'] = sum(G.edges[u, v]['baseline_avg_gal'].values())

    baseline_total_ton_mi = dict(zip(
        comm_list,
        [sum([G.edges[u, v]['baseline_avg_ton'][c] * G.edges[u, v]['miles'] for u, v in G.edges]) for c in comm_list]))
    G.graph['operations'] = dict(
        baseline_avg_distance_mi=dict(zip(
            comm_list,
            [baseline_total_ton_mi[c] / f[comm_idx_dict[c], :, 0].sum() for c in comm_list])),
        baseline_total_tonmi=baseline_total_ton_mi,
        baseline_total_annual_tonmi=dict(zip(
            comm_list,
            [365 * baseline_total_ton_mi[c] for c in comm_list])),
        baseline_commodity_gal=dict(zip(
            comm_list,
            [sum([G.edges[u, v]['baseline_avg_gal'][c] for u, v in G.edges]) for c in comm_list]))
    )

    return G


def route_peak_avg_flows(G: nx.DiGraph, H: nx.DiGraph, fuel_type: str, D: float, flow_data_filename: str,
                         year: int = None, freq: str = 'M',
                         reroute=True, switch_tech=False, max_reroute_inc: float = None):
    G = G.copy().to_directed()

    rr = G.graph['railroad']
    t0 = time.time()
    # load grouped OD flow data
    flow_df = load_flow_data_date_df_csv(filename=flow_data_filename, rr=rr)
    time_window_list = list(set(flow_df.index.get_level_values(2)))
    time_window_list_tuples = [(tw[1:9], tw[10:]) for tw in time_window_list]

    tw_len_dict = dict()
    for s, e in time_window_list_tuples:
        tw_len_dict['S' + s + 'E' + e] = len(pd.date_range(start=mmddyyyy_to_datetime(s), end=mmddyyyy_to_datetime(e)))
    total_tw_len = sum(tw_len_dict.values())  # total length of all time windows

    print('\t DATA LOADING:: %s seconds ---' % round(time.time() - t0, 3))

    # get set of SPLC codes and a dict to map to nodes in G
    splc_node_dict = splc_to_node(G)
    splc_set = set(splc_node_dict.keys())
    # list of all od pairs in dataset that exist in G in str format "'000000DDDDDD"
    od_str_list = list({od_str for od_str, _, _ in flow_df.index if od_str[1:7] in splc_set and od_str[7:] in splc_set})
    # list of all od pairs (nodeids) in dataset that exist in G
    od_list = [(splc_node_dict[od_str[1:7]], splc_node_dict[od_str[7:]]) for od_str in od_str_list]
    # get list of edges in G
    edges_G = list(G.edges)

    # TO REMOVE
    # df_ods = pd.read_csv('/Users/adrianhz/Desktop/ods.csv', header=0)
    # ods_set = set(zip(df_ods['0'], df_ods['1']))
    # for splc_str in flow_df.index.get_level_values(0):
    #     if splc_str[1:7] in splc_set and splc_str[7:] in splc_set:
    #         if (splc_node_dict[splc_str[1:7]], splc_node_dict[splc_str[7:]]) not in ods_set:
    #             flow_df.loc[splc_str, 'Expanded Tons'] = 0
    # TO REMOVE

    t0 = time.time()
    # get path-link incidence matrix for G and H (both alt. tech. and support diesel) for the selected list of OD pairs
    pli_mat = path_link_incidence_mat(G=G, od_list=od_list, H=H,
                                      reroute=reroute, switch_tech=switch_tech, max_reroute_inc=max_reroute_inc)
    print('\t OD ROUTING:: %s seconds ---' % round(time.time() - t0, 3))

    # get list of all commodity groupings
    comm_list_orig = list({c[1] for c in flow_df.index})
    comm_list = comm_list_orig + ['TOTAL']
    comm_idx_dict = {comm_list[i]: i for i in range(len(comm_list))}
    # initialize set of vectors containing commodity group flows for:
    #   - each commodity grouping
    #   - each time window + the average over all time windows
    #   - each OD pair

    t0 = time.time()
    f = np.zeros((len(comm_list), len(time_window_list_tuples) + 1, len(od_list), 1))
    # for each time window
    for tw_idx in range(len(time_window_list_tuples)):
        s, e = time_window_list_tuples[tw_idx]
        # convert tw to the format in flow_df.index
        tw_str = 'S' + s + 'E' + e
        # average weight for this time window; for computing average ton flows over all time periods
        aw = tw_len_dict[tw_str] / total_tw_len
        # for each commodity group
        for c_idx in range(len(comm_list[:-1])):
            c = comm_list[c_idx]
            flow_df_c_tw = flow_df.reset_index(level='Origin-Destination SPLC').sort_index()
            if (c, tw_str) not in flow_df_c_tw.index:
                continue
            flow_df_c_tw = flow_df_c_tw.loc[(c, tw_str)].reset_index()
            flow_df_c_tw.index = flow_df_c_tw['Origin-Destination SPLC']
            # assign the tons of flow for this commodity group to the respective index
            f[c_idx, tw_idx, :, 0] = [flow_df_c_tw.loc[od_str, 'Tons'] if od_str in flow_df_c_tw.index else 0
                                      for od_str in od_str_list]
            # increment the TOTAL comm group sum
            f[comm_idx_dict['TOTAL'], tw_idx, :, 0] += f[c_idx, tw_idx, :, 0]
        # update average comm group sum for this time window
        f[:, -1, :, 0] += np.multiply(aw, f[:, tw_idx, :, 0])
    print('\t OD FLOW EXTRACTION:: %s seconds ---' % round(time.time() - t0, 3))

    t0 = time.time()
    x = np.zeros((3, len(comm_list), len(time_window_list_tuples) + 1, len(edges_G), 1))
    for tw_idx in range(len(time_window_list_tuples) + 1):
        for c_idx in range(len(comm_list)):
            # (0) Baseline network flows
            x[0, c_idx, tw_idx, :, :] = np.dot(pli_mat[0, :, :], f[c_idx, tw_idx, :, :])
            # (1) Alt. Tech. network flows
            x[1, c_idx, tw_idx, :, :] = np.dot(pli_mat[1, :, :], f[c_idx, tw_idx, :, :])
            # (2) Support Diesel network flows
            x[2, c_idx, tw_idx, :, :] = np.dot(pli_mat[2, :, :], f[c_idx, tw_idx, :, :])
    print('\t LINK FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

    # lookup dataframes for constants
    ft_ef = load_fuel_tech_eff_factor().loc[fuel_type]  # fuel tech efficiency factors
    cf = load_conversion_factors()['Value']  # numerical constants for conversion across units
    # load rr specific values
    rr_v = load_railroad_values().loc[rr]
    # arrays ordered in same order as <comm_list> and stored as np arrays for vectorization
    rr_tc = rr_v['ton/car']  # tons/car
    comm_er = load_comm_energy_ratios()['Weighted ratio'][comm_list[:-1]].to_numpy()  # commodity energy ratios

    # tonmi2kwh = btu/ton-mi * kWh/btu * <energy_correction> * <energy_efficiency> * <energy_loss> =
    # kWh/ton-mi- not adjusted by commodity
    if fuel_type == 'battery':
        tonmi2energy = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kwh']) *
                        (1 / rr_v['Energy correction factor']) * (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))
    elif fuel_type == 'hydrogen':
        tonmi2energy = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kgh2']) *
                        (1 / rr_v['Energy correction factor']) * (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))

    # battery locomotive range given from D used to calculate battery locomotive energy capacity
    # loc2kwh = kWh/ton-mi * ton/loc * km * mi/km * loc/batt = kWh/loc
    loc2energy = tonmi2energy * rr_v['ton/loc'] * D * cf['mi/km']
    # tonmi2gal = btu/ton-mi * gal/btu * <energy_correction> = gal/ton-mi- not adjusted by commodity
    tonmi2gal = rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/gal']) * (1 / rr_v['Energy correction factor'])
    # tonmi2loc = kWh/ton-mi * loc/kWh = loc/tonmi - not adjusted by commodity
    tonmi2loc = tonmi2energy * (1 / loc2energy)
    # car2loc = loc/train * train/car- not adjusted by commodity
    car2loc = rr_v['loc/train'] * (1 / rr_v['car/train'])

    t0 = time.time()
    for i in range(len(edges_G)):
        u, v = edges_G[i]
        mi = G.edges[u, v]['miles']
        # max time windows
        tw_m0 = np.argmax(x[0, comm_idx_dict['TOTAL'], :-1, i, 0])
        tw_m1 = np.argmax(x[1, comm_idx_dict['TOTAL'], :-1, i, 0])
        tw_m2 = np.argmax(x[2, comm_idx_dict['TOTAL'], :-1, i, 0])
        # (0) Baseline
        # tons extracted from link flow assignment vector x
        G.edges[u, v]['baseline_avg_ton'] = dict(zip(comm_list, x[0, :, -1, i, 0]))
        G.edges[u, v]['baseline_peak_ton'] = dict(zip(comm_list, x[0, :, tw_m0, i, 0]))
        # loc = loc/car * <commodity_car/ton> * ton
        G.edges[u, v]['baseline_avg_loc'] = dict(zip(comm_list[:-1],
                                                     np.ceil(car2loc * (1 / rr_tc) * x[0, :-1, -1, i, 0])))
        G.edges[u, v]['baseline_peak_loc'] = dict(zip(comm_list[:-1],
                                                      np.ceil(car2loc * (1 / rr_tc) * x[0, :-1, tw_m0, i, 0])))
        # gal = gal/ton-mi * <commodity_factor> * ton-mi
        G.edges[u, v]['baseline_avg_gal'] = dict(zip(comm_list[:-1],
                                                     tonmi2gal * comm_er * x[0, :-1, -1, i, 0] * mi))
        G.edges[u, v]['baseline_peak_gal'] = dict(zip(comm_list[:-1],
                                                      tonmi2gal * comm_er * x[0, :-1, tw_m0, i, 0] * mi))
        # sum 'TOTAL' values for locomotive and energy flow
        G.edges[u, v]['baseline_avg_loc']['TOTAL'] = sum(G.edges[u, v]['baseline_avg_loc'].values())
        G.edges[u, v]['baseline_peak_loc']['TOTAL'] = sum(G.edges[u, v]['baseline_peak_loc'].values())
        G.edges[u, v]['baseline_avg_gal']['TOTAL'] = sum(G.edges[u, v]['baseline_avg_gal'].values())
        G.edges[u, v]['baseline_peak_gal']['TOTAL'] = sum(G.edges[u, v]['baseline_peak_gal'].values())
        # (1) Alt. Tech.
        # tons extracted from link flow assignment vector x
        G.edges[u, v][fuel_type + '_avg_ton'] = dict(zip(comm_list, x[1, :, -1, i, 0]))
        G.edges[u, v][fuel_type + '_peak_ton'] = dict(zip(comm_list, x[1, :, tw_m1, i, 0]))
        # loc = loc/car * <commodity_car/ton> * ton
        G.edges[u, v][fuel_type + '_avg_loc'] = dict(zip(comm_list[:-1],
                                                         np.ceil(tonmi2loc * comm_er * x[1, :-1, -1, i, 0] * mi)))
        G.edges[u, v][fuel_type + '_peak_loc'] = dict(zip(comm_list[:-1],
                                                          np.ceil(tonmi2loc * comm_er * x[1, :-1, tw_m1, i, 0] * mi)))
        if fuel_type == 'battery':
            # kwh = kwh/ton-mi * <commodity_factor> * ton-mi
            G.edges[u, v][fuel_type + '_avg_kwh'] = dict(zip(comm_list[:-1],
                                                             (tonmi2energy * comm_er * x[1, :-1, -1, i, 0] * mi)))
            G.edges[u, v][fuel_type + '_peak_kwh'] = dict(zip(comm_list[:-1],
                                                              (tonmi2energy * comm_er * x[1, :-1, tw_m1, i, 0] * mi)))
        elif fuel_type == 'hydrogen':
            # kwh = kwh/ton-mi * <commodity_factor> * ton-mi
            G.edges[u, v][fuel_type + '_avg_kgh2'] = dict(zip(comm_list[:-1],
                                                              (tonmi2energy * comm_er * x[1, :-1, -1, i, 0] * mi)))
            G.edges[u, v][fuel_type + '_peak_kgh2'] = dict(zip(comm_list[:-1],
                                                               (tonmi2energy * comm_er * x[1, :-1, tw_m1, i, 0] * mi)))

        # sum 'TOTAL' values for locomotive and energy flow
        G.edges[u, v][fuel_type + '_avg_loc']['TOTAL'] = sum(G.edges[u, v][fuel_type + '_avg_loc'].values())
        G.edges[u, v][fuel_type + '_peak_loc']['TOTAL'] = sum(G.edges[u, v][fuel_type + '_peak_loc'].values())
        if fuel_type == 'battery':
            G.edges[u, v][fuel_type + '_avg_kwh']['TOTAL'] = sum(G.edges[u, v][fuel_type + '_avg_kwh'].values())
            G.edges[u, v][fuel_type + '_peak_kwh']['TOTAL'] = sum(G.edges[u, v][fuel_type + '_peak_kwh'].values())
        elif fuel_type == 'hydrogen':
            G.edges[u, v][fuel_type + '_avg_kgh2']['TOTAL'] = sum(G.edges[u, v][fuel_type + '_avg_kgh2'].values())
            G.edges[u, v][fuel_type + '_peak_kgh2']['TOTAL'] = sum(G.edges[u, v][fuel_type + '_peak_kgh2'].values())

        # (2) Support Diesel
        # tons extracted from link flow assignment vector x
        G.edges[u, v]['support_diesel_avg_ton'] = dict(zip(comm_list, x[2, :, -1, i, 0]))
        G.edges[u, v]['support_diesel_peak_ton'] = dict(zip(comm_list, x[2, :, tw_m2, i, 0]))
        # loc = loc/car * <commodity_car/ton> * ton
        G.edges[u, v]['support_diesel_avg_loc'] = dict(zip(comm_list[:-1],
                                                           np.ceil(car2loc * (1 / rr_tc) * x[2, :-1, -1, i, 0])))
        G.edges[u, v]['support_diesel_peak_loc'] = dict(zip(comm_list[:-1],
                                                            np.ceil(car2loc * (1 / rr_tc) * x[2, :-1, tw_m2, i, 0])))
        # gal = gal/ton-mi * <commodity_factor> * ton-mi
        G.edges[u, v]['support_diesel_avg_gal'] = dict(zip(comm_list[:-1],
                                                           tonmi2gal * comm_er * x[2, :-1, -1, i, 0] * mi))
        G.edges[u, v]['support_diesel_peak_gal'] = dict(zip(comm_list[:-1],
                                                            (tonmi2gal * comm_er * x[2, :-1, tw_m2, i, 0] * mi)))
        # sum 'TOTAL' values for locomotive and energy flow
        G.edges[u, v]['support_diesel_avg_loc']['TOTAL'] = sum(G.edges[u, v]['support_diesel_avg_loc'].values())
        G.edges[u, v]['support_diesel_peak_loc']['TOTAL'] = sum(G.edges[u, v]['support_diesel_peak_loc'].values())
        G.edges[u, v]['support_diesel_avg_gal']['TOTAL'] = sum(G.edges[u, v]['support_diesel_avg_gal'].values())
        G.edges[u, v]['support_diesel_peak_gal']['TOTAL'] = sum(G.edges[u, v]['support_diesel_peak_gal'].values())

        # compute and store service shares by fuel technology
        battery_tot_flow = G.edges[u, v][fuel_type + '_avg_ton']['TOTAL']
        support_tot_flow = G.edges[u, v]['support_diesel_avg_ton']['TOTAL']
        if H.has_edge(u, v) and battery_tot_flow + support_tot_flow > 0:
            G.edges[u, v][fuel_type + '_perc_ton'] = 100 * battery_tot_flow / (battery_tot_flow + support_tot_flow)
            G.edges[u, v]['support_diesel_perc_ton'] = 100 * support_tot_flow / (battery_tot_flow + support_tot_flow)
        elif support_tot_flow == 0:
            G.edges[u, v][fuel_type + '_perc_ton'] = 0
            G.edges[u, v]['support_diesel_perc_ton'] = 0
        else:
            G.edges[u, v][fuel_type + '_perc_ton'] = 0
            G.edges[u, v]['support_diesel_perc_ton'] = 100
    print('\t LINK FLOW EXTRACTION:: %s seconds ---' % round(time.time() - t0, 3))

    # calculate the percentage distance increase for those goods actually rerouted
    baseline_total_tonmi_rerouted = 0  # baseline (original) ton-miles for those ton-miles that were rerouted
    alt_tech_total_tonmi_rerouted = 0  # new ton-miles for those ton-miles that were actually rerouted to alt. tech.
    for od in range(len(f[-1, -1, :, 0])):
        # if the path from the baseline and alt. tech. networks differ for a given OD pair, there was rerouting
        if list(pli_mat[0, :, od]) != list(pli_mat[1, :, od]) and sum(pli_mat[1, :, od]) != 0:
            # calculate the baseline (original) and alt. tech. (new) ton-miles associated with this rerouting
            tons = f[-1, -1, od, 0]
            baseline_total_tonmi_rerouted += tons * sum(G.edges[edges_G[i][0], edges_G[i][1]]['miles']
                                                        for i in np.where(pli_mat[0, :, od] == 1)[0])
            alt_tech_total_tonmi_rerouted += tons * sum(G.edges[edges_G[i][0], edges_G[i][1]]['miles']
                                                        for i in np.where(pli_mat[1, :, od] == 1)[0])

    baseline_total_tonmi = dict(zip(
        comm_list,
        [sum([G.edges[u, v]['baseline_avg_ton'][c] * G.edges[u, v]['miles'] for u, v in G.edges]) for c in comm_list]))
    alt_tech_total_tonmi = dict(zip(
        comm_list,
        [sum([G.edges[u, v][fuel_type + '_avg_ton'][c] * G.edges[u, v]['miles'] for u, v in G.edges])
         for c in comm_list]))
    support_diesel_total_tonmi = dict(zip(
        comm_list,
        [sum([G.edges[u, v]['support_diesel_avg_ton'][c] * G.edges[u, v]['miles'] for u, v in G.edges]) for c in
         comm_list]))
    scenario_total_tonmi = dict(zip(
        comm_list,
        [alt_tech_total_tonmi[c] + support_diesel_total_tonmi[c] for c in comm_list]))
    perc_tonmi_inc = dict(zip(
        comm_list,
        [100 * (scenario_total_tonmi[c] - baseline_total_tonmi[c]) / baseline_total_tonmi[c] for c in comm_list]))
    total_tons = dict(zip(comm_list, [f[comm_idx_dict[c], -1, :, 0].sum() for c in comm_list]))
    G.graph['operations'] = dict(
        baseline_avg_distance_mi=dict(zip(comm_list, [baseline_total_tonmi[c] / total_tons[c] for c in comm_list])),
        baseline_total_tonmi=baseline_total_tonmi,
        baseline_total_annual_tonmi=dict(zip(comm_list, [365 * baseline_total_tonmi[c] for c in comm_list])),
        baseline_total_gal=dict(zip(
            comm_list,
            [sum([G.edges[u, v]['baseline_avg_gal'][c] for u, v in G.edges]) for c in comm_list])),
        scenario_avg_distance_mi=dict(zip(comm_list, [scenario_total_tonmi[c] / total_tons[c] for c in comm_list])),
        alt_tech_total_tonmi=alt_tech_total_tonmi,
        alt_tech_total_annual_tonmi=dict(zip(comm_list, [365 * alt_tech_total_tonmi[c] for c in comm_list])),
        alt_tech_total_locmi=dict(zip(
            comm_list,
            [sum([G.edges[u, v][fuel_type + '_avg_loc'][c] * G.edges[u, v]['miles'] for u, v in G.edges])
             for c in comm_list])),
        support_diesel_total_tonmi=support_diesel_total_tonmi,
        support_diesel_total_annual_tonmi=dict(zip(comm_list,
                                                   [365 * support_diesel_total_tonmi[c] for c in comm_list])),
        scenario_total_tonmi=scenario_total_tonmi,
        scenario_total_annual_tonmi=dict(zip(comm_list, [365 * scenario_total_tonmi[c] for c in comm_list])),
        support_diesel_total_locmi=dict(zip(
            comm_list,
            [sum([G.edges[u, v]['support_diesel_avg_loc'][c] * G.edges[u, v]['miles'] for u, v in G.edges])
             for c in comm_list])),
        support_diesel_total_gal=dict(zip(
            comm_list,
            [sum([G.edges[u, v]['support_diesel_avg_gal'][c] for u, v in G.edges]) for c in comm_list])),
        perc_tonmi_inc=perc_tonmi_inc,
        perc_mi_inc=perc_tonmi_inc,
        perc_tonmi_inc_conditional_reroute=dict(zip(
            comm_list,
            [100 * (alt_tech_total_tonmi_rerouted - baseline_total_tonmi_rerouted) / baseline_total_tonmi_rerouted
             if baseline_total_tonmi_rerouted != 0 else 0 for c in comm_list])),
        deployment_perc=dict(zip(comm_list, [alt_tech_total_tonmi[c] / scenario_total_tonmi[c] for c in comm_list]))
    )

    if fuel_type == 'battery':
        G.graph['operations'].update(dict(
            alt_tech_total_kwh=dict(zip(
                comm_list,
                [sum([G.edges[u, v][fuel_type + '_avg_kwh'][c] for u, v in G.edges]) for c in comm_list])),
            eff_kwh_p_loc=loc2energy,
            listed_kwh_p_loc=loc2energy * (1 / ft_ef['Effective capacity'])
        ))
    elif fuel_type == 'hydrogen':
        G.graph['operations'].update(dict(
            alt_tech_total_kgh2=dict(zip(
                comm_list,
                [sum([G.edges[u, v][fuel_type + '_avg_kgh2'][c] for u, v in G.edges]) for c in comm_list])),
            eff_kgh2_p_loc=loc2energy,
            listed_kgh2_p_loc=loc2energy * (1 / ft_ef['Effective capacity'])
        ))

    return G, H


def route_peak_avg_flows_hybrid(G: nx.DiGraph, H: nx.DiGraph, fuel_type: str, flow_data_filename: str = None,
                                reroute=True, switch_tech=False, max_reroute_inc: float = None):
    G = G.copy().to_directed()

    rr = G.graph['railroad']
    # load grouped OD flow data
    flow_df = load_flow_data_date_df_csv(filename=flow_data_filename, rr=rr)
    time_window_list = list(set(flow_df.index.get_level_values(2)))
    time_window_list_tuples = [(tw[1:9], tw[10:]) for tw in time_window_list]

    tw_len_dict = dict()
    for s, e in time_window_list_tuples:
        tw_len_dict['S' + s + 'E' + e] = len(pd.date_range(start=mmddyyyy_to_datetime(s), end=mmddyyyy_to_datetime(e)))
    total_tw_len = sum(tw_len_dict.values())  # total length of all time windows

    # print('\t DATA LOADING:: %s seconds ---' % round(time.time() - t0, 3))

    # get set of SPLC codes and a dict to map to nodes in G
    splc_node_dict = splc_to_node(G)
    splc_set = set(splc_node_dict.keys())
    # list of all od pairs in dataset that exist in G in str format "'000000DDDDDD"
    od_str_list = list({od_str for od_str, _, _ in flow_df.index if od_str[1:7] in splc_set and od_str[7:] in splc_set})
    # list of all od pairs (nodeids) in dataset that exist in G
    od_list = [(splc_node_dict[od_str[1:7]], splc_node_dict[od_str[7:]]) for od_str in od_str_list]
    # get list of edges in G
    edges_G = list(G.edges)

    t0 = time.time()
    # get path-link incidence matrix for G and H (both alt. tech. and support diesel) for the selected list of OD pairs
    pli_mat = path_link_incidence_mat(G=G, od_list=od_list, H=H,
                                      reroute=reroute, switch_tech=switch_tech, max_reroute_inc=max_reroute_inc)
    print('\t OD ROUTING:: %s seconds ---' % round(time.time() - t0, 3))

    # get list of all commodity groupings
    comm_list_orig = list({c[1] for c in flow_df.index})
    comm_list = comm_list_orig + ['TOTAL']
    comm_idx_dict = {comm_list[i]: i for i in range(len(comm_list))}
    # initialize set of vectors containing commodity group flows for:
    #   - each commodity grouping
    #   - each time window + the average over all time windows
    #   - each OD pair

    t0 = time.time()
    f = np.zeros((len(comm_list), len(time_window_list_tuples) + 1, len(od_list), 1))
    # for each time window
    for tw_idx in range(len(time_window_list_tuples)):
        s, e = time_window_list_tuples[tw_idx]
        # convert tw to the format in flow_df.index
        tw_str = 'S' + s + 'E' + e
        # average weight for this time window; for computing average ton flows over all time periods
        aw = tw_len_dict[tw_str] / total_tw_len
        # for each commodity group
        for c_idx in range(len(comm_list[:-1])):
            c = comm_list[c_idx]
            flow_df_c_tw = flow_df.reset_index(level='Origin-Destination SPLC').sort_index()
            if (c, tw_str) not in flow_df_c_tw.index:
                continue
            flow_df_c_tw = flow_df_c_tw.loc[(c, tw_str)].reset_index()
            flow_df_c_tw.index = flow_df_c_tw['Origin-Destination SPLC']
            # assign the tons of flow for this commodity group to the respective index
            f[c_idx, tw_idx, :, 0] = [flow_df_c_tw.loc[od_str, 'Tons'] if od_str in flow_df_c_tw.index else 0
                                      for od_str in od_str_list]
            # increment the TOTAL comm group sum
            f[comm_idx_dict['TOTAL'], tw_idx, :, 0] += f[c_idx, tw_idx, :, 0]
        # update average comm group sum for this time window
        f[:, -1, :, 0] += np.multiply(aw, f[:, tw_idx, :, 0])
    print('\t OD FLOW EXTRACTION:: %s seconds ---' % round(time.time() - t0, 3))

    t0 = time.time()
    x = np.zeros((3, len(comm_list), len(time_window_list_tuples) + 1, len(edges_G), 1))
    for tw_idx in range(len(time_window_list_tuples) + 1):
        for c_idx in range(len(comm_list)):
            # (0) Baseline network flows
            x[0, c_idx, tw_idx, :, :] = np.dot(pli_mat[0, :, :], f[c_idx, tw_idx, :, :])
            # (1) Alt. Tech. network flows
            x[1, c_idx, tw_idx, :, :] = np.dot(pli_mat[1, :, :], f[c_idx, tw_idx, :, :])
            # (2) Support Diesel network flows
            x[2, c_idx, tw_idx, :, :] = np.dot(pli_mat[2, :, :], f[c_idx, tw_idx, :, :])
    print('\t LINK FLOW ASSIGNMENT:: %s seconds ---' % round(time.time() - t0, 3))

    hybrid_ratio = fuel_type[6:]
    # lookup dataframes for constants
    ft_ef = load_fuel_tech_eff_factor().loc['battery']  # fuel tech efficiency factors
    cf = load_conversion_factors()['Value']  # numerical constants for conversion across units
    # load rr specific values
    rr_v = load_railroad_values().loc[rr]
    # arrays ordered in same order as <comm_list> and stored as np arrays for vectorization
    rr_tc = rr_v['ton/car']  # tons/car
    comm_er = load_comm_energy_ratios()['Weighted ratio'][comm_list[:-1]].to_numpy()  # commodity energy ratios
    hybrid_ei = load_hybrid_energy_intensity_values()  # hybrid energy intensities
    h_diesel_ei = dict(zip(hybrid_ei.index, hybrid_ei['diesel ' + hybrid_ratio + ' (btu/ton-mi)']))
    h_battery_ei = dict(zip(hybrid_ei.index, hybrid_ei['battery ' + hybrid_ratio + ' (btu/ton-mi)']))
    # for pure diesel operations
    pure_diesel_ei = dict(zip(hybrid_ei.index, hybrid_ei['diesel ' + '1:0' + ' (btu/ton-mi)']))

    # used to adjust energy consumption estimates for hybrid technology based on railroad-specific characteristics
    # units: unitless
    hybrid_energy_factor = (1 / rr_v['Energy correction factor']) * (1 / rr_v['hybrid energy factor'])
    # units: [kwh/btu]
    hybrid_energy_factor_btu2kwh = (1 / cf['btu/kwh']) * hybrid_energy_factor
    # if 'hybrid' not in fuel_type:
    #     # tonmi2kwh = btu/ton-mi * kWh/btu * <energy_correction> * <energy_efficiency> * <energy_loss> =
    #     # kWh/ton-mi- not adjusted by commodity
    #     if fuel_type == 'battery':
    #         tonmi2energy = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kwh']) *
    #                         (1 / rr_v['Energy correction factor']) * (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))
    #     elif fuel_type == 'hydrogen':
    #         tonmi2energy = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kgh2']) *
    #                         (1 / rr_v['Energy correction factor']) * (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))

    # battery locomotive range given from D used to calculate battery locomotive energy capacity
    # loc2energy_factor = ton/loc * km * mi/km = ton-mi/loc
    # btu2loc = kwh/btu * batt/kwh * 1 loc/batt = loc/btu
    eff_kwh_p_batt = G.graph['scenario']['eff_kwh_p_batt']
    btu2loc = hybrid_energy_factor_btu2kwh / eff_kwh_p_batt
    # loc2energy_factor = hybrid_energy_factor * rr_v['ton/loc'] * D * cf['mi/km']
    # btu2gal = gal/btu * <energy_correction> = gal/btu- not adjusted by commodity or region
    btu2gal = (1 / cf['btu/gal']) * hybrid_energy_factor
    # tonmi2loc = kWh/ton-mi * loc/kWh = loc/tonmi - not adjusted by commodity
    # tonmi2loc = tonmi2energy * (1 / loc2energy_factor)
    # car2loc = loc/train * train/car- not adjusted by commodity
    car2loc = rr_v['loc/train'] * (1 / rr_v['car/train'])

    fuel_type_battery = fuel_type + '_battery'
    fuel_type_diesel = fuel_type + '_diesel'

    t0 = time.time()
    for i in range(len(edges_G)):
        u, v = edges_G[i]
        mi = G.edges[u, v]['miles']
        # extract the edge-specific hybrid energy intensities and apply them
        algn = G.edges[u, v]['region_alignment']
        # provide region-specific btu/ton-mi
        h_diesel_tonmi2btu = h_diesel_ei[algn]
        h_battery_tonmi2btu = h_battery_ei[algn]
        pure_diesel_tonmi2btu = pure_diesel_ei[algn]
        # compute tonmi2loc = btu/ton-mi * loc/btu = loc/ton-mi
        tonmi2loc_batt = h_battery_tonmi2btu * btu2loc

        # max time windows
        tw_m0 = np.argmax(x[0, comm_idx_dict['TOTAL'], :-1, i, 0])
        tw_m1 = np.argmax(x[1, comm_idx_dict['TOTAL'], :-1, i, 0])
        tw_m2 = np.argmax(x[2, comm_idx_dict['TOTAL'], :-1, i, 0])
        # (0) Baseline
        # tons extracted from link flow assignment vector x
        G.edges[u, v]['baseline_avg_ton'] = dict(zip(comm_list, x[0, :, -1, i, 0]))
        G.edges[u, v]['baseline_peak_ton'] = dict(zip(comm_list, x[0, :, tw_m0, i, 0]))
        # loc = loc/car * <commodity_car/ton> * ton
        G.edges[u, v]['baseline_avg_loc'] = dict(zip(comm_list[:-1],
                                                     np.ceil(car2loc * (1 / rr_tc) * x[0, :-1, -1, i, 0])))
        G.edges[u, v]['baseline_peak_loc'] = dict(zip(comm_list[:-1],
                                                      np.ceil(car2loc * (1 / rr_tc) * x[0, :-1, tw_m0, i, 0])))
        # gal = gal/btu * <hybrid_energy_factor> * <commodity_factor> * ton * mi * btu/ton-mi
        G.edges[u, v]['baseline_avg_gal'] = dict(zip(comm_list[:-1],
                                                     btu2gal * comm_er * x[0, :-1, -1, i, 0] * mi *
                                                     pure_diesel_tonmi2btu))
        G.edges[u, v]['baseline_peak_gal'] = dict(zip(comm_list[:-1],
                                                      btu2gal * comm_er * x[0, :-1, tw_m0, i, 0] * mi *
                                                      pure_diesel_tonmi2btu))
        # sum 'TOTAL' values for locomotive and energy flow
        G.edges[u, v]['baseline_avg_loc']['TOTAL'] = sum(G.edges[u, v]['baseline_avg_loc'].values())
        G.edges[u, v]['baseline_peak_loc']['TOTAL'] = sum(G.edges[u, v]['baseline_peak_loc'].values())
        G.edges[u, v]['baseline_avg_gal']['TOTAL'] = sum(G.edges[u, v]['baseline_avg_gal'].values())
        G.edges[u, v]['baseline_peak_gal']['TOTAL'] = sum(G.edges[u, v]['baseline_peak_gal'].values())
        # (1) Alt. Tech.
        # tons extracted from link flow assignment vector x
        G.edges[u, v][fuel_type + '_avg_ton'] = dict(zip(comm_list, x[1, :, -1, i, 0]))
        G.edges[u, v][fuel_type + '_peak_ton'] = dict(zip(comm_list, x[1, :, tw_m1, i, 0]))
        # loc = loc/car * <commodity_car/ton> * ton
        G.edges[u, v][fuel_type_battery + '_avg_loc'] = dict(zip(comm_list[:-1],
                                                                 np.ceil(tonmi2loc_batt * comm_er *
                                                                         x[1, :-1, -1, i, 0] * mi)))
        G.edges[u, v][fuel_type_battery + '_peak_loc'] = dict(zip(comm_list[:-1],
                                                                  np.ceil(tonmi2loc_batt * comm_er *
                                                                          x[1, :-1, tw_m1, i, 0] * mi)))
        G.edges[u, v][fuel_type_diesel + '_avg_loc'] = dict(zip(comm_list[:-1],
                                                                np.ceil(car2loc * (1 / rr_tc) *
                                                                        x[1, :-1, -1, i, 0])))
        G.edges[u, v][fuel_type_diesel + '_peak_loc'] = dict(zip(comm_list[:-1],
                                                                 np.ceil(car2loc * (1 / rr_tc) *
                                                                         x[1, :-1, tw_m1, i, 0])))
        # kwh = kwh/btu * <hybrid_energy_factor> * <commodity_factor> * ton * mi * btu/ton-mi
        G.edges[u, v][fuel_type_battery + '_avg_kwh'] = dict(zip(comm_list[:-1],
                                                                 (hybrid_energy_factor_btu2kwh * comm_er *
                                                                  x[1, :-1, -1, i, 0] * mi *
                                                                  h_battery_tonmi2btu)))
        G.edges[u, v][fuel_type_battery + '_peak_kwh'] = dict(zip(comm_list[:-1],
                                                                  (hybrid_energy_factor_btu2kwh * comm_er *
                                                                   x[1, :-1, tw_m1, i, 0] * mi *
                                                                   h_battery_tonmi2btu)))
        # gal = gal/btu * <hybrid_energy_factor> * <commodity_factor> * ton * mi * btu/ton-mi
        G.edges[u, v][fuel_type_diesel + '_avg_gal'] = dict(zip(comm_list[:-1],
                                                                (btu2gal * comm_er * x[1, :-1, -1, i, 0] *
                                                                 mi * h_diesel_tonmi2btu)))
        G.edges[u, v][fuel_type_diesel + '_peak_gal'] = dict(zip(comm_list[:-1],
                                                                 (btu2gal * comm_er * x[1, :-1, tw_m1, i, 0] *
                                                                  mi * h_diesel_tonmi2btu)))
        # sum 'TOTAL' values for locomotive and energy flow
        G.edges[u, v][fuel_type_battery + '_avg_loc']['TOTAL'] = \
            sum(G.edges[u, v][fuel_type_battery + '_avg_loc'].values())
        G.edges[u, v][fuel_type_battery + '_peak_loc']['TOTAL'] = \
            sum(G.edges[u, v][fuel_type_battery + '_peak_loc'].values())

        G.edges[u, v][fuel_type_battery + '_avg_kwh']['TOTAL'] = \
            sum(G.edges[u, v][fuel_type_battery + '_avg_kwh'].values())
        G.edges[u, v][fuel_type_battery + '_peak_kwh']['TOTAL'] = \
            sum(G.edges[u, v][fuel_type_battery + '_peak_kwh'].values())

        G.edges[u, v][fuel_type_diesel + '_avg_loc']['TOTAL'] = \
            sum(G.edges[u, v][fuel_type_diesel + '_avg_loc'].values())
        G.edges[u, v][fuel_type_diesel + '_peak_loc']['TOTAL'] = \
            sum(G.edges[u, v][fuel_type_diesel + '_peak_loc'].values())

        G.edges[u, v][fuel_type_diesel + '_avg_gal']['TOTAL'] = \
            sum(G.edges[u, v][fuel_type_diesel + '_avg_gal'].values())
        G.edges[u, v][fuel_type_diesel + '_peak_gal']['TOTAL'] = \
            sum(G.edges[u, v][fuel_type_diesel + '_peak_gal'].values())
        fuel_type
        # (2) Support Diesel
        # tons extracted from link flow assignment vector x
        G.edges[u, v]['support_diesel_avg_ton'] = dict(zip(comm_list, x[2, :, -1, i, 0]))
        G.edges[u, v]['support_diesel_peak_ton'] = dict(zip(comm_list, x[2, :, tw_m2, i, 0]))
        # loc = loc/car * <commodity_car/ton> * ton
        G.edges[u, v]['support_diesel_avg_loc'] = dict(zip(comm_list[:-1],
                                                           np.ceil(car2loc * (1 / rr_tc) * x[2, :-1, -1, i, 0])))
        G.edges[u, v]['support_diesel_peak_loc'] = dict(zip(comm_list[:-1],
                                                            np.ceil(car2loc * (1 / rr_tc) * x[2, :-1, tw_m2, i, 0])))
        # gal = gal/ton-mi * <commodity_factor> * ton-mi
        G.edges[u, v]['support_diesel_avg_gal'] = dict(zip(comm_list[:-1],
                                                           (btu2gal * comm_er * x[2, :-1, -1, i, 0] * mi *
                                                            pure_diesel_tonmi2btu)))
        G.edges[u, v]['support_diesel_peak_gal'] = dict(zip(comm_list[:-1],
                                                            (btu2gal * comm_er * x[2, :-1, tw_m2, i, 0] * mi *
                                                             pure_diesel_tonmi2btu)))
        # sum 'TOTAL' values for locomotive and energy flow
        G.edges[u, v]['support_diesel_avg_loc']['TOTAL'] = sum(G.edges[u, v]['support_diesel_avg_loc'].values())
        G.edges[u, v]['support_diesel_peak_loc']['TOTAL'] = sum(G.edges[u, v]['support_diesel_peak_loc'].values())
        G.edges[u, v]['support_diesel_avg_gal']['TOTAL'] = sum(G.edges[u, v]['support_diesel_avg_gal'].values())
        G.edges[u, v]['support_diesel_peak_gal']['TOTAL'] = sum(G.edges[u, v]['support_diesel_peak_gal'].values())

        # compute and store service shares by fuel technology
        hybrid_tot_flow = G.edges[u, v][fuel_type + '_avg_ton']['TOTAL']
        support_tot_flow = G.edges[u, v]['support_diesel_avg_ton']['TOTAL']
        if H.has_edge(u, v) and hybrid_tot_flow + support_tot_flow > 0:
            G.edges[u, v][fuel_type + '_perc_ton'] = 100 * hybrid_tot_flow / (hybrid_tot_flow + support_tot_flow)
            G.edges[u, v]['support_diesel_perc_ton'] = 100 * support_tot_flow / (hybrid_tot_flow + support_tot_flow)
        elif support_tot_flow == 0:
            G.edges[u, v][fuel_type + '_perc_ton'] = 0
            G.edges[u, v]['support_diesel_perc_ton'] = 0
        else:
            G.edges[u, v][fuel_type + '_perc_ton'] = 0
            G.edges[u, v]['support_diesel_perc_ton'] = 100
    print('\t LINK FLOW EXTRACTION:: %s seconds ---' % round(time.time() - t0, 3))

    # calculate the percentage distance increase for those goods actually rerouted
    baseline_total_tonmi_rerouted = 0  # baseline (original) ton-miles for those ton-miles that were rerouted
    alt_tech_total_tonmi_rerouted = 0  # new ton-miles for those ton-miles that were actually rerouted to alt. tech.
    for od in range(len(f[-1, -1, :, 0])):
        # if the path from the baseline and alt. tech. networks differ for a given OD pair, there was rerouting
        if list(pli_mat[0, :, od]) != list(pli_mat[1, :, od]) and sum(pli_mat[1, :, od]) != 0:
            # calculate the baseline (original) and alt. tech. (new) ton-miles associated with this rerouting
            tons = f[-1, -1, od, 0]
            baseline_total_tonmi_rerouted += tons * sum(G.edges[edges_G[i][0], edges_G[i][1]]['miles']
                                                        for i in np.where(pli_mat[0, :, od] == 1)[0])
            alt_tech_total_tonmi_rerouted += tons * sum(G.edges[edges_G[i][0], edges_G[i][1]]['miles']
                                                        for i in np.where(pli_mat[1, :, od] == 1)[0])

    baseline_total_tonmi = dict(zip(
        comm_list,
        [sum([G.edges[u, v]['baseline_avg_ton'][c] * G.edges[u, v]['miles'] for u, v in G.edges]) for c in comm_list]))
    alt_tech_total_tonmi = dict(zip(
        comm_list,
        [sum([G.edges[u, v][fuel_type + '_avg_ton'][c] * G.edges[u, v]['miles'] for u, v in G.edges])
         for c in comm_list]))
    support_diesel_total_tonmi = dict(zip(
        comm_list,
        [sum([G.edges[u, v]['support_diesel_avg_ton'][c] * G.edges[u, v]['miles'] for u, v in G.edges]) for c in
         comm_list]))
    scenario_total_tonmi = dict(zip(
        comm_list,
        [alt_tech_total_tonmi[c] + support_diesel_total_tonmi[c] for c in comm_list]))
    perc_tonmi_inc = dict(zip(
        comm_list,
        [100 * (scenario_total_tonmi[c] - baseline_total_tonmi[c]) / baseline_total_tonmi[c] for c in comm_list]))
    total_tons = dict(zip(comm_list, [f[comm_idx_dict[c], -1, :, 0].sum() for c in comm_list]))
    G.graph['operations'] = dict(
        baseline_avg_distance_mi=dict(zip(comm_list, [baseline_total_tonmi[c] / total_tons[c] for c in comm_list])),
        baseline_total_tonmi=baseline_total_tonmi,
        baseline_total_annual_tonmi=dict(zip(comm_list, [365 * baseline_total_tonmi[c] for c in comm_list])),
        baseline_total_gal=dict(zip(
            comm_list,
            [sum([G.edges[u, v]['baseline_avg_gal'][c] for u, v in G.edges]) for c in comm_list])),
        scenario_avg_distance_mi=dict(zip(comm_list, [scenario_total_tonmi[c] / total_tons[c] for c in comm_list])),
        alt_tech_total_tonmi=alt_tech_total_tonmi,
        alt_tech_total_annual_tonmi=dict(zip(comm_list, [365 * alt_tech_total_tonmi[c] for c in comm_list])),
        alt_tech_total_locmi=dict(zip(
            comm_list,
            [sum([(G.edges[u, v][fuel_type_battery + '_avg_loc'][c] +
                   G.edges[u, v][fuel_type_diesel + '_avg_loc'][c]) * G.edges[u, v]['miles'] for u, v in G.edges])
             for c in comm_list])),
        support_diesel_total_tonmi=support_diesel_total_tonmi,
        support_diesel_total_annual_tonmi=dict(zip(comm_list,
                                                   [365 * support_diesel_total_tonmi[c] for c in comm_list])),
        scenario_total_tonmi=scenario_total_tonmi,
        scenario_total_annual_tonmi=dict(zip(comm_list, [365 * scenario_total_tonmi[c] for c in comm_list])),
        support_diesel_total_locmi=dict(zip(
            comm_list,
            [sum([G.edges[u, v]['support_diesel_avg_loc'][c] * G.edges[u, v]['miles'] for u, v in G.edges])
             for c in comm_list])),
        support_diesel_total_gal=dict(zip(
            comm_list,
            [sum([G.edges[u, v]['support_diesel_avg_gal'][c] for u, v in G.edges]) for c in comm_list])),
        perc_tonmi_inc=perc_tonmi_inc,
        perc_mi_inc=perc_tonmi_inc,
        perc_tonmi_inc_conditional_reroute=dict(zip(
            comm_list,
            [100 * (alt_tech_total_tonmi_rerouted - baseline_total_tonmi_rerouted) / baseline_total_tonmi_rerouted
             if baseline_total_tonmi_rerouted != 0 else 0 for c in comm_list])),
        deployment_perc=dict(zip(comm_list, [alt_tech_total_tonmi[c] / scenario_total_tonmi[c] for c in comm_list]))
    )

    G.graph['operations'].update(dict(
        alt_tech_total_kwh=dict(zip(
            comm_list,
            [sum([G.edges[u, v][fuel_type_battery + '_avg_kwh'][c] for u, v in G.edges]) for c in comm_list])),
        eff_kwh_p_loc=eff_kwh_p_batt,
        listed_kwh_p_loc=eff_kwh_p_batt * (1 / ft_ef['Effective capacity'])
    ))

    return G, H


def ods_by_perc_ton_mi(G: nx.DiGraph, flow_data_filename: str):
    """

    Parameters
    ----------
    G
    perc_ods
    flow_data_filename
    time_window

    Returns
    -------

    """
    # return O-D pairs in CCWS tha provide ton flows >= <perc_ods> * total CCWS ton flows
    # od_flows is average daily ton-miles

    # load dict that maps SPLC codes to node_ids in G
    splc_node_dict = splc_to_node(G)
    # load grouped OD flow data
    rr = G.graph['railroad']
    flow_df = load_flow_data_df_csv(filename=flow_data_filename, rr=rr)
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

    flow_df = flow_df.groupby(by=['Origin-Destination nodeid']).sum(numeric_only=True)[['Tons']]
    flow_df['Origin-Destination nodeid'] = flow_df.index
    tons = flow_df['Tons'].to_dict()
    # load from json or compute if does not exist
    filepath_sp_dict = os.path.join(NX_DIR, rr + '_SP_dict_miles.json')
    if os.path.exists(filepath_sp_dict):
        miles = load_dict_from_json(filepath_sp_dict)
    else:
        miles = dict(nx.all_pairs_bellman_ford_path_length(G=G, weight='miles'))
        dict_to_json(miles, filepath_sp_dict)
    flow_df['Ton-Miles Routed'] = flow_df['Origin-Destination nodeid'].apply(lambda x:
                                                                                      tons[x] * miles[x[0]][x[1]])
    flow_df.drop(columns=['Origin-Destination nodeid', 'Tons'], inplace=True)
    # group by OD pair nodeid, summing all commodity groupings for the total ton-mile values (over all commodities)
    # keep only dataframe with ton-miles sum
    # flow_df = flow_df.groupby(by=['Origin-Destination nodeid']).sum(numeric_only=True)[['Expanded Ton-Miles Routed']]
    # sort OD pairs by ton-miles in descending order
    flow_df.sort_values(by='Ton-Miles Routed', ascending=False, inplace=True)

    # write to csv
    # flow_df_nodeids = flow_df
    # flow_df_nodeids['O-D nodeid'] = flow_df_nodeids.index
    # flow_df_nodeids['O'] = flow_df_nodeids['O-D nodeid'].apply(lambda x: x[0])
    # flow_df_nodeids['D'] = flow_df_nodeids['O-D nodeid'].apply(lambda x: x[1])
    # flow_df_nodeids.to_csv('/Users/adrianhz/Desktop/OD_flows_USA1.csv')

    # compute cumulative percentage of the ton-miles
    ods = list(flow_df.index)
    # convert OD pair strings into node_id pair tuples
    # get O-D flows for all O-D pairs as a dict
    od_flows = flow_df['Ton-Miles Routed'].to_dict()

    return ods, od_flows
