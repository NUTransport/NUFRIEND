from util import *
# MODULES
from helper import od_pairs, shortest_path, gurobi_suppress_output, node_to_edge_path
from network_representation import remove_from_graph

sns.set_palette('colorblind', n_colors=4)

'''
ILP SOLVER
'''


def facility_network_cycle_ilp(G: nx.Graph, D: float, paths: list, od_flows: dict, suppress_output=True):
    """
    Solve and plot solution for path constrained facility location problem

    For infeasible stretches
    -Can facility set to be priority set nodes and include all 'Other' type nodes as clients in the future
    -Find shortest path from all priority set nodes to ALL nodes (priority set + 'Other') using nx.multi_source_dijkstra
    -Only allow facilities to be placed at priority set nodes and allow for additional variables z_j to mark infeasible
        locations for each of the 'Other' nodes not satisfied.
    :param tolerance:
    :param rr:
    :param all_pairs:
    :param intertypes:
    :param D: [float] range in km
    :param binary_prog:
    :param path_cnstrs_only:
    :param plot:
    :param plot_paths: for plotting only paths
    :param origin:
    :return:
    """

    node_list = list(G.nodes())
    # adjacency matrix based on all pairs shortest path distances
    # extract path-based adjacency matrices for each shortest path from paths_od to apply to model constraints
    cycle_adj_mat_list = cycle_adjacency_matrix(G, paths, D)

    # set up model
    m = gp.Model('Facility Network Problem', env=gurobi_suppress_output(suppress_output))
    # add new variables - for facility location
    # n many variables; 0 <= x_i <= 1 for all i; if binary==True, will be solved as integer program
    x = m.addVars(node_list, obj=1, vtype=GRB.BINARY, name=[str(n) for n in node_list])
    # add objective fxn
    # m.setObjective(gp.quicksum(x[i] for i in node_list), GRB.MINIMIZE)

    path_flows = []  # for storing path_flows
    ods_connected = set()   # for storing O-Ds that are actually connected
    path_info = []
    # add path coverage constraints
    for k in range(len(cycle_adj_mat_list)):
        ca, cao, can, cac = cycle_adj_mat_list[k]
        p = ca.index  # list of nodes on path (ordered)
        o = p[0]  # 0-th node on path
        n = p[-1]  # n-th node on path

        # extract path flow along path <p> based on <od_flows> and specific path O-D pair (<o>, <n>)
        ods_connected.add((o, n))
        path_flows.append(od_flows[o, n] if (o, n) in od_flows.keys() else 0)
        path_info.append((o, G.nodes[o]['city'] + ', ' + G.nodes[o]['state'],
                          n, G.nodes[n]['city'] + ', ' + G.nodes[n]['state']))

        for j_idx in range(len(p)):
            j = p[j_idx]  # nodeid
            ca_j = ca.loc[j]  # access j-th row of ca matrix
            cao_j = cao.loc[j]  # access j-th row of cao matrix
            can_j = can.loc[j]  # access j-th row of can matrix
            cac_j = cac.loc[j]  # access j-th row of cac matrix
            # a_p_j = a_p[j]  # access jth row of a_p matrix
            # i->j via n
            if j != n:
                m.addConstr(gp.quicksum(x[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx + 1, len(p))) +
                            gp.quicksum(x[p[i_idx]] * can_j[p[i_idx]] for i_idx in range(len(p) - 1)) +
                            gp.quicksum(x[p[i_idx]] * cac_j[p[i_idx]] for i_idx in range(j_idx + 1)) >= 1,
                            name='via_n' + str(k) + str(j))
            # i->j via 0
            if j != o:
                m.addConstr(gp.quicksum(x[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx)) +
                            gp.quicksum(x[p[i_idx]] * cao_j[p[i_idx]] for i_idx in range(1, len(p))) +
                            gp.quicksum(x[p[i_idx]] * cac_j[p[i_idx]] for i_idx in range(j_idx + 1)) >= 1,
                            name='via_0' + str(k) + str(j))

    # calculate and store the value for percentage of feasible O-D coverage
    tm_capt = sum(path_flows)
    if 'framework' in G.graph.keys():
        G.graph['framework']['perc_feasible_od_flow'] = tm_capt / sum(od_flows.values())

    # optimize
    m.update()
    m.optimize()
    # extract solution values
    x_val = m.getAttr('x', x).items()  # get facility placement values
    z_val = m.objval  # get objective fxn value
    # print('# Facilities:: %s' % sum(v for _, v in x_val))
    # print('Ton-miles captured:: %s' % sum(path_flows))
    # print('Percentage ton-miles captured:: %s' % (tm_capt / sum(od_flows.values())))
    # print([path_info[k] for k in range(len(cycle_adj_mat_list))])
    # print([(path_info[k][0], path_info[k][2]) for k in range(len(cycle_adj_mat_list))])

    return x_val, z_val, tm_capt


def facility_network_cycle_ilp_select(G: nx.Graph, D: float, paths: list, od_flows: dict, flow_min: float,
                                      suppress_output=False):
    """
    Solve and plot solution for path constrained facility location problem

    For infeasible stretches
    -Can facility set to be priority set nodes and include all 'Other' type nodes as clients in the future
    -Find shortest path from all priority set nodes to ALL nodes (priority set + 'Other') using nx.multi_source_dijkstra
    -Only allow facilities to be placed at priority set nodes and allow for additional variables z_j to mark infeasible
        locations for each of the 'Other' nodes not satisfied.
    :param tolerance:
    :param rr:
    :param all_pairs:
    :param intertypes:
    :param D: [float] range in km
    :param binary_prog:
    :param path_cnstrs_only:
    :param plot:
    :param plot_paths: for plotting only paths
    :param origin:
    :return:
    """

    node_list = list(G.nodes())
    # adjacency matrix based on all pairs shortest path distances
    # extract path-based adjacency matrices for each shortest path from paths_od to apply to model constraints
    # t0 = time.time()
    cycle_adj_mat_list = cycle_adjacency_matrix(G, paths, D)
    # print('CYCLE ADJACENCY MATRIX:: ' + str(time.time() - t0))

    # set up model
    m = gp.Model('Facility Network Problem', env=gurobi_suppress_output(suppress_output))
    # add new variables - for facility location
    # n many variables; 0 <= x_i <= 1 for all i; if binary==True, will be solved as integer program
    x = m.addVars(node_list, obj=1, vtype=GRB.BINARY, name=[str(n) for n in node_list])
    z = m.addVars([k for k in range(len(cycle_adj_mat_list))], obj=0, lb=0, ub=1,
                  name=[str(k) for k in range(len(cycle_adj_mat_list))])

    # add objective fxn
    # m.setObjective(gp.quicksum(x[i] for i in node_list), GRB.MINIMIZE)
    # add path coverage constraints

    # print('Total OD flow:: %.0f' % sum(od_flows.values()))

    path_flows = []     # for storing path_flows
    ods_connected = set()
    path_info = []
    for k in range(len(cycle_adj_mat_list)):
        ca, cao, can, cac = cycle_adj_mat_list[k]
        p = ca.index  # list of nodes on path (ordered)
        o = p[0]  # 0-th node on path (nodeid)
        n = p[-1]  # n-th node on path (nodeid)

        # extract path flow along path <p> based on <od_flows> and specific path O-D pair (<o>, <n>)
        ods_connected.add((o, n))
        path_flows.append(od_flows[o, n] if (o, n) in od_flows.keys() else 0)
        path_info.append((o, G.nodes[o]['city'] + ', ' + G.nodes[o]['state'],
                          n, G.nodes[n]['city'] + ', ' + G.nodes[n]['state']))

        for j_idx in range(len(p)):
            j = p[j_idx]  # nodeid
            ca_j = ca.loc[j]  # access j-th row of ca matrix
            cao_j = cao.loc[j]  # access j-th row of cao matrix
            can_j = can.loc[j]  # access j-th row of can matrix
            cac_j = cac.loc[j]  # access j-th row of cac matrix
            # a_p_j = a_p[j]  # access jth row of a_p matrix
            # i->j via n
            if j != n:
                # m.addConstr(gp.quicksum(x[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx + 1, len(p))) +
                #             gp.quicksum(x[p[i_idx]] * can_j[p[i_idx]] for i_idx in range(len(p) - 1)) >= 1,
                #             name='via_n' + str(k) + str(j))
                m.addConstr(gp.quicksum(x[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx + 1, len(p))) +
                            gp.quicksum(x[p[i_idx]] * can_j[p[i_idx]] for i_idx in range(len(p) - 1)) +
                            gp.quicksum(x[p[i_idx]] * cac_j[p[i_idx]] for i_idx in range(j_idx + 1))
                            >= z[k],
                            name='via_n' + str(k) + str(j))
            # i->j via 0
            if j != o:
                # m.addConstr(gp.quicksum(x[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx)) +
                #             gp.quicksum(x[p[i_idx]] * cao_j[p[i_idx]] for i_idx in range(1, len(p))) >= 1,
                #             name='via_0' + str(k) + str(j))
                m.addConstr(gp.quicksum(x[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx)) +
                            gp.quicksum(x[p[i_idx]] * cao_j[p[i_idx]] for i_idx in range(1, len(p))) +
                            gp.quicksum(x[p[i_idx]] * cac_j[p[i_idx]] for i_idx in range(j_idx, len(p)))
                            >= z[k],
                            name='via_0' + str(k) + str(j))

    # add coverage constraint for a minimum percentage of O-D flow capture <flow_min>
    # the amount of ton-miles the ratio <flow_min> represents for the O-Ds that are connectible by <paths>
    # i.e., the portion of all possible O-D ton-miles that can be served by a given range
    # only sum those for one-way O-D flows (not round-trip, i.e., if i->j included, do not necessarily add flow of j->i)
    tm_min = flow_min * sum(od_flows.values())
    # tm_min = flow_min * sum([od_flows[o, d] for o, d in ods_connected if (o, d) in od_flows.keys()])
    m.addConstr(gp.quicksum(path_flows[k] * z[k] for k in range(len(cycle_adj_mat_list))) >= tm_min, name='coverage')

    # optimize
    m.update()
    m.optimize()
    # extract solution values
    x_val = m.getAttr('x', x).items()  # get facility placement values
    zz_val = m.getAttr('x', z).items()
    z_val = m.objval  # get objective fxn value

    # print('# Facilities:: %s' % sum(v for _, v in x_val))
    tm_capt = sum(path_flows[k] * zz_val[k][1] for k in range(len(cycle_adj_mat_list)))
    # print('Ton-miles captured:: %s' % sum(path_flows[k] * zz_val[k][1] for k in range(len(cycle_adj_mat_list))))
    # print('Percentage ton-miles captured:: %s' % (tm_capt / sum(od_flows.values())))
    # print([path_info[k] for k in range(len(cycle_adj_mat_list)) if zz_val[k][1] == 1])
    path_nodeids = [p.index.to_list() for p, _, _, _ in cycle_adj_mat_list]

    G.graph['framework'] = dict(
        ods=[(path_nodeids[k][0], path_nodeids[k][-1]) for k in range(len(path_nodeids)) if zz_val[k][1]],
        path_nodes=[path_nodeids[k] for k in range(len(path_nodeids)) if zz_val[k][1]],
        path_edges=[node_to_edge_path(path_nodeids[k]) for k in range(len(path_nodeids)) if zz_val[k][1]],
        tm_capt_perc=tm_capt / sum(od_flows.values()),
        tm_capt=tm_capt,
        # od_flows=od_flows,
    )

    return x_val, z_val, tm_capt, G


def max_flow_facility_network_cycle_ilp_select(G: nx.Graph, D: float, paths: list, od_flows: dict, budget: float=None,
                                               suppress_output=False):
    """
    Solve and plot solution for path constrained facility location problem

    For infeasible stretches
    -Can facility set to be priority set nodes and include all 'Other' type nodes as clients in the future
    -Find shortest path from all priority set nodes to ALL nodes (priority set + 'Other') using nx.multi_source_dijkstra
    -Only allow facilities to be placed at priority set nodes and allow for additional variables z_j to mark infeasible
        locations for each of the 'Other' nodes not satisfied.
    :param tolerance:
    :param rr:
    :param all_pairs:
    :param intertypes:
    :param D: [float] range in km
    :param binary_prog:
    :param path_cnstrs_only:
    :param plot:
    :param plot_paths: for plotting only paths
    :param origin:
    :return:
    """

    node_list = list(G.nodes())

    # if budget not specified
    if budget is None:
        # want the upper bound on the maximum flow capture
        budget = len(node_list)

    # adjacency matrix based on all pairs shortest path distances
    # extract path-based adjacency matrices for each shortest path from paths_od to apply to model constraints
    # t0 = time.time()
    # # START CORRIDOR EXAMPLE - COMMENT OUT
    # # ['S48113001906', 'S48231001397', 'S48063001511', 'S48343001491', 'S22017000336', 'S22119000236', 'S22065000403']
    # # [       1,             2,              3,              4,              5,               6,              7      ]
    # od_keys = set((o, d) for o, d in od_flows.keys()).union(set((d, o) for o, d in od_flows.keys()))
    # # Corridor 1:
    # # od_flows_corr = {('S48113001906', 'S22065000403'): 100, ('S22119000236', 'S48231001397'): 10,
    # #                  ('S48113001906', 'S22017000336'): 50, ('S48063001511', 'S22017000336'): 5}
    # # od_flows_corr = {(1, 7): 100, (6, 2): 10, (1, 5): 50, (3, 5): 5}
    # # Corridor 2:
    # od_flows_corr = {('S48113001906', 'S22065000403'): 50, ('S22119000236', 'S48231001397'): 10,
    #                  ('S48113001906', 'S22017000336'): 100, ('S48063001511', 'S22017000336'): 5}
    # # od_flows_corr = {(1, 7): 50, (6, 2): 10, (1, 5): 100, (3, 5): 5}
    # od_keys = od_keys.union(set(od_flows_corr.keys()))
    # od_flows = {(o, d): od_flows_corr[o, d] if (o, d) in od_flows_corr.keys() else 0 for o, d in od_keys}
    # paths = [shortest_path(G, source=o, target=d, weight='km') for o, d in od_flows.keys() if od_flows[o, d] > 0]
    # # END CORRIDOR EXAMPLE

    # # START RANDOM EXAMPLE - COMMENT OUT
    # od_keys = set((o, d) for o, d in od_flows.keys()).union(set((d, o) for o, d in od_flows.keys()))
    # ods_select = list(od_keys).copy()
    # seed = 1000
    # num_ods = 500
    # np.random.seed(seed)
    # od_flows_rand = dict(zip(ods_select[:num_ods], np.random.randint(low=1, high=10000, size=num_ods)))
    # od_keys = od_keys.union(set(od_flows_rand.keys()))
    # od_flows = {(o, d): od_flows_rand[o, d] if (o, d) in od_flows_rand.keys() else 0 for o, d in od_keys}
    # paths = [shortest_path(G, source=o, target=d, weight='km') for o, d in od_flows.keys() if od_flows[o, d] > 0]
    # # END RANDOM EXAMPLE

    cycle_adj_mat_list = cycle_adjacency_matrix(G, paths, D)
    # print('CYCLE ADJACENCY MATRIX:: ' + str(time.time() - t0))

    path_flows = []  # for storing path_flows
    for k in range(len(cycle_adj_mat_list)):
        ca, _, _, _ = cycle_adj_mat_list[k]
        p = ca.index  # list of nodes on path (ordered)
        o = p[0]  # 0-th node on path (nodeid)
        n = p[-1]  # n-th node on path (nodeid)

        # extract path flow along path <p> based on <od_flows> and specific path O-D pair (<o>, <n>)
        path_flows.append(od_flows[o, n] if (o, n) in od_flows.keys() else 0)

    # print('Total OD flow:: %.0f' % sum(od_flows.values()))

    # set up model
    m = gp.Model('Facility Network Problem', env=gurobi_suppress_output(suppress_output))
    # add new variables - for facility location
    # n many variables; 0 <= x_i <= 1 for all i; if binary==True, will be solved as integer program
    x = m.addVars(node_list, obj=0, vtype=GRB.BINARY, name=[str(n) for n in node_list])
    z = m.addVars([k for k in range(len(cycle_adj_mat_list))], obj=path_flows, lb=0, ub=1,
                  name=['z' + str(k) for k in range(len(cycle_adj_mat_list))])

    # add objective fxn
    m.setObjective(gp.quicksum(z[k] * path_flows[k] for k in range(len(cycle_adj_mat_list))), GRB.MAXIMIZE)
    # add path coverage constraints

    # # START SET NODES
    # facility_ids = ['S47151000103', 'S51550001469', 'S1073000406', 'S17179002310', 'S29189001087', 'S18089000359',
    #                 'S19179001790', 'S29077001850', 'S29095000814', 'S46035000568', 'S20079001634', 'S31137001646',
    #                 'S31045000027', 'S48343001491', 'S27003001256', 'S27173001726', 'S30085000307', 'S30077000681',
    #                 'S38015001138', 'S46013000076', 'S56013000257', 'S35009000443', 'S35049000167', 'S4017000236',
    #                 'S8009001357', 'S48439001558', 'S42003001832', 'S48201004876', 'S53063000457', 'S6071002652']
    # for nf in facility_ids:
    #     m.addConstr(x[nf] == 1, name='set_fac' + str(nf))
    # # END SET NODES

    for k in range(len(cycle_adj_mat_list)):
        ca, cao, can, cac = cycle_adj_mat_list[k]
        p = ca.index  # list of nodes on path (ordered)
        o = p[0]  # 0-th node on path (nodeid)
        n = p[-1]  # n-th node on path (nodeid)

        for j_idx in range(len(p)):
            j = p[j_idx]  # nodeid
            ca_j = ca.loc[j]  # access j-th row of ca matrix
            cao_j = cao.loc[j]  # access j-th row of cao matrix
            can_j = can.loc[j]  # access j-th row of can matrix
            cac_j = cac.loc[j]  # access j-th row of cac matrix
            # a_p_j = a_p[j]  # access jth row of a_p matrix
            # i->j via n
            if j != n:
                # m.addConstr(gp.quicksum(x[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx + 1, len(p))) +
                #             gp.quicksum(x[p[i_idx]] * can_j[p[i_idx]] for i_idx in range(len(p) - 1)) >= 1,
                #             name='via_n' + str(k) + str(j))
                m.addConstr(gp.quicksum(x[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx + 1, len(p))) +
                            gp.quicksum(x[p[i_idx]] * can_j[p[i_idx]] for i_idx in range(len(p) - 1)) +
                            gp.quicksum(x[p[i_idx]] * cac_j[p[i_idx]] for i_idx in range(j_idx + 1))
                            >= z[k],
                            name='via_n' + str(k) + str(j))
            # i->j via 0
            if j != o:
                # m.addConstr(gp.quicksum(x[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx)) +
                #             gp.quicksum(x[p[i_idx]] * cao_j[p[i_idx]] for i_idx in range(1, len(p))) >= 1,
                #             name='via_0' + str(k) + str(j))
                m.addConstr(gp.quicksum(x[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx)) +
                            gp.quicksum(x[p[i_idx]] * cao_j[p[i_idx]] for i_idx in range(1, len(p))) +
                            gp.quicksum(x[p[i_idx]] * cac_j[p[i_idx]] for i_idx in range(j_idx, len(p)))
                            >= z[k],
                            name='via_0' + str(k) + str(j))

    # add coverage constraint for a minimum percentage of O-D flow capture <flow_min>
    # the amount of ton-miles the ratio <flow_min> represents for the O-Ds that are connectible by <paths>
    # i.e., the portion of all possible O-D ton-miles that can be served by a given range
    # only sum those for one-way O-D flows (not round-trip, i.e., if i->j included, do not necessarily add flow of j->i)
    # tm_min = flow_min * sum(od_flows.values())
    # tm_min = flow_min * sum([od_flows[o, d] for o, d in ods_connected if (o, d) in od_flows.keys()])
    m.addConstr(gp.quicksum(x[n] for n in node_list) <= budget, name='budget')

    # write ILP model
    # m.write('/Users/adrianhz/Desktop/KCS_test_P2P_ILP.lp')
    # optimize
    m.update()
    m.optimize()
    # extract solution values
    x_val = m.getAttr('x', x).items()  # get facility placement values
    zz_val = m.getAttr('x', z).items()
    z_val = m.objval  # get objective fxn value
    # print('# Facilities:: %s' % sum(v for _, v in x_val))
    # print('Ton-miles captured:: %s' % z_val)
    # print('Percentage ton-miles captured:: %s' % (z_val / sum(od_flows.values())))

    path_nodeids = [p.index.to_list() for p, _, _, _ in cycle_adj_mat_list]
    if 'framework' in G.graph.keys():
        G.graph['framework'] = dict(
            ods=[(path_nodeids[k][0], path_nodeids[k][-1]) for k in range(len(path_nodeids)) if zz_val[k][1]],
            path_nodes=[path_nodeids[k] for k in range(len(path_nodeids)) if zz_val[k][1]],
            path_edges=[node_to_edge_path(path_nodeids[k]) for k in range(len(path_nodeids)) if zz_val[k][1]],
            tm_capt_perc=z_val / sum(od_flows.values()),
            tm_capt=z_val,
            # od_flows=od_flows,
            path_flows=path_flows,
            z=zz_val,
            x=x_val,
            # c=cycle_adj_mat_list,
            p=paths
        )

    return x_val, z_val, G



'''
SOLUTION PROCESSING
'''


def facility_location(G: nx.Graph, D: float, ods=None, od_flows: dict = None, flow_min: float = None,
                      budget: int = None, max_flow=False, extend_graph=True, suppress_output=True):
    """

    Parameters
    ----------
    budget : object
    G
    D
    ods
    od_flows
    flow_min
    select_cycles
    binary_prog
    suppress_output

    Returns
    -------

    """
    # 2. locate facilities and extract graph form of this
    # (for now, looking at all pairs paths b/w terminals 'T', can update based on flow routing information)
    if ods is None:
        ods = od_pairs(G, intertypes={'T'})

    paths = [shortest_path(G, source=o, target=d, weight='km') for o, d in ods]

    if max_flow:
        x_val, _, G = max_flow_facility_network_cycle_ilp_select(G=G, D=D, paths=paths, od_flows=od_flows,
                                                                 budget=budget, suppress_output=suppress_output)
    else:
        x_val, _, _, G = facility_network_cycle_ilp_select(G=G, D=D, paths=paths, od_flows=od_flows,
                                                           flow_min=flow_min, suppress_output=suppress_output)

    G.graph['approx_range_km'] = D
    G.graph['approx_range_mi'] = D / 1.6

    G = covered_graph(G, x_val, D, extend_graph=extend_graph)
    H = selected_subgraph(G)

    return G, H


def covered_graph(G: nx.DiGraph, x_val: list, D: float, extend_graph=True):
    # facility nodes and edges
    facility_set = set(fac_nodeids_lp_sol(x_val))  # set of facility locations by nodeid
    if extend_graph:
        # includes all those edges within the range (out-and-back-in) of the selected facilities
        covered_edges = path_edges_covered(G, fac_set=facility_set, D=D, weight='km')
    else:
        # includes only those edges along the selected O-D paths
        if isinstance(G.graph['framework']['path_edges'], dict):
            covered_edges = [e for p in G.graph['framework']['path_edges'].values() for e in p]
        else:
            covered_edges = [e for p in G.graph['framework']['path_edges'] for e in p]
        covered_edges = list({(u, v) for (u, v) in covered_edges}.union({(v, u) for (u, v) in covered_edges}))
    # include both endpoints of each covered edge as a covered node
    covered_nodes = {u for u, _ in covered_edges}.union({v for _, v in covered_edges})

    for n in G.nodes:
        if n in facility_set:
            G.nodes[n]['facility'] = 1
        else:
            G.nodes[n]['facility'] = 0
        if n in covered_nodes:
            G.nodes[n]['covered'] = 1
        else:
            G.nodes[n]['covered'] = 0

    for u, v in G.edges:
        if (u, v) in covered_edges:
            G.edges[u, v]['covered'] = 1
        else:
            G.edges[u, v]['covered'] = 0

    G.graph['number_facilities'] = sum(G.nodes[n]['facility'] for n in G)
    if extend_graph:
        # includes only those edges along the selected O-D paths
        if isinstance(G.graph['framework']['path_edges'], dict):
            covered_edges = [e for p in G.graph['framework']['path_edges'].values() for e in p]
        else:
            covered_edges = [e for p in G.graph['framework']['path_edges'] for e in p]
        covered_edges = list({(u, v) for (u, v) in covered_edges}.union({(v, u) for (u, v) in covered_edges}))
        # include both endpoints of each covered edge as a covered node
        covered_nodes = {u for u, _ in covered_edges}.union({v for _, v in covered_edges})

        G.graph['number_covered'] = len(covered_nodes)
        G.graph['number_covered_extended'] = sum(G.nodes[n]['covered'] for n in G)
    else:
        G.graph['number_covered'] = sum(G.nodes[n]['covered'] for n in G)
        G.graph['number_covered_extended'] = sum(G.nodes[n]['covered'] for n in G)

    return G


def selected_subgraph(G):
    path_nodes = {n for n in G if G.nodes[n]['covered'] == 1}
    edge_set = {(u, v) for u, v in G.edges if G.edges[u, v]['covered'] == 1}
    edges_to_remove = set(G.edges()).difference(set(edge_set))
    nodes_to_remove = set(G.nodes()).difference(set(path_nodes))
    # Graph with feasible subnetwork(s) for coverage of technology
    return remove_from_graph(G, nodes_to_remove=nodes_to_remove, edges_to_remove=edges_to_remove, connected_only=False)


def cycle_adjacency_matrix(G: nx.Graph, paths: list, D: float):
    # take in G: undirected graph, paths: list of list of nodeids on path for all paths of interest in G, D: range of tech.
    # return: dict with multikey = (path_index (in <paths>), nodeids (in <paths[path_index]>)) and 3-tuple entry
    #  (a_ij, a^0_ij, a^n_ij) is the adjacency indicator for ij on path (simple, longway via node 0, longway via node n)
    # Steps - Dynamically update/grow dict by each path in <paths>:
    # 1. calculate path distance matrix for each path in paths on G
    # 2. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # 3. for each (i,j) in path calculate:
    #   i)      a_ij (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)

    feasible_paths = []
    for k in range(len(paths)):
        p = paths[k]
        if not nx.has_path(G, source=p[0], target=p[-1]):
            paths.pop(k)
            continue
        p_dists = []
        for i, j in zip(p[:-1], p[1:]):
            # p_dists[a] = G.edges[p[a], p[a+1]]['km'], distance to exit node in index a to next node on path
            p_dists.append(G.edges[i, j]['km'])
        infeas_idx = np.where(np.array(p_dists) > D)[0] + 1

        # print(any([set(p).issubset(set(fp)) for fp in feasible_paths]))

        if len(infeas_idx) > 0:
            infeas_idx = np.insert(infeas_idx, 0, 0)
            infeas_idx = np.insert(infeas_idx, len(infeas_idx), len(p))
            for i, j in zip(infeas_idx[:-1], infeas_idx[1:]):
                sub_p = p[i:j]
                # if len(sub_p) > 1 and not any([set(sub_p).issubset(set(fp)) for fp in feasible_paths]):
                if len(sub_p) > 1:
                    feasible_paths.append(sub_p)
        # elif not any([set(p).issubset(set(fp)) for fp in feasible_paths]):
        #     feasible_paths.append(p)
        else:
            feasible_paths.append(p)

    cycle_adj_mats = []
    covered_ods = set()
    for k in range(len(feasible_paths)):
        p = feasible_paths[k]
        if (p[0], p[-1]) not in covered_ods:
            # if OD of this path not yet served
            covered_ods.add((p[0], p[-1]))
        else:
            # OD served by this path already served
            continue

        df = pd.DataFrame(data=0, index=p, columns=p)
        for i, j in zip(p[:-1], p[1:]):
            # p_dists[a] = G.edges[p[a], p[a+1]]['km'], distance to exit node in index a to next node on path
            df.loc[i, j] = G.edges[i, j]['km']
            df.loc[j, i] = df.loc[i, j]
        for i_idx in range(len(p)):
            for j_idx in range(i_idx + 2, len(p)):
                df.loc[p[i_idx], p[j_idx]] = sum([df.loc[p[u], p[u + 1]] for u in range(i_idx, j_idx)])
                df.loc[p[j_idx], p[i_idx]] = df.loc[p[i_idx], p[j_idx]]
        df_a = pd.DataFrame(data=0, index=p, columns=p)
        df_ao = pd.DataFrame(data=0, index=p, columns=p)
        df_an = pd.DataFrame(data=0, index=p, columns=p)
        df_ac = pd.DataFrame(data=0, index=p, columns=p)
        o = p[0]
        n = p[-1]
        for i in p:
            for j in p:
                d = df.loc[i, j]  # i to j via shortest (direct) path
                d_o = df.loc[i, o] + df.loc[o, j]  # i to j via o (0-th node on path)
                d_n = df.loc[i, n] + df.loc[n, j]  # i to j via n (n-th node on path)
                d_c = df.loc[i, n] + df.loc[n, o] + df.loc[o, j]  # i to j via n->o (return path)
                df_a.loc[i, j] = int(d <= D)
                df_ao.loc[i, j] = int(d_o <= D)
                df_an.loc[i, j] = int(d_n <= D)
                df_ac.loc[i, j] = int(d_c <= D)

        cycle_adj_mats.append((df_a, df_ao, df_an, df_ac))

    return cycle_adj_mats


def cycle_deviation_adjacency_matrix(G: nx.Graph, ods: list, od_flows: dict, D: float, od_flow_perc: float = 1):
    # take in G: undirected graph, paths: list list of nodeids on path for all paths of interest in G, D: range of tech.
    # return: dict with multikey = (path_index (in <paths>), nodeids (in <paths[path_index]>)) and 3-tuple entry
    #  (a_ij, a^0_ij, a^n_ij) is the adjacency indicator for ij on path (simple, longway via node 0, longway via node n)
    # Steps - Dynamically update/grow dict by each path in <paths>:
    # 0. generate shortest paths for each O-D pair in <ods>
    # 1. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # 2. calculate path distance matrix for each path on G
    # 3. generate set of paths (incl. super paths) for each O-D pair
    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)

    # 0. generate shortest paths for each O-D pair in <ods>

    # # START CORRIDOR TEST
    # mat_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' + 'corridor_adjacency_mat.pkl')
    # dev_paths_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' + 'corridor_deviation_paths.pkl')
    # # END CORRIDOR TEST - OUTCOMMENT BELOW
    # # START RANDOM TEST
    # mat_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' +
    #                             'random_s1000_n500_adjacency_mat.pkl')
    # dev_paths_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' +
    #                                   'random_s1000_n500_deviation_paths.pkl')
    # # END RANDOM TEST - OUTCOMMENT BELOW
    mat_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_'
                                + str(od_flow_perc) + '_adjacency_mat.pkl')
    dev_paths_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_'
                                      + str(od_flow_perc) + '_deviation_paths.pkl')
    if os.path.exists(mat_filepath) and os.path.exists(dev_paths_filepath):
        return pkl.load(open(mat_filepath, 'rb')), pkl.load(open(dev_paths_filepath, 'rb'))

    # paths = [shortest_path(G, source=o, target=d, weight='km') for o, d in ods]

    # 1. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # for k in range(len(paths)):
    #     p = paths[k]
    #     if not nx.has_path(G, source=p[0], target=p[-1]):
    #         paths.pop(k)
    #         continue
    #     p_dists = []
    #     for i, j in zip(p[:-1], p[1:]):
    #         # p_dists[a] = G.edges[p[a], p[a+1]]['km'], distance to exit node in index a to next node on path
    #         p_dists.append(G.edges[i, j]['km'])
    #     infeas_idx = np.where(np.array(p_dists) > D)[0] + 1
    #
    #     if len(infeas_idx) > 0:
    #         infeas_idx = np.insert(infeas_idx, 0, 0)
    #         infeas_idx = np.insert(infeas_idx, len(infeas_idx), len(p))
    #         for i, j in zip(infeas_idx[:-1], infeas_idx[1:]):
    #             sub_p = p[i:j]
    #             # if len(sub_p) > 1 and not any([set(sub_p).issubset(set(fp)) for fp in feasible_paths]):
    #             if len(sub_p) > 1:
    #                 feasible_paths.append(sub_p)
    #     # elif not any([set(p).issubset(set(fp)) for fp in feasible_paths]):
    #     #     feasible_paths.append(p)
    #     else:
    #         feasible_paths.append(p)

    # 3. generate set of paths (incl. super paths) for each O-D pair
    # ods = [(o, d) for o, d in ods if (o, d) in od_flows.keys() and od_flows[o, d] > 0]
    # paths = {(p[0], p[-1]): p for p in paths if (p[0], p[-1]) in ods}
    ods = [(o, d) for o, d in ods if ((o, d) in od_flows.keys() and od_flows[o, d] > 0) or
           ((d, o) in od_flows.keys() and od_flows[d, o] > 0)]
    paths = {(o, d): shortest_path(G, source=o, target=d, weight='km') for o, d in ods}
    print('ODs %d  Paths %d' %(len(ods), len(paths)))
    deviation_paths = {od: [p] for od, p in paths.items()}
    unique_ods = set()
    # for each O-D pair (i, j)
    for i, j in ods:
        if (i, j) not in unique_ods and (j, i) not in unique_ods:
            unique_ods.update({(i, j)})
        # for each O-D pair (k, l)
        for k, l in ods:
            # if O-D (or its reverse) is already selected, skip
            if (k == i and l == j) or (k == j and l == i):
                continue
            # if O-D (or its reverse) is not already selected
            else:
                # get base path for O-D pair (k, l)
                p_kl = paths[k, l]
                # if i and j on base path for O-D pair (k, l)
                if i in p_kl and j in p_kl:
                    i_idx = p_kl.index(i)
                    j_idx = p_kl.index(j)
                    # if j comes after i on this path
                    if i_idx < j_idx:
                        # orient superpath: i->k + k+1->l-1 + l->j
                        # p_kl = [k, ..., i, ..., j, ..., l] => [i, ..., k, ..., i, ..., j, ..., l, ..., j]
                        p_iklj = list(reversed(p_kl[:i_idx + 1])) + p_kl[1:-1] + list(reversed(p_kl[j_idx:]))
                        # add this new path
                        deviation_paths[i, j].append(p_iklj)
                        # if the reverse O-D pair (j, i) exists in the set to consider
                        # if (j, i) in deviation_paths.keys():
                        #     # add this new path reversed
                        #     deviation_paths[j, i].append(list(reversed(p_iklj)))
                    # if i comes after j on this path
                    else:
                        # orient superpath the opposite way: i->l + l->k + k->j
                        p_ilkj = p_kl[i_idx:] + list(reversed(p_kl[1:-1])) + p_kl[:j_idx + 1]
                        # add this new path
                        deviation_paths[i, j].append(p_ilkj)
                        # if the reverse O-D pair (j, i) exists in the set to consider
                        # if (j, i) in deviation_paths.keys():
                        #     # add this new path reversed
                        #     deviation_paths[j, i].append(list(reversed(p_ilkj)))

    # remove repeated deviation paths for each OD (keep only unique ones) and
    # create reverse of all deviation paths for return paths for each O-D pair (i, j)
    for i, j in unique_ods:
        deviation_paths[i, j] = [list(q) for q in set(tuple(p) for p in deviation_paths[i, j])]
        deviation_paths[j, i] = [list(reversed(p)) for p in deviation_paths[i, j]]

    print(len(deviation_paths))
    print(sum(len(deviation_paths[od]) for od in deviation_paths.keys()))

    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)
    # each entry of mats is indexed by actual nodeid
    # G.edges[i, j]['km']

    A = {(o, d): [[[0, 0, 0] for the in range(len(deviation_paths[d, o]))]
                  for phi in range(len(deviation_paths[o, d]))] for o, d in ods}
    print(2 * sum(3 * len(A[od])**2 for od in A.keys()))
    B = {(d, o): [[[0, 0, 0] for the in range(len(deviation_paths[d, o]))]
                  for phi in range(len(deviation_paths[o, d]))] for o, d in ods}
    counter = 0
    for o, d in ods:
        counter += 1
        if counter % 500 == 0:
            print(counter)
        for phi_idx in range(len(deviation_paths[o, d])):
            phi = deviation_paths[o, d][phi_idx]
            # phi_rev = deviation_paths[d, o][phi_idx]    # reverse

            df_phi = pd.DataFrame(data=0, index=list(set(phi)), columns=list(set(phi)))
            a_phi = pd.DataFrame(data=0, index=list(set(phi)), columns=list(set(phi)))
            # neighboring nodes (from edges on path), use this to get baseline distances between nodes on path
            for i, j in zip(phi[:-1], phi[1:]):
                if a_phi.loc[i, j] != 1:
                    df_phi.loc[i, j] = G.edges[i, j]['km']
                    df_phi.loc[j, i] = df_phi.loc[i, j]  # reverse
                    a_phi.loc[i, j] = int(df_phi.loc[i, j] <= D)  # i to j
                    a_phi.loc[j, i] = int(df_phi.loc[j, i] <= D)  # reverse
            for i_idx in range(len(phi)):
                for j_idx in range(i_idx + 2, len(phi)):
                    i = phi[i_idx]
                    j = phi[j_idx]
                    if a_phi.loc[i, j] != 1:
                        # adjacency/range indicator
                        df_phi.loc[i, j] = sum(df_phi.loc[phi[u], phi[u + 1]] for u in range(i_idx, j_idx))
                        df_phi.loc[j, i] = df_phi.loc[i, j]  # reverse
                        a_phi.loc[i, j] = int(df_phi.loc[i, j] <= D)  # i to j via shortest (direct) path
                        a_phi.loc[j, i] = int(df_phi.loc[j, i] <= D)  # reverse

            for the_idx in range(len(deviation_paths[d, o])):
                the = deviation_paths[d, o][the_idx]
                # the_rev = deviation_paths[o, d][the_idx]    # reverse

                df_the = pd.DataFrame(data=0, index=list(set(the)), columns=list(set(the)))
                for i, j in zip(the[:-1], the[1:]):
                    if df_the.loc[i, j] == 0:
                        df_the.loc[i, j] = G.edges[i, j]['km']
                        df_the.loc[j, i] = df_the.loc[i, j]  # reverse
                for i_idx in range(len(the)):
                    for j_idx in range(i_idx + 2, len(the)):
                        i = the[i_idx]
                        j = the[j_idx]
                        if df_the.loc[i, j] == 0:
                            df_the.loc[i, j] = sum(df_the.loc[the[u], the[u + 1]] for u in range(i_idx, j_idx))
                            df_the.loc[j, i] = df_the.loc[i, j]  # reverse

                a_cir = pd.DataFrame(data=0, index=list(set(phi)), columns=list(set(phi)))
                a_cir_rev = pd.DataFrame(data=0, index=list(set(phi)), columns=list(set(phi)))
                phi_the_set = list(set(phi).union(set(the)))
                a_bar = pd.DataFrame(data=0, index=phi_the_set, columns=phi_the_set)
                a_bar_rev = pd.DataFrame(data=0, index=phi_the_set, columns=phi_the_set)
                # for each i along phi
                for i_idx in range(len(phi)):
                    i = phi[i_idx]
                    for j_idx in range(i_idx + 1):
                        j = phi[j_idx]
                        # if j is before or up to i along phi
                        if j_idx <= i_idx and a_cir.loc[i, j] != 1:
                            # i to j, using the as the return (long-way round) path from i->d->o->j
                            a_cir.loc[i, j] = int((df_phi.loc[i, d] + df_the.loc[d, o] + df_phi.loc[o, j]) <= D)
                        # reverse: if j is at i or after along phi (means j is at or before i along phi_rev)
                        if j_idx >= i_idx and a_cir_rev.loc[i, j] != 1:
                            a_cir_rev.loc[i, j] = int((df_phi.loc[j, d] + df_the.loc[d, o] + df_phi.loc[o, i]) <= D)

                    for j in the:
                        # i to j via n, where i is on phi and j is on the: i->n->j
                        if a_bar.loc[i, j] != 1:
                            a_bar.loc[i, j] = int((df_phi.loc[i, d] + df_the.loc[d, j]) <= D)
                        if a_bar_rev.loc[i, j] != 1:
                            a_bar_rev.loc[i, j] = int((df_phi.loc[i, o] + df_the.loc[o, j]) <= D)   # reverse

                A[o, d][phi_idx][the_idx] = [a_phi, a_cir, a_bar]
                # reverse
                B[d, o][phi_idx][the_idx] = [a_phi, a_cir_rev, a_bar_rev]

    A.update(B)

    # store <A> and <deviation_paths>, as this takes a lot of time to precompute
    A_str = {str(od): A[od] for od in A.keys()}
    with open(mat_filepath, 'wb') as f:
        pkl.dump(A, f)
        f.close()
    deviation_paths_str = {str(od): deviation_paths[od] for od in deviation_paths.keys()}
    with open(dev_paths_filepath, 'wb') as f:
        pkl.dump(deviation_paths, f)
        f.close()

    return A, deviation_paths


def cycle_deviation_adjacency_matrix_nr(G: nx.Graph, ods: list, od_flows: dict, D: float, od_flow_perc: float = 1):
    # non-redundant path sets
    # take in G: undirected graph, paths: list list of nodeids on path for all paths of interest in G, D: range of tech.
    # return: dict with multikey = (path_index (in <paths>), nodeids (in <paths[path_index]>)) and 3-tuple entry
    #  (a_ij, a^0_ij, a^n_ij) is the adjacency indicator for ij on path (simple, longway via node 0, longway via node n)
    # Steps - Dynamically update/grow dict by each path in <paths>:
    # 0. generate shortest paths for each O-D pair in <ods>
    # 1. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # 2. calculate path distance matrix for each path on G
    # 3. generate set of paths (incl. super paths) for each O-D pair
    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)

    # 0. generate shortest paths for each O-D pair in <ods>

    # # START CORRIDOR TEST
    # mat_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' + 'corridor_adjacency_mat.pkl')
    # dev_paths_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' + 'corridor_deviation_paths.pkl')
    # # END CORRIDOR TEST - OUTCOMMENT BELOW
    # # START RANDOM TEST
    # mat_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' +
    #                             'random_s1000_n500_adjacency_mat.pkl')
    # dev_paths_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' +
    #                                   'random_s1000_n500_deviation_paths.pkl')
    # # END RANDOM TEST - OUTCOMMENT BELOW
    mat_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_'
                                + str(od_flow_perc) + '_adjacency_mat_nr.pkl')
    dev_paths_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_'
                                      + str(od_flow_perc) + '_deviation_paths_nr.pkl')
    if os.path.exists(mat_filepath) and os.path.exists(dev_paths_filepath):
        return pkl.load(open(mat_filepath, 'rb')), pkl.load(open(dev_paths_filepath, 'rb'))

    # paths = [shortest_path(G, source=o, target=d, weight='km') for o, d in ods]

    # 1. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # for k in range(len(paths)):
    #     p = paths[k]
    #     if not nx.has_path(G, source=p[0], target=p[-1]):
    #         paths.pop(k)
    #         continue
    #     p_dists = []
    #     for i, j in zip(p[:-1], p[1:]):
    #         # p_dists[a] = G.edges[p[a], p[a+1]]['km'], distance to exit node in index a to next node on path
    #         p_dists.append(G.edges[i, j]['km'])
    #     infeas_idx = np.where(np.array(p_dists) > D)[0] + 1
    #
    #     if len(infeas_idx) > 0:
    #         infeas_idx = np.insert(infeas_idx, 0, 0)
    #         infeas_idx = np.insert(infeas_idx, len(infeas_idx), len(p))
    #         for i, j in zip(infeas_idx[:-1], infeas_idx[1:]):
    #             sub_p = p[i:j]
    #             # if len(sub_p) > 1 and not any([set(sub_p).issubset(set(fp)) for fp in feasible_paths]):
    #             if len(sub_p) > 1:
    #                 feasible_paths.append(sub_p)
    #     # elif not any([set(p).issubset(set(fp)) for fp in feasible_paths]):
    #     #     feasible_paths.append(p)
    #     else:
    #         feasible_paths.append(p)

    # 3. generate set of paths (incl. super paths) for each O-D pair
    # ods = [(o, d) for o, d in ods if (o, d) in od_flows.keys() and od_flows[o, d] > 0]
    # paths = {(p[0], p[-1]): p for p in paths if (p[0], p[-1]) in ods}
    ods = [(o, d) for o, d in ods if ((o, d) in od_flows.keys() and od_flows[o, d] > 0) or
           ((d, o) in od_flows.keys() and od_flows[d, o] > 0)]
    paths = {(o, d): shortest_path(G, source=o, target=d, weight='km') for o, d in ods}
    print('ODs %d  Paths %d' %(len(ods), len(paths)))
    deviation_paths = {od: [p] for od, p in paths.items()}
    unique_ods = set()
    # for each O-D pair (i, j)
    for i, j in ods:
        if (i, j) not in unique_ods and (j, i) not in unique_ods:
            unique_ods.update({(i, j)})
        # for each O-D pair (k, l)
        for k, l in ods:
            # if O-D (or its reverse) is already selected, skip
            if (k == i and l == j) or (k == j and l == i):
                continue
            # if O-D (or its reverse) is not already selected
            else:
                # get base path for O-D pair (k, l)
                p_kl = paths[k, l]
                # if i and j on base path for O-D pair (k, l)
                if i in p_kl and j in p_kl:
                    i_idx = p_kl.index(i)
                    j_idx = p_kl.index(j)
                    # if j comes after i on this path
                    if i_idx < j_idx:
                        # orient superpath: i->k + k+1->l-1 + l->j
                        # p_kl = [k, ..., i, ..., j, ..., l] => [i, ..., k, ..., i, ..., j, ..., l, ..., j]
                        p_iklj = list(reversed(p_kl[:i_idx + 1])) + p_kl[1:-1] + list(reversed(p_kl[j_idx:]))
                        # add this new path
                        deviation_paths[i, j].append(p_iklj)
                        # if the reverse O-D pair (j, i) exists in the set to consider
                        # if (j, i) in deviation_paths.keys():
                        #     # add this new path reversed
                        #     deviation_paths[j, i].append(list(reversed(p_iklj)))
                    # if i comes after j on this path
                    else:
                        # orient superpath the opposite way: i->l + l->k + k->j
                        p_ilkj = p_kl[i_idx:] + list(reversed(p_kl[1:-1])) + p_kl[:j_idx + 1]
                        # add this new path
                        deviation_paths[i, j].append(p_ilkj)
                        # if the reverse O-D pair (j, i) exists in the set to consider
                        # if (j, i) in deviation_paths.keys():
                        #     # add this new path reversed
                        #     deviation_paths[j, i].append(list(reversed(p_ilkj)))

    # remove repeated deviation paths for each OD (keep only unique ones) and
    # create reverse of all deviation paths for return paths for each O-D pair (i, j)
    for i, j in unique_ods:
        deviation_paths[i, j] = [list(q) for q in set(tuple(p) for p in deviation_paths[i, j])]
        deviation_paths[j, i] = [list(reversed(p)) for p in deviation_paths[i, j]]

    print(len(deviation_paths))
    print(sum(len(deviation_paths[od]) for od in deviation_paths.keys()))

    # 4. for each (i,j) in path calculate:
    #   i)      a_ij = (d_ij <= D),
    #   ii)     a^0_ij (d_{i,path[0]} + d_{path[0],j} <= D),
    #   iii)    a^n_ij (d_{i,path[-1]} + d_{path[-1],j} <= D)
    # each entry of mats is indexed by actual nodeid
    # G.edges[i, j]['km']

    A = {(o, d): [[[0, 0, 0]] for phi in range(len(deviation_paths[o, d]))] for o, d in ods}
    print(2 * sum(3 * len(A[od])**2 for od in A.keys()))
    B = {(d, o): [[[0, 0, 0]] for the in range(len(deviation_paths[d, o]))] for o, d in ods}
    counter = 0
    for o, d in ods:
        counter += 1
        if counter % 500 == 0:
            print(counter)
        for phi_idx in range(len(deviation_paths[o, d])):
            phi = deviation_paths[o, d][phi_idx]
            # phi_rev = deviation_paths[d, o][phi_idx]    # reverse

            df_phi = pd.DataFrame(data=0, index=list(set(phi)), columns=list(set(phi)))
            a_phi = pd.DataFrame(data=0, index=list(set(phi)), columns=list(set(phi)))
            # neighboring nodes (from edges on path), use this to get baseline distances between nodes on path
            for i, j in zip(phi[:-1], phi[1:]):
                if a_phi.loc[i, j] != 1:
                    df_phi.loc[i, j] = G.edges[i, j]['km']
                    df_phi.loc[j, i] = df_phi.loc[i, j]  # reverse
                    a_phi.loc[i, j] = int(df_phi.loc[i, j] <= D)  # i to j
                    a_phi.loc[j, i] = int(df_phi.loc[j, i] <= D)  # reverse
            for i_idx in range(len(phi)):
                for j_idx in range(i_idx + 2, len(phi)):
                    i = phi[i_idx]
                    j = phi[j_idx]
                    if a_phi.loc[i, j] != 1:
                        # adjacency/range indicator
                        df_phi.loc[i, j] = sum(df_phi.loc[phi[u], phi[u + 1]] for u in range(i_idx, j_idx))
                        df_phi.loc[j, i] = df_phi.loc[i, j]  # reverse
                        a_phi.loc[i, j] = int(df_phi.loc[i, j] <= D)  # i to j via shortest (direct) path
                        a_phi.loc[j, i] = int(df_phi.loc[j, i] <= D)  # reverse

            # reverse path
            the = deviation_paths[d, o][phi_idx]

            df_the = pd.DataFrame(data=0, index=list(set(the)), columns=list(set(the)))
            for i, j in zip(the[:-1], the[1:]):
                if df_the.loc[i, j] == 0:
                    df_the.loc[i, j] = G.edges[i, j]['km']
                    df_the.loc[j, i] = df_the.loc[i, j]  # reverse
            for i_idx in range(len(the)):
                for j_idx in range(i_idx + 2, len(the)):
                    i = the[i_idx]
                    j = the[j_idx]
                    if df_the.loc[i, j] == 0:
                        df_the.loc[i, j] = sum(df_the.loc[the[u], the[u + 1]] for u in range(i_idx, j_idx))
                        df_the.loc[j, i] = df_the.loc[i, j]  # reverse

            a_cir = pd.DataFrame(data=0, index=list(set(phi)), columns=list(set(phi)))
            a_cir_rev = pd.DataFrame(data=0, index=list(set(phi)), columns=list(set(phi)))
            phi_the_set = list(set(phi).union(set(the)))
            a_bar = pd.DataFrame(data=0, index=phi_the_set, columns=phi_the_set)
            a_bar_rev = pd.DataFrame(data=0, index=phi_the_set, columns=phi_the_set)
            # for each i along phi
            for i_idx in range(len(phi)):
                i = phi[i_idx]
                for j_idx in range(i_idx + 1):
                    j = phi[j_idx]
                    # if j is before or up to i along phi
                    if j_idx <= i_idx and a_cir.loc[i, j] != 1:
                        # i to j, using the as the return (long-way round) path from i->d->o->j
                        a_cir.loc[i, j] = int((df_phi.loc[i, d] + df_the.loc[d, o] + df_phi.loc[o, j]) <= D)
                    # reverse: if j is at i or after along phi (means j is at or before i along phi_rev)
                    if j_idx >= i_idx and a_cir_rev.loc[i, j] != 1:
                        a_cir_rev.loc[i, j] = int((df_phi.loc[j, d] + df_the.loc[d, o] + df_phi.loc[o, i]) <= D)

                for j in the:
                    # i to j via n, where i is on phi and j is on the: i->n->j
                    if a_bar.loc[i, j] != 1:
                        a_bar.loc[i, j] = int((df_phi.loc[i, d] + df_the.loc[d, j]) <= D)
                    if a_bar_rev.loc[i, j] != 1:
                        a_bar_rev.loc[i, j] = int((df_phi.loc[i, o] + df_the.loc[o, j]) <= D)   # reverse

            A[o, d][phi_idx][0] = [a_phi, a_cir, a_bar]
            # reverse
            B[d, o][phi_idx][0] = [a_phi, a_cir_rev, a_bar_rev]

    A.update(B)

    # store <A> and <deviation_paths>, as this takes a lot of time to precompute
    A_str = {str(od): A[od] for od in A.keys()}
    with open(mat_filepath, 'wb') as f:
        pkl.dump(A, f)
        f.close()
    deviation_paths_str = {str(od): deviation_paths[od] for od in deviation_paths.keys()}
    with open(dev_paths_filepath, 'wb') as f:
        pkl.dump(deviation_paths, f)
        f.close()

    return A, deviation_paths


def path_adjacency_matrix(G: nx.Graph, paths: list, D: float):
    # take in G: undirected graph, paths: list list of nodeids on path for all paths of interest in G, D: range of tech.
    # return: dict with multikey = (path_index (in <paths>), nodeids (in <paths[path_index]>)) and entry
    #  a_ij is the simple path adjacency indicator for ij on path
    # Steps - Dynamically update/grow dict by each path in <paths>:
    # 1. calculate path distance matrix for each path in paths on G
    # 2. split paths if d_ij > D for any (i,j) in each matrix into [(0,...i), (j,...,n)]
    # 3. for each (i,j) in path calculate: a_ij (d_ij <= D)

    feasible_paths = []
    for k in range(len(paths)):
        p = paths[k]
        if not nx.has_path(G, source=p[0], target=p[-1]):
            paths.pop(k)
            continue
        p_dists = []
        for i, j in zip(p[:-1], p[1:]):
            # p_dists[a] = G.edges[p[a], p[a+1]]['km'], distance to exit node in index a to next node on path
            p_dists.append(G.edges[i, j]['km'])
        infeas_idx = np.where(np.array(p_dists) > D)[0] + 1
        if len(infeas_idx) > 0:
            infeas_idx = np.insert(infeas_idx, 0, 0)
            infeas_idx = np.insert(infeas_idx, len(infeas_idx), len(p))
            for i, j in zip(infeas_idx[:-1], infeas_idx[1:]):
                sub_p = p[i:j]
                if len(sub_p) > 1 and not any([set(sub_p).issubset(set(fp)) for fp in feasible_paths]):
                    feasible_paths.append(sub_p)
        elif not any([set(p).issubset(set(fp)) for fp in feasible_paths]):
            feasible_paths.append(p)

    path_adj_mats = []
    for k in range(len(feasible_paths)):
        p = feasible_paths[k]
        df = pd.DataFrame(data=0, index=p, columns=p)
        for i, j in zip(p[:-1], p[1:]):
            # p_dists[a] = G.edges[p[a], p[a+1]]['km'], distance to exit node in index a to next node on path
            df.loc[i, j] = G.edges[i, j]['km']
            df.loc[j, i] = df.loc[i, j]
        for i_idx in range(len(p)):
            for j_idx in range(i_idx + 2, len(p)):
                df.loc[p[i_idx], p[j_idx]] = sum([df.loc[p[u], p[u + 1]] for u in range(i_idx, j_idx)])
                df.loc[p[j_idx], p[i_idx]] = df.loc[p[i_idx], p[j_idx]]
        df_a = pd.DataFrame(data=0, index=p, columns=p)
        for i in p:
            for j in p:
                d = df.loc[i, j]  # i to j via shortest (direct) path
                df_a.loc[i, j] = int(d <= D)

        path_adj_mats.append(df_a)

    return path_adj_mats


def fac_nodeids_lp_sol(x_val: list):
    # not relevant if solution is integer

    # analyze solution output by LP and convert to list of selected facilities
    fac_list = []  # list of facility locations by nodeid
    not_selected = []  # list of facility locations not selected by nodeid
    eps = 1e-4  # margin of error for integrality
    # for each decision variable in the solution
    for i, v in x_val:
        # if its value is sufficiently great to be considered a selection; x_i~=1 means i is selected
        if v >= (1 - eps):
            # append the nodeid to list of selected facilities
            fac_list.append(i)

    return fac_list


# def path_edges_covered(G: nx.Graph, fac_set: set, D: float, weight: str = 'km') -> list:
#     # create list of edges connected by selected facilities within the range
#     path_dict = dict(nx.all_pairs_dijkstra_path(G, cutoff=D, weight=weight))
#
#     path_edges = set()
#     for u in path_dict.keys():
#         if u not in fac_set:
#             continue
#         for v in path_dict[u].keys():
#             if v not in fac_set or u == v:
#                 continue
#             p = path_dict[u][v]
#             edges = {e for e in zip(p[:-1], p[1:])}
#             path_edges.update(edges)
#
#     path_dict_d2 = dict(nx.all_pairs_dijkstra_path(G, cutoff=D / 2, weight=weight))
#     for u in path_dict_d2.keys():
#         if u not in fac_set:
#             continue
#         for v in path_dict_d2[u].keys():
#             # print(nx.shortest_path(G, source=u, target=v))
#             p = path_dict_d2[u][v]
#             edges = {e for e in zip(p[:-1], p[1:])}
#             path_edges.update(edges)
#
#     return list(path_edges)


def path_edges_covered(G: nx.Graph, fac_set: set, D: float, weight: str = 'km') -> list:
    # create list of edges connected by selected facilities within the range

    node_list = list(G.nodes)
    dist_mat = nx.floyd_warshall_numpy(G=G, nodelist=node_list, weight=weight)

    node_idx_dict = {node_list[i]: i for i in range(len(node_list))}
    fac_idxs = [node_idx_dict[i] for i in fac_set]

    visited_edges = set()
    covered_edges = set()
    for i, j in G.edges:
        u, v = (node_idx_dict[i], node_idx_dict[j])
        if (i, j) not in visited_edges:
            visited_edges.update({(i, j), (j, i)})
            d_ku = dist_mat[fac_idxs, u].min()
            d_vk = dist_mat[v, fac_idxs].min()
            if d_ku + G.edges[i, j][weight] + d_vk <= D:
                covered_edges.update({(i, j), (j, i)})
    # path_edges = set()
    # for u in path_dict.keys():
    #     if u not in fac_set:
    #         continue
    #     for v in path_dict[u].keys():
    #         if v not in fac_set or u == v:
    #             continue
    #         p = path_dict[u][v]
    #         edges = {e for e in zip(p[:-1], p[1:])}
    #         path_edges.update(edges)
    #
    # path_dict_d2 = dict(nx.all_pairs_dijkstra_path(G, cutoff=D / 2, weight=weight))
    # for u in path_dict_d2.keys():
    #     if u not in fac_set:
    #         continue
    #     for v in path_dict_d2[u].keys():
    #         # print(nx.shortest_path(G, source=u, target=v))
    #         p = path_dict_d2[u][v]
    #         edges = {e for e in zip(p[:-1], p[1:])}
    #         path_edges.update(edges)

    return list(covered_edges)

