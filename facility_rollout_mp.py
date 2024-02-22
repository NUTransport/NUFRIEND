from util import *
from helper import gurobi_suppress_output, node_to_edge_path, shortest_path
from plotting_mp import *

red = matplotlib.cm.get_cmap('Set1').colors[0]
[teal, orange, blue, pink, green, yellow, brown, grey] = matplotlib.cm.get_cmap('Set2').colors


# LOCATION


def facility_location_mp(G: nx.DiGraph, D: float, time_horizon: list, od_flows: dict, facility_costs: dict = None,
                         max_flow=False, greedy=False, flow_mins: float = None, budgets: list = None,
                         candidate_facilities: list = None, discount_rates: any = None, fixed_facilities: dict = None,
                         nested=True, extend_graph=False, od_flow_perc: float = 1.0,
                         binary_prog=False, suppress_output=True):
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

    if max_flow:
        if greedy:
            G, y_val, _ = greedy_max_flow_facility_cycle(G=G, D=D, time_horizon=time_horizon, od_flows=od_flows,
                                                         candidate_facilities=candidate_facilities, budgets=budgets,
                                                         facility_costs=facility_costs, discount_rates=discount_rates,
                                                         fixed_facilities=fixed_facilities,
                                                         nested=nested, od_flow_perc=od_flow_perc,
                                                         binary_prog=binary_prog, suppress_output=suppress_output)
        else:
            G, y_val, _, _ = max_flow_facility_cycle_mp_ilp(G=G, D=D, time_horizon=time_horizon, od_flows=od_flows,
                                                            candidate_facilities=candidate_facilities, budgets=budgets,
                                                            facility_costs=facility_costs,
                                                            discount_rates=discount_rates,
                                                            fixed_facilities=fixed_facilities,
                                                            nested=nested, od_flow_perc=od_flow_perc,
                                                            binary_prog=binary_prog, suppress_output=suppress_output)
    else:
        G, y_val, _ = min_cost_facility_cycle_mp_ilp(G=G, D=D, time_horizon=time_horizon, od_flows=od_flows,
                                                     candidate_facilities=candidate_facilities, flow_mins=flow_mins,
                                                     facility_costs=facility_costs, discount_rates=discount_rates,
                                                     fixed_facilities=fixed_facilities,
                                                     nested=nested, od_flow_perc=od_flow_perc,
                                                     binary_prog=binary_prog, suppress_output=suppress_output)

    # if select_cycles:
    #     # for b in range(1, 8):
    #     #     _, _ = max_flow_facility_network_cycle_ilp_select(G=G, D=D, paths=paths, od_flows=od_flows, budget=b,
    #     #                                                       binary_prog=binary_prog, suppress_output=True)
    #     if max_flow:
    #         if deviation_paths:
    #             x_val, _, _, G = max_flow_facility_deviation_paths_select(G=G, D=D, ods=ods, od_flows=od_flows,
    #                                                                       budget=budget, od_flow_perc=od_flow_perc,
    #                                                                       binary_prog=binary_prog,
    #                                                                       suppress_output=True)
    #         else:
    #             x_val, _ = max_flow_facility_network_cycle_ilp_select(G=G, D=D, paths=paths, od_flows=od_flows,
    #                                                                   budget=budget,
    #                                                                   binary_prog=binary_prog, suppress_output=True)
    #     else:
    #         if deviation_paths:
    #             x_val, _, _, G = facility_deviation_paths_select(G=G, D=D, ods=ods, od_flows=od_flows,
    #                                                              flow_min=flow_min, od_flow_perc=od_flow_perc,
    #                                                              binary_prog=binary_prog, suppress_output=True)
    #         else:
    #             x_val, _, _, G = facility_network_cycle_ilp_select(G=G, D=D, paths=paths, od_flows=od_flows,
    #                                                                flow_min=flow_min,
    #                                                                binary_prog=binary_prog, suppress_output=True)
    # else:
    #     # store path information for future analysis
    #     G.graph['framework'] = dict(ods=ods, path_nodes=paths, path_edges=[node_to_edge_path(p) for p in paths])
    #     x_val, _, _ = facility_network_cycle_ilp(G=G, D=D, paths=paths, od_flows=od_flows,
    #                                              binary_prog=binary_prog, suppress_output=suppress_output)

    G.graph['approx_range_km'] = D
    G.graph['approx_range_mi'] = D / 1.6

    G = covered_graph(G, D, extend_graph=extend_graph)
    # here we construct the final time step subgraph and index each node and edge by the first time step in which
    #  they appear in the solution in fields 'time_step_facility', 'time_step_covered' for nodes
    #  and 'time_step_covered' for edges
    # H = selected_subgraphs(G)
    # or a list of subgraphs, each with H.graph['time_step'] = time_step
    # Hs = {time_step: selected_subgraph(G, time_step) for time_step in time_horizon}

    return G


def input_cleaning(G: nx.DiGraph, time_horizon: list, od_flows: dict,
                   candidate_facilities: dict = None, budgets: dict = None, flow_mins: dict = None,
                   facility_costs: dict = None, discount_rates: any = None):
    node_list = list(G.nodes())
    ods = list(od_flows.keys())

    # candidate facilites not provided, assume all facilities
    if candidate_facilities is None:
        candidate_facilities = {t: set(node_list) for t in time_horizon}
    # provided candidate facilities not for full time_horizon, extend
    missing_times = [t for t in time_horizon if t not in candidate_facilities.keys()]
    if len(missing_times) > 0:
        for t in missing_times:
            candidate_facilities[t] = set(node_list)
    # ensure all candidate facilities are subsets: a candidate facility at step t is included in all future steps t+k
    for t_idx in range(len(time_horizon) - 1):
        if not candidate_facilities[time_horizon[t_idx]].issubset(candidate_facilities[time_horizon[t_idx + 1]]):
            candidate_facilities[time_horizon[t_idx + 1]].add(
                candidate_facilities[time_horizon[t_idx]].difference(candidate_facilities[time_horizon[t_idx + 1]]))

    # budgets not provided, assume to be # of facilities
    if budgets is None:
        budgets = {t: len(candidate_facilities[t]) for t in time_horizon}
    # provided budgets not for full time_horizon, extend
    else:
        missing_times = [t for t in time_horizon if t not in budgets.keys()]
        if len(missing_times) > 0:
            for t in missing_times:
                budgets[t] = len(candidate_facilities[t])

    # facility costs not provided, assume all candidate facilities have the same cost
    if facility_costs is None:
        facility_costs = {(n, t): 1 for t in time_horizon for n in node_list if n in candidate_facilities[t]}
    # facilities with no costs specified
    # if len(set(node_list).difference(set(facility_costs.keys()))) != 0:
    #     missing_facilities = set(node_list).difference(set(facility_costs.keys()))
    #     for d in missing_facilities:
    #         facility_costs[d] = {t: np.inf for t in time_horizon}
    # facilities with missing time_horizon costs
    # missing_times = [n for n in node_list if len(facility_costs[n]) < T]
    # if len(missing_times) > 0:
    #     t0 = time_horizon[0]
    #     for d in missing_times:
    #         # if the smallest time period is empty, set it to np.inf
    #         if t0 not in facility_costs[d].keys():
    #             facility_costs[d][t0] = np.inf
    #     # fill in remaining time steps
    #     for t_idx in range(1, len(time_horizon)):
    #         for d in missing_times:
    #             # if the value for this time step is missing, take on the value of the previously available time step
    #             if time_horizon[t_idx] not in facility_costs[d].keys():
    #                 facility_costs[d][time_horizon[t_idx]] = facility_costs[d][time_horizon[t_idx - 1]]
    # facility_costs = {(n, t): facility_costs[n][t] for n, t in product(node_list, time_horizon)}

    # discount_rates not provided, assumed all to be 1
    if discount_rates is None:
        discount_rates = {t: 1 for t in time_horizon}
    # provided discount_rates not for full time_horizon, extend
    elif isinstance(discount_rates, float) or isinstance(discount_rates, int):
        discount_rates = {t: (1 / (1 + discount_rates)) ** t for t in time_horizon}
    else:
        missing_times = [t for t in time_horizon if t not in discount_rates.keys()]
        if len(missing_times) > 0:
            for t in missing_times:
                discount_rates[t] = 1

    # OD pairs with missing time_horizon flows
    missing_times = set(od for od in ods if not isinstance(od_flows[od], dict))
    if len(missing_times) > 0:
        t0 = time_horizon[0]
        for od in missing_times:
            # if the smallest time period is empty, set it to 0
            if not isinstance(od_flows[od], dict):
                od_flows[od] = {t0: od_flows[od]}
                # fill in remaining time steps
                for t_idx in range(1, len(time_horizon)):
                    # take on the value of the previously available time step
                    od_flows[od][time_horizon[t_idx]] = od_flows[od][time_horizon[t_idx - 1]]

    return od_flows, candidate_facilities, budgets, facility_costs, discount_rates


def max_flow_facility_cycle_mp_ilp(G: nx.Graph, D: float, time_horizon: list, od_flows: dict,
                                   candidate_facilities: dict = None, budgets: dict = None,
                                   facility_costs: dict = None, discount_rates: any = None,
                                   fixed_facilities: dict = None, barred_facilities: dict = None,
                                   nested=True, od_flow_perc: float = 1.0, binary_prog=True, suppress_output=False):
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

    [od_flows, candidate_facilities, budgets,
     facility_costs, discount_rates] = input_cleaning(G=G, time_horizon=time_horizon, od_flows=od_flows,
                                                      candidate_facilities=candidate_facilities,
                                                      budgets=budgets, flow_mins=None,
                                                      facility_costs=facility_costs, discount_rates=discount_rates)

    t0 = time.time()
    ods = list(od_flows.keys())
    paths = [shortest_path(G, source=o, target=d, weight='km') for o, d in ods]
    print('SHORTEST PATHS:: ' + str(time.time() - t0))

    t0 = time.time()
    cycle_adj_mat_list = cycle_adjacency_matrix_mp(G, paths, D, od_flow_perc)
    print('CYCLE ADJACENCY MATRIX:: ' + str(time.time() - t0))

    paths = dict()  # for storing new paths (nodeids) indexed by (o, d)
    path_dists = dict()  # for storing path distances (in miles)
    path_flows = dict()  # for storing path_flows
    for k in range(len(cycle_adj_mat_list)):
        ca, _, _, _ = cycle_adj_mat_list[k]
        p = ca.index  # list of nodes on path (ordered)
        o = p[0]  # origin node on path (nodeid)
        d = p[-1]  # destination node on path (nodeid)
        paths[(o, d)] = p
        path_dists[(o, d)] = sum(G.edges[u, v]['miles'] for (u, v) in zip(p[0:-1], p[1:]))
        # extract path flow along path <p> based on <od_flows> and specific path O-D pair (<o>, <d>) for all time steps
        if (o, d) in od_flows.keys():
            path_flows[o, d] = {t: od_flows[o, d][t] for t in time_horizon if t in od_flows[o, d].keys()}
        # else:
        #     path_flows[o, d] = {t: 0 for t in time_horizon}

    ods = [od for od in path_flows.keys()]
    scale = 1e-5
    path_flows = {((o, d), t): path_flows[o, d][t] * scale for o, d in path_flows.keys()
                  for t in path_flows[o, d].keys()}
    path_tons = {(od, t): path_flows[od, t] / path_dists[od] if path_dists[od] > 0 else 0
                 for od, t in path_flows.keys()}

    # set up model
    m = gp.Model('Facility Rollout Problem', env=gurobi_suppress_output(suppress_output))
    # add new variables - for facility location
    # n many variables; 0 <= x_i <= 1 for all i; if binary==True, will be solved as integer program
    fac_vars, fac_costs = gp.multidict(facility_costs)
    od_vars, od_fs = gp.multidict(path_flows)
    if binary_prog:
        # y = m.addVars(fac_vars, obj=fac_costs, vtype=GRB.BINARY, name=[str(i) for i in fac_vars])
        y = m.addVars(fac_vars, obj=0, vtype=GRB.BINARY, name=[str(j) for j in fac_vars])
        z = m.addVars(od_vars, obj=od_fs, vtype=GRB.BINARY, name=[str(k) for k in od_vars])
    else:
        # y = m.addVars(fac_vars, obj=fac_costs, lb=0, ub=1, name=[str(i) for i in fac_vars])
        y = m.addVars(fac_vars, obj=0, vtype=GRB.BINARY, name=[str(j) for j in fac_vars])
        z = m.addVars(od_vars, obj=od_fs, lb=0, ub=1, name=[str(k) for k in od_vars])

    # add objective fxn
    m.setObjective(gp.quicksum(discount_rates[t] * gp.quicksum(z[od, t] * path_flows[od, t]
                                                               for od in ods if (od, t) in od_vars)
                               for t in time_horizon), GRB.MAXIMIZE)
    # add path coverage constraints

    for k in range(len(cycle_adj_mat_list)):
        ca, cao, can, cac = cycle_adj_mat_list[k]
        p = ca.index  # list of nodes on path (ordered)
        o = p[0]  # origin node on path (nodeid)
        d = p[-1]  # destination node on path (nodeid)

        for j_idx in range(len(p)):
            j = p[j_idx]  # nodeid
            ca_j = ca.loc[j]  # access j-th row of ca matrix
            cao_j = cao.loc[j]  # access j-th row of cao matrix
            can_j = can.loc[j]  # access j-th row of can matrix
            cac_j = cac.loc[j]  # access j-th row of cac matrix
            # a_p_j = a_p[j]  # access jth row of a_p matrix
            # i->j via n
            if j != d:
                m.addConstrs((gp.quicksum(y[p[i_idx], t] * ca_j[p[i_idx]] for i_idx in range(j_idx + 1, len(p))
                                          if (p[i_idx], t) in fac_vars) +
                              gp.quicksum(y[p[i_idx], t] * can_j[p[i_idx]] for i_idx in range(len(p) - 1)
                                          if (p[i_idx], t) in fac_vars) +
                              gp.quicksum(y[p[i_idx], t] * cac_j[p[i_idx]] for i_idx in range(j_idx + 1)
                                          if (p[i_idx], t) in fac_vars)
                              >= z[(o, d), t] for t in time_horizon if ((o, d), t) in od_vars), name='via D')
            # i->j via 0
            if j != o:
                m.addConstrs((gp.quicksum(y[p[i_idx], t] * ca_j[p[i_idx]] for i_idx in range(j_idx)
                                          if (p[i_idx], t) in fac_vars) +
                              gp.quicksum(y[p[i_idx], t] * cao_j[p[i_idx]] for i_idx in range(1, len(p))
                                          if (p[i_idx], t) in fac_vars) +
                              gp.quicksum(y[p[i_idx], t] * cac_j[p[i_idx]] for i_idx in range(j_idx, len(p))
                                          if (p[i_idx], t) in fac_vars)
                              >= z[(o, d), t] for t in time_horizon if ((o, d), t) in od_vars), name='via O')

    # add coverage constraint for a minimum percentage of O-D flow capture <flow_min>
    # the amount of ton-miles the ratio <flow_min> represents for the O-Ds that are connectible by <paths>
    # i.e., the portion of all possible O-D ton-miles that can be served by a given range
    # only sum those for one-way O-D flows (not round-trip, i.e., if i->j included, do not necessarily add flow of j->i)
    # tm_min = flow_min * sum(od_flows.values())
    # tm_min = flow_min * sum([od_flows[o, d] for o, d in ods_connected if (o, d) in od_flows.keys()])
    # budget constraints
    for t_idx, t in enumerate(time_horizon):
        if t_idx == 0:
            m.addConstr(gp.quicksum(fac_costs[j, t] * y[j, t] for j in candidate_facilities[t]) <= budgets[t],
                        name='budget' + str(t))
        else:
            m.addConstr(gp.quicksum(fac_costs[j, t] * (y[j, t] - y[j, time_horizon[t_idx - 1]])
                                    for j in candidate_facilities[t]) <= budgets[t], name='budget' + str(t))

    if nested:
        # facility nestedness constraints
        m.addConstrs((y[j, time_horizon[t_idx]] <= y[j, time_horizon[t_idx + 1]]
                      for t_idx in range(len(time_horizon) - 1) for j in candidate_facilities[time_horizon[t_idx]]),
                     name='facility_nestedness')
        # O-D pair nestedness constraints (should be implied by facility nestedness and coverage constraints
        # m.addConstrs((z[od, time_horizon[t_idx]] <= z[od, u] for od in ods
        #               for t_idx in range(len(time_horizon)) for u in time_horizon[t_idx + 1:]),
        #              name='od_nestedness')
        # m.addConstrs((z[od, time_horizon[t_idx]] <= z[od, time_horizon[t_idx + 1]] for od in ods
        #               for t_idx in range(len(time_horizon) - 1)),
        #              name='od_nestedness')

    if fixed_facilities:
        # facilities that must be built (based on facilities selected in previous time steps)
        m.addConstrs((y[j, t] == 1 for t in fixed_facilities.keys() for j in fixed_facilities[t]),
                     name='fixed_facilities')

    if barred_facilities:
        # facilities that are not to be built (based on facilities not selected in future time steps)
        m.addConstrs((y[j, t] == 0 for t in barred_facilities.keys() for j in barred_facilities[t]),
                     name='barred_facilities')

    # write ILP model
    # m.write('/Users/adrianhz/Desktop/KCS_test_ILP.lp')
    # m.write('/Users/adrianhz/Desktop/KCS_test_unnested_ILP.lp')
    # optimize
    m.update()
    m.optimize()
    # extract solution values
    y_val = m.getAttr('x', y).items()  # get facility placement values
    z_val = m.getAttr('x', z).items()
    obj_val = m.objval  # get objective fxn value

    y_val = {i: v for i, v in y_val}
    z_val = {i: v for i, v in z_val}

    # print('# Facilities:: %s' % sum(v for _, v in x_val))
    # print('Ton-miles captured:: %s' % z_val)
    # print('Percentage ton-miles captured:: %s' % (z_val / sum(od_flows.values())))

    # path_nodeids = [p.index for p, _, _, _ in cycle_adj_mat_list]
    G.graph['framework'] = dict(
        time_horizon=time_horizon,
        facility_costs=facility_costs,
        candidate_facilities=candidate_facilities,
        discount_rates=discount_rates,
        budgets=budgets,
        cum_budgets={t_step: sum(budgets[t] for t in time_horizon[:t_idx + 1])
                     for t_idx, t_step in enumerate(time_horizon)},
        path_flows={k: path_flows[k] / scale for k in path_flows.keys()},
        path_dists_mi=path_dists,
        path_tons=path_tons,
        ods=ods,
        paths=paths,
        selected_ods={t: set(od for od in ods if z_val[od, t]) for t in time_horizon},
        selected_facilities={t: set(n for n in candidate_facilities[t] if y_val[n, t]) for t in time_horizon},
        covered_path_nodes={t: {od: paths[od] for od in ods if z_val[od, t]} for t in time_horizon},
        covered_path_edges={t: {od: node_to_edge_path(paths[od]) for od in ods if z_val[od, t]}
                            for t in time_horizon},
        tm_available={t: sum(path_flows[od, t] / scale for od in ods) for t in time_horizon},
        tm_capt={t: sum(z_val[od, t] * path_flows[od, t] / scale for od in ods) for t in time_horizon},
        tm_capt_perc={t: (sum(z_val[od, t] * path_flows[od, t] for od in ods) /
                          sum([path_flows[od, t] for od in ods])) for t in time_horizon},
        tm_capt_final=obj_val / scale,
        tm_capt_perc_final=obj_val / sum([path_flows[(o, d), time_horizon[-1]] for o, d in ods]),
        od_flows=od_flows,
        z_val=z_val,
        y_val=y_val,
        c=cycle_adj_mat_list,
    )

    return G, y_val, z_val, obj_val


def min_cost_facility_cycle_mp_ilp(G: nx.Graph, D: float, time_horizon: list, od_flows: dict,
                                   candidate_facilities: dict = None, flow_mins: dict = None,
                                   facility_costs: dict = None, discount_rates: any = None,
                                   fixed_facilities: list = None,
                                   nested=True, od_flow_perc: float = 1.0,
                                   binary_prog=True, suppress_output=False):
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

    [od_flows, candidate_facilities, _,
     facility_costs, discount_rates] = input_cleaning(G=G, time_horizon=time_horizon, od_flows=od_flows,
                                                      candidate_facilities=candidate_facilities,
                                                      budgets=None, flow_mins=flow_mins,
                                                      facility_costs=facility_costs, discount_rates=discount_rates)

    node_list = list(G.nodes)

    t0 = time.time()
    ods = list(od_flows.keys())
    paths = [shortest_path(G, source=o, target=d, weight='km') for o, d in ods]
    print('SHORTEST PATHS:: ' + str(time.time() - t0))

    t0 = time.time()
    cycle_adj_mat_list = cycle_adjacency_matrix_mp(G, paths, D, od_flow_perc)
    print('CYCLE ADJACENCY MATRIX:: ' + str(time.time() - t0))

    paths = dict()  # for storing new paths (nodeids) indexed by (o, d)
    path_dists = dict()  # for storing path distances (in miles)
    path_flows = dict()  # for storing path_flows
    for k in range(len(cycle_adj_mat_list)):
        ca, _, _, _ = cycle_adj_mat_list[k]
        p = ca.index  # list of nodes on path (ordered)
        o = p[0]  # origin node on path (nodeid)
        d = p[-1]  # destination node on path (nodeid)
        paths[(o, d)] = p
        path_dists[(o, d)] = sum(G.edges[u, v]['miles'] for (u, v) in zip(p[0:-1], p[1:]))
        # extract path flow along path <p> based on <od_flows> and specific path O-D pair (<o>, <d>) for all time steps
        if (o, d) in od_flows.keys():
            path_flows[o, d] = {t: od_flows[o, d][t] for t in time_horizon if t in od_flows[o, d].keys()}
        # else:
        #     path_flows[o, d] = {t: 0 for t in time_horizon}

    # flow_mins not provided, assume to be 50% of total flows
    if flow_mins is None:
        flow_mins = {t: 0.5 * sum(path_flows[od][t] for od in path_flows.keys()) for t in time_horizon}
    # provided flow_mins are in terms of rate of flow to capture
    else:
        flow_mins = {t: flow_mins[t] * sum(path_flows[od][t] for od in path_flows.keys()) for t in time_horizon}

    ods = [od for od in path_flows.keys()]
    path_flows = {((o, d), t): path_flows[o, d][t] for o, d in path_flows.keys() for t in path_flows[o, d].keys()}
    path_tons = {(od, t): path_flows[od, t] / path_dists[od] for od, t in path_flows.keys()}

    # set up model
    m = gp.Model('Facility Rollout Problem', env=gurobi_suppress_output(suppress_output))
    # add new variables - for facility location
    # n many variables; 0 <= x_i <= 1 for all i; if binary==True, will be solved as integer program
    fac_vars, fac_costs = gp.multidict(facility_costs)
    od_vars, od_fs = gp.multidict(path_flows)
    if binary_prog:
        # y = m.addVars(fac_vars, obj=fac_costs, vtype=GRB.BINARY, name=[str(i) for i in fac_vars])
        y = m.addVars(fac_vars, obj=fac_costs, vtype=GRB.BINARY, name=[str(j) for j in fac_vars])
        z = m.addVars(od_vars, obj=0, vtype=GRB.BINARY, name=[str(k) for k in od_vars])
    else:
        # y = m.addVars(fac_vars, obj=fac_costs, lb=0, ub=1, name=[str(i) for i in fac_vars])
        y = m.addVars(fac_vars, obj=fac_costs, vtype=GRB.BINARY, name=[str(j) for j in fac_vars])
        z = m.addVars(od_vars, obj=0, lb=0, ub=1, name=[str(k) for k in od_vars])

    # add objective fxn
    m.setObjective(gp.quicksum(discount_rates[t] * gp.quicksum(y[j, t] * facility_costs[j, t]
                                                               for j in node_list if (j, t) in fac_vars)
                               for t in time_horizon), GRB.MINIMIZE)
    # add path coverage constraints
    for k in range(len(cycle_adj_mat_list)):
        ca, cao, can, cac = cycle_adj_mat_list[k]
        p = ca.index  # list of nodes on path (ordered)
        o = p[0]  # origin node on path (nodeid)
        d = p[-1]  # destination node on path (nodeid)

        for j_idx in range(len(p)):
            j = p[j_idx]  # nodeid
            ca_j = ca.loc[j]  # access j-th row of ca matrix
            cao_j = cao.loc[j]  # access j-th row of cao matrix
            can_j = can.loc[j]  # access j-th row of can matrix
            cac_j = cac.loc[j]  # access j-th row of cac matrix
            # a_p_j = a_p[j]  # access jth row of a_p matrix
            # i->j via n
            if j != d:
                m.addConstrs((gp.quicksum(y[p[i_idx], t] * ca_j[p[i_idx]] for i_idx in range(j_idx + 1, len(p))
                                          if (p[i_idx], t) in fac_vars) +
                              gp.quicksum(y[p[i_idx], t] * can_j[p[i_idx]] for i_idx in range(len(p) - 1)
                                          if (p[i_idx], t) in fac_vars) +
                              gp.quicksum(y[p[i_idx], t] * cac_j[p[i_idx]] for i_idx in range(j_idx + 1)
                                          if (p[i_idx], t) in fac_vars)
                              >= z[(o, d), t] for t in time_horizon if ((o, d), t) in od_vars), name='via D')
            # i->j via 0
            if j != o:
                m.addConstrs((gp.quicksum(y[p[i_idx], t] * ca_j[p[i_idx]] for i_idx in range(j_idx)
                                          if (p[i_idx], t) in fac_vars) +
                              gp.quicksum(y[p[i_idx], t] * cao_j[p[i_idx]] for i_idx in range(1, len(p))
                                          if (p[i_idx], t) in fac_vars) +
                              gp.quicksum(y[p[i_idx], t] * cac_j[p[i_idx]] for i_idx in range(j_idx, len(p))
                                          if (p[i_idx], t) in fac_vars)
                              >= z[(o, d), t] for t in time_horizon if ((o, d), t) in od_vars), name='via O')

    # add coverage constraint for a minimum percentage of O-D flow capture <flow_min>
    # the amount of ton-miles the ratio <flow_min> represents for the O-Ds that are connectible by <paths>
    # i.e., the portion of all possible O-D ton-miles that can be served by a given range
    # only sum those for one-way O-D flows (not round-trip, i.e., if i->j included, do not necessarily add flow of j->i)
    # tm_min = flow_min * sum(od_flows.values())
    # tm_min = flow_min * sum([od_flows[o, d] for o, d in ods_connected if (o, d) in od_flows.keys()])
    # budget constraints
    m.addConstrs((gp.quicksum(z[od, t] * path_flows[od, t] for od in ods if (od, t) in od_vars) >= flow_mins[t]
                  for t in time_horizon), name='flow_min')

    if nested:
        # facility nestedness constraints
        m.addConstrs((y[j, time_horizon[t_idx]] <= y[j, time_horizon[t_idx + 1]]
                      for t_idx in range(len(time_horizon) - 1) for j in candidate_facilities[time_horizon[t_idx]]),
                     name='facility_nestedness')
        # O-D pair nestedness constraints (should be implied by facility nestedness and coverage constraints
        # m.addConstrs((z[od, time_horizon[t_idx]] <= z[od, u] for od in ods
        #               for t_idx in range(len(time_horizon)) for u in time_horizon[t_idx + 1:]),
        #              name='od_nestedness')
        # m.addConstrs((z[od, time_horizon[t_idx]] <= z[od, time_horizon[t_idx + 1]] for od in ods
        #               for t_idx in range(len(time_horizon) - 1)),
        #              name='od_nestedness')

    if fixed_facilities:
        m.addConstrs((y[j, t] == 1 for t in fixed_facilities.keys() for j in fixed_facilities[t]),
                     name='fixed_facilities')

    # write ILP model
    # m.write('/Users/adrianhz/Desktop/KCS_test_ILP.lp')
    # m.write('/Users/adrianhz/Desktop/KCS_test_unnested_ILP.lp')
    # optimize
    m.update()
    m.optimize()
    # extract solution values
    y_val = m.getAttr('x', y).items()  # get facility placement values
    z_val = m.getAttr('x', z).items()
    obj_val = m.objval  # get objective fxn value

    y_val = {i: v for i, v in y_val}
    z_val = {i: v for i, v in z_val}

    # print('# Facilities:: %s' % sum(v for _, v in x_val))
    # print('Ton-miles captured:: %s' % z_val)
    # print('Percentage ton-miles captured:: %s' % (z_val / sum(od_flows.values())))

    # path_nodeids = [p.index for p, _, _, _ in cycle_adj_mat_list]
    G.graph['framework'] = dict(
        time_horizon=time_horizon,
        facility_costs=facility_costs,
        candidate_facilities=candidate_facilities,
        discount_rates=discount_rates,
        flow_mins=flow_mins,
        path_flows=path_flows,
        path_dists_mi=path_dists,
        path_tons=path_tons,
        ods=ods,
        paths=paths,
        selected_ods={t: set(od for od in ods if z_val[od, t]) for t in time_horizon},
        selected_facilities={t: set(n for n in candidate_facilities[t] if y_val[n, t]) for t in time_horizon},
        covered_path_nodes={t: {od: paths[od] for od in ods if z_val[od, t]} for t in time_horizon},
        covered_path_edges={t: {od: node_to_edge_path(paths[od]) for od in ods if z_val[od, t]}
                            for t in time_horizon},
        tm_available={t: sum(path_flows[od, t] for od in ods) for t in time_horizon},
        tm_capt={t: sum(z_val[od, t] * path_flows[od, t] for od in ods) for t in time_horizon},
        tm_capt_perc={t: (sum(z_val[od, t] * path_flows[od, t] for od in ods) /
                          sum([path_flows[od, t] for od in ods])) for t in time_horizon},
        tm_capt_final=obj_val,
        tm_capt_perc_final=obj_val / sum([path_flows[(o, d), time_horizon[-1]] for o, d in ods]),
        od_flows=od_flows,
        z_val=z_val,
        y_val=y_val,
        c=cycle_adj_mat_list,
    )

    return G, y_val, obj_val


def greedy_max_flow_facility_cycle(G: nx.Graph, D: float, time_horizon: list, od_flows: dict,
                                   candidate_facilities: dict = None, budgets: dict = None,
                                   facility_costs: dict = None, discount_rates: any = None,
                                   fixed_facilities: dict = None,
                                   nested=True, od_flow_perc: float = 1.0, binary_prog=True, suppress_output=False):
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

    [od_flows, candidate_facilities, budgets,
     facility_costs, discount_rates] = input_cleaning(G=G, time_horizon=time_horizon, od_flows=od_flows,
                                                      candidate_facilities=candidate_facilities,
                                                      budgets=budgets, flow_mins=None,
                                                      facility_costs=facility_costs, discount_rates=discount_rates)

    t0 = time.time()
    ods = list(od_flows.keys())
    paths = [shortest_path(G, source=o, target=d, weight='km') for o, d in ods]
    print('SHORTEST PATHS:: ' + str(time.time() - t0))

    t0 = time.time()
    cycle_adj_mat_list = cycle_adjacency_matrix_mp(G, paths, D, od_flow_perc)
    print('CYCLE ADJACENCY MATRIX:: ' + str(time.time() - t0))

    paths = dict()  # for storing new paths (nodeids) indexed by (o, d)
    path_flows = dict()  # for storing path_flows
    for k in range(len(cycle_adj_mat_list)):
        ca, _, _, _ = cycle_adj_mat_list[k]
        p = ca.index  # list of nodes on path (ordered)
        o = p[0]  # origin node on path (nodeid)
        d = p[-1]  # destination node on path (nodeid)
        paths[(o, d)] = p
        # extract path flow along path <p> based on <od_flows> and specific path O-D pair (<o>, <d>) for all time steps
        if (o, d) in od_flows.keys():
            path_flows[o, d] = {t: od_flows[o, d][t] for t in time_horizon if t in od_flows[o, d].keys()}
        # else:
        #     path_flows[o, d] = {t: 0 for t in time_horizon}

    # ods = [od for od in path_flows.keys()]
    path_flows = {(o, d): path_flows[o, d][t] for o, d in path_flows.keys()
                  for t in path_flows[o, d].keys() if t == time_horizon[0]}
    ods = [od for od in path_flows.keys()]

    nodes = list(G.nodes)
    selected_facilities = {0: []}
    obj = dict()
    for b in range(1, len(fixed_facilities[time_horizon[-1]]) + 1):
        print(b)
        m = gp.Model('Facility Rollout Problem', env=gurobi_suppress_output(True))

        fac_vars, fac_costs = gp.multidict(facility_costs)
        od_vars, od_fs = gp.multidict(path_flows)
        if binary_prog:
            # y = m.addVars(fac_vars, obj=fac_costs, vtype=GRB.BINARY, name=[str(i) for i in fac_vars])
            y = m.addVars(nodes, obj=0, vtype=GRB.BINARY, name=[str(j) for j in nodes])
            z = m.addVars(od_vars, obj=od_fs, vtype=GRB.BINARY, name=[str(k) for k in od_vars])
        else:
            # y = m.addVars(fac_vars, obj=fac_costs, lb=0, ub=1, name=[str(i) for i in fac_vars])
            y = m.addVars(nodes, obj=0, vtype=GRB.BINARY, name=[str(j) for j in nodes])
            z = m.addVars(od_vars, obj=od_fs, lb=0, ub=1, name=[str(k) for k in od_vars])

        # in terms of ton-miles
        m.setObjective(gp.quicksum(z[od] * path_flows[od] for od in od_vars), GRB.MAXIMIZE)

        for k in range(len(cycle_adj_mat_list)):
            ca, cao, can, cac = cycle_adj_mat_list[k]
            p = ca.index  # list of nodes on path (ordered)
            o = p[0]  # origin node on path (nodeid)
            d = p[-1]  # destination node on path (nodeid)

            if (o, d) not in od_vars:
                continue

            for j_idx in range(len(p)):
                j = p[j_idx]  # nodeid
                ca_j = ca.loc[j]  # access j-th row of ca matrix
                cao_j = cao.loc[j]  # access j-th row of cao matrix
                can_j = can.loc[j]  # access j-th row of can matrix
                cac_j = cac.loc[j]  # access j-th row of cac matrix
                # a_p_j = a_p[j]  # access jth row of a_p matrix
                # i->j via n
                if j != d:
                    m.addConstr(gp.quicksum(y[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx + 1, len(p))) +
                                gp.quicksum(y[p[i_idx]] * can_j[p[i_idx]] for i_idx in range(len(p) - 1)) +
                                gp.quicksum(y[p[i_idx]] * cac_j[p[i_idx]] for i_idx in range(j_idx + 1))
                                >= z[(o, d)])
                # i->j via 0
                if j != o:
                    m.addConstr(gp.quicksum(y[p[i_idx]] * ca_j[p[i_idx]] for i_idx in range(j_idx)) +
                                gp.quicksum(y[p[i_idx]] * cao_j[p[i_idx]] for i_idx in range(1, len(p))) +
                                gp.quicksum(y[p[i_idx]] * cac_j[p[i_idx]] for i_idx in range(j_idx, len(p)))
                                >= z[(o, d)])

        # add coverage constraint for a minimum percentage of O-D flow capture <flow_min>
        # the amount of ton-miles the ratio <flow_min> represents for the O-Ds that are connectible by <paths>
        # i.e., the portion of all possible O-D ton-miles that can be served by a given range
        # only sum those for one-way O-D flows (not round-trip, i.e., if i->j included, do not necessarily add flow of j->i)
        # tm_min = flow_min * sum(od_flows.values())
        # tm_min = flow_min * sum([od_flows[o, d] for o, d in ods_connected if (o, d) in od_flows.keys()])
        # budget constraints
        m.addConstr(gp.quicksum(y[j] for j in nodes) <= b, name='budget')

        m.addConstrs((y[j] == 0 for j in nodes if j not in fixed_facilities[time_horizon[-1]]), name='final_facilities')
        m.addConstrs((y[j] == 1 for j in selected_facilities[b - 1]), name='selected_facilities')

        m.update()
        m.optimize()
        # extract solution values
        y_val = m.getAttr('x', y).items()  # get facility placement values
        # z_val = m.getAttr('x', z).items()
        obj_val = m.objval  # get objective fxn value

        y_val = {i: v for i, v in y_val}
        # z_val = {i: v for i, v in z_val}

        obj[b] = obj_val  # in ton-miles
        selected_facilities[b] = [j for j, v in y_val.items() if v == 1]

    # print('# Facilities:: %s' % sum(v for _, v in x_val))
    # print('Ton-miles captured:: %s' % z_val)
    # print('Percentage ton-miles captured:: %s' % (z_val / sum(od_flows.values())))

    # path_nodeids = [p.index for p, _, _, _ in cycle_adj_mat_list]

    G, y_val, obj_val = greedy_to_ilp_plotting(G=G, D=D, time_horizon=time_horizon, od_flows=od_flows,
                                               obj=obj, selected_facilities=selected_facilities,
                                               candidate_facilities=candidate_facilities, budgets=budgets,
                                               facility_costs=facility_costs, discount_rates=discount_rates,
                                               nested=nested, od_flow_perc=od_flow_perc,
                                               binary_prog=binary_prog, suppress_output=suppress_output)

    return G, y_val, obj_val


def greedy_to_ilp_plotting(G: nx.Graph, D: float, time_horizon: list, od_flows: dict,
                           obj: dict, selected_facilities: dict,
                           candidate_facilities: dict = None, budgets: dict = None,
                           facility_costs: dict = None, discount_rates: any = None,
                           nested=True, od_flow_perc: float = 1.0, binary_prog=True, suppress_output=False):
    [od_flows, candidate_facilities, budgets,
     facility_costs, discount_rates] = input_cleaning(G=G, time_horizon=time_horizon, od_flows=od_flows,
                                                      candidate_facilities=candidate_facilities,
                                                      budgets=budgets, flow_mins=None,
                                                      facility_costs=facility_costs, discount_rates=discount_rates)
    b = 0
    fixed_facilities = dict()
    cum_flow_capt = dict()
    for t_idx, t in enumerate(time_horizon):
        b += budgets[t]  # cumulative sum of annual budget
        fixed_facilities[t] = selected_facilities[b]
        cum_flow_capt[t] = obj[b]

    G, x_val, obj_val = max_flow_facility_cycle_mp_ilp(G=G, D=D, time_horizon=time_horizon, od_flows=od_flows,
                                                       candidate_facilities=candidate_facilities, budgets=budgets,
                                                       facility_costs=facility_costs, discount_rates=discount_rates,
                                                       fixed_facilities=fixed_facilities,
                                                       nested=nested, od_flow_perc=od_flow_perc,
                                                       binary_prog=binary_prog, suppress_output=suppress_output)

    G.graph['framework']['tm_capt_2'] = cum_flow_capt

    return G, x_val, obj_val


def cycle_adjacency_matrix_mp(G: nx.Graph, paths: list, D: float, od_flow_perc: float = 1.0):
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

    mat_filepath = os.path.join(NX_DIR, G.graph['railroad'] + '_' + str(D) + '_' + str(od_flow_perc) +
                                '_p2p_adjacency_mat_mp.pkl')
    if os.path.exists(mat_filepath):
        return pkl.load(open(mat_filepath, 'rb'))

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

    with open(mat_filepath, 'wb') as f:
        pkl.dump(cycle_adj_mats, f)
        f.close()

    return cycle_adj_mats


def covered_graph(G: nx.DiGraph, D: float, extend_graph=False):
    # facility nodes and edges
    time_horizon = G.graph['framework']['time_horizon']
    selected_facilities = G.graph['framework']['selected_facilities']
    if extend_graph:
        # includes all those edges within the range (out-and-back-in) of the selected facilities
        covered_edges = {t: set(path_edges_covered(G, fac_set=selected_facilities[t], D=D, weight='km'))
                         for t in time_horizon}
    else:
        # includes only those edges along the selected O-D paths
        covered_path_edges = G.graph['framework']['covered_path_edges']
        covered_edges = {t: set((u, v) for p in covered_path_edges[t].values() for u, v in p).union(
            set((v, u) for p in covered_path_edges[t].values() for u, v in p))
            for t in time_horizon}
    # include both endpoints of each covered edge as a covered node
    covered_nodes = {t: set(u for u, _ in covered_edges[t]).union(set(v for _, v in covered_edges[t]))
                     for t in time_horizon}

    for n in G.nodes:
        # suppose time_horizon = [0, 1, 2, 3],
        #  -then G.nodes[n]['facility'] = {0: 0, 1: 1, 2: 1, 3: 1} means a facility was placed at n in time step 1
        #  -and G.nodes[n]['covered'] = {0: 0, 1: 0, 2: 1, 3: 1} means node n was covered starting in time step 2
        G.nodes[n]['facility'] = {t: n in selected_facilities[t] for t in time_horizon}
        G.nodes[n]['covered'] = {t: n in covered_nodes[t] for t in time_horizon}

    for u, v in G.edges:
        G.edges[u, v]['covered'] = {t: (u, v) in covered_edges[t] for t in time_horizon}

    G.graph['number_facilities'] = {t: sum(G.nodes[i]['facility'][t] for i in G) for t in time_horizon}

    return G


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
