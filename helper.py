from util import *


'''
GENERAL
'''


def extract_rr(df: pd.DataFrame, rr: str):
    # extract rows with data for rr from df

    agg_fxn = {'Expanded Number of Samples': np.sum, 'Expanded Tons': np.sum, 'Expanded Carloads': np.sum,
               'Expanded Trailer/Container Count': np.sum, 'Expanded Ton-Miles': np.sum,
               'Expanded Car-Miles': np.sum,
               'Expanded Container-Miles': np.sum, 'Total Distance Mean': np.mean,
               'Total Distance Standard Deviation': np.std, 'Total Distance Max': np.max,
               'Total Distance Min': np.min}
    # # apply aggregation function
    # return df_class1.agg(agg_fxn)

    if rr != 'WCAN' and rr != 'EAST' and rr != 'USA1':
        return df.loc[rr]

    if rr == 'WCAN':
        rrs = ['BNSF', 'CP', 'CN']
    elif rr == 'EAST':
        rrs = ['CSXT', 'NS', 'KCS']
    elif rr == 'USA1':
        rrs = ['BNSF', 'NS', 'KCS']

    orig_idx_names = list(df.index.names)
    orig_idx_names.remove('Railroad')
    # df.reset_index(level=list(set(orig_idx_names).difference({'Railroad'})), inplace=True)
    df.rename(index={r: rr for r in rrs}, level='Railroad', inplace=True)
    df = df.loc[rr]
    # df.groupby(by=orig_idx_names).agg(agg_fxn)
    # apply aggregation function
    return df.groupby(by=orig_idx_names).agg(agg_fxn)


def gurobi_suppress_output(suppress_output=True):

    env = gp.Env(empty=True)
    if suppress_output:
        env.setParam('OutputFlag', 0)
    env.start()
    return env


def load_lca_battery_lookup():
    # return dataframe for emissions factors: index is state abbrev., value is associated [g CO2/kWh] - 2036 projections
    # NUFRIEND
    filename = 'lca_battery_state_2036_g_kwh.csv'
    # SF Paper
    print('SF')
    filename = 'lca_egrid_battery_state_2020_g_kwh.csv'
    # filename = 'lca_egrid_battery_state_2030_g_kwh.csv'

    return pd.read_csv(os.path.join(LCA_DIR, filename), header=0, index_col='state')


def load_lca_hydrogen_lookup():
    # return dataframe for emissions factors; value is associated [g CO2/kgh2] - 2021 and 2034 projections
    filename = 'lca_hydrogen_g_kgh2.csv'

    return pd.read_csv(os.path.join(LCA_DIR, filename), header=0, index_col='Type')


def load_tea_battery_lookup():
    return pd.read_csv(os.path.join(TEA_DIR, 'tea_battery_lookup_table_dollar_kwh.csv'), header=0)


def load_tea_hydrogen_lookup():
    return pd.read_csv(os.path.join(TEA_DIR, 'tea_hydrogen_lookup_table_dollar_kgh2.csv'),
                       header=0, index_col=['Dispenser Type', 'Operation Hours', 'Fleet Size'])


def load_comm_energy_ratios():
    return pd.read_csv(os.path.join(COMM_DIR, 'commodity_energy_ratios.csv'), header=0, index_col='Commodity group')


def load_railroad_battery_LCO_tonmi():
    return pd.read_csv(os.path.join(RR_DIR, 'railroad_battery_LCO_tonmi.csv'), header=0, index_col='Railroad')


def load_railroad_energy_intensities():
    return pd.read_csv(os.path.join(RR_DIR, 'railroad_energy_intensities.csv'), header=0, index_col='Railroad')


def load_railroad_values():
    return pd.read_csv(os.path.join(RR_DIR, 'railroad_values.csv'), header=0, index_col='Railroad')


def load_railroad_loc_car_train():
    return pd.read_csv(os.path.join(RR_DIR, 'railroad_loc_car_train.csv'), header=0, index_col='Railroad')


def load_railroad_comm_ton_car():
    return pd.read_csv(os.path.join(CCWS_DIR, 'WB2019_tons_per_car_rr_comm.csv'), header=0, index_col='Railroad')


def load_conversion_factors():
    """
    Currently have as indices:
        - btu/kwh
        - mi/km
        - btu/gal
        - ton/loc
    :return: [pd.DataFrame]
    """

    return pd.read_csv(os.path.join(GEN_DIR, 'constants.csv'), header=0, index_col='Constant')


def load_fuel_tech_eff_factor():
    return pd.read_csv(os.path.join(GEN_DIR, 'fuel_tech_efficiency_factor.csv'), header=0, index_col='Fuel technology')


'''
network_representation.py
'''


def project_point(coords, from_crs, to_crs):
    g = gpd.GeoDataFrame(geometry=[Point(coords)], crs=from_crs)
    g.to_crs(crs=to_crs, inplace=True)
    p = g.loc[0]['geometry'].coords[0]

    return p[0], p[1]


def RR_line(line_df, rr: str, rrs: list = None):
    # return a boolean series same length as line_df: TRUE = links with RR as any of the rrowners/tr/hr in line_df
    RR_found = [False] * len(line_df)
    RR_col = ['rrowner1', 'rrowner2', 'rrowner3', 'tr1', 'tr2', 'tr3', 'tr4', 'tr5', 'tr6', 'tr7', 'tr8',
              'tr9', 'hr1', 'hr2', 'hr3', 'hr4', 'hr5']
    for col in RR_col:
        if rrs is None:
            RR_found = RR_found | (line_df[col] == rr)
        else:
            RR_found = RR_found | (line_df[col] in rrs)
    return RR_found


def vals_str(x: float) -> str:
    f = 10 ** (len(str(int(x))) - 1)

    return str(int(f * np.ceil(x / f)))


'''
routing.py
'''


# ROUTING ALGORITHMS

def route_od(G: nx.Graph, source, target, inter_nodes: list = [], method=None) -> list:
    """
    Wrapper for applying desired routing method between source and target on graph G; provides list of edges on route
    :param inter_nodes:
    :param G:
    :param source:
    :param target:
    :param method: method name for routing algorithm, must take in:
                    - G: [nx.Graph]
                    - source: [str/int]
                    - target: [str/int]
                    - inter_nodes: [list] may be empty
    :return: [list] of edges on path from <source> to <target> through <inter_nodes> on <G> by routing method <method>
    """
    # default method is shortest path method
    if method is None:
        method = shortest_path_edges

    nodes = [source] + inter_nodes + [target]
    nodes = [update_node(G, n) for n in nodes]
    if None in nodes:
        # one of the nodes cannot be found in the graph
        return []

    return method(G, nodes[0], nodes[-1], inter_nodes=nodes[1:-1])


def shortest_path_edges(G: nx.DiGraph, source, target, weight='km', inter_nodes: list = []) -> list:
    """
    Return list of edges on shortest path in G
    between source and target that passes through inter_nodes in the order listed
    :param inter_nodes:
    :param G:
    :param source:
    :param target:
    :param weight:
    :param inter_nodes:
    :return: [list] of edges, e.g., [(1, 2), (2, 3)] for path from 1->3

    Cicero: 17031000719, LA: 6037003164
    [Galesburg, Burlington, Barstow]: [17095001986, 19057001911, 6071002447]
    s = 17031000719
    e = 6037003164
    t = [17095001986, 19057001911, 6071002447]
    """

    if not nx.has_path(G, source=source, target=target):
        # if path does not exist, return empty list
        return []

    s = source
    node_path = [s]
    for v in inter_nodes:
        if not nx.has_path(G, source=s, target=v):
            # if path does not exist, return empty list
            return []
        node_path.extend(nx.shortest_path(G, s, v, weight=weight)[1:])
        s = v

    node_path.extend(nx.shortest_path(G, s, target, weight=weight)[1:])

    return node_to_edge_path(node_path)


def node_to_edge_path(node_path: list):
    # returns edge format of <node_path> e.g., if <node_path> = [0, 1, 2, 3], returns [(0, 1), (1, 2), (2, 3)]

    return list(zip(node_path[:-1], node_path[1:]))


def shortest_path(G: nx.Graph, source: int, target: int, weight='km', inter_nodes=None) -> list:
    """
    Return shortest path in G between source and target that passes through inter_nodes in the order listed
    :param inter_nodes:
    :param G:
    :param source:
    :param target:
    :param weight:
    :param inter_nodes:
    :return:

    Cicero: 17031000719, LA: 6037003164
    [Galesburg, Burlington, Barstow]: [17095001986, 19057001911, 6071002447]
    s = 17031000719
    e = 6037003164
    t = [17095001986, 19057001911, 6071002447]
    """

    if inter_nodes is None:
        inter_nodes = []
    # update nodeids to the correct super nodeids in G, if they are grouped as such
    inter_nodes = [update_node(G, n) for n in inter_nodes]

    s = update_node(G, source)
    node_path = [s]
    for v in inter_nodes:
        node_path.extend(nx.shortest_path(G, source=s, target=v, weight=weight)[1:])
        s = v

    node_path.extend(nx.shortest_path(G, source=s, target=update_node(G, target), weight=weight)[1:])

    return node_path


# GRAPH MANAGEMENT

def update_node(G: nx.Graph, node):
    # return updated node name that contains the node <node> in G

    find_node = False
    if node not in G:
        find_node = True
    for n in G:
        if find_node:
            if node in G.nodes[n]['original_nodeids']:
                return n
    if find_node:
        return None

    return node


def splc_to_node(G: nx.Graph) -> dict:
    # return dict indexed by splc codes with nodeid values; for routing
    # e.g., splc_node_dict[<splc>] = <nodeid>

    splc_node_dict = dict()

    for n in G:
        splcs = G.nodes[n]['splc']
        if not isinstance(splcs, list):
            splcs = [splcs]
        for s in splcs:
            splc_node_dict[s] = n

    return splc_node_dict


'''
facility_deployment.py
'''


def od_pairs(G: nx.Graph, source=None, target=None, intertypes=None) -> list:
    """
    Generate list of OD pairs possible in G
    :param G: [nx.Graph] railroad name
    :param source: [None]/[int]/[list]
    :param target: [None]/[int]/[list]
    :param intertypes: [None]/[set]
    :return: [list] list of tuples with OD pairs, e.g., [(O1, D1), (O1, D2)]
    """

    if intertypes is None:
        intertypes = {'T', 'P'}

    if source is None and target is None:
        # all pairs shortest path between nodes of 'inttype' in intertypes
        nodes = [n for n in G.nodes() if G.nodes[n]['inttype'] in intertypes]
        n = len(nodes)
        return [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]

    if type(source) is int or type(source) is str:
        source = [source]
    if type(target) is int or type(target) is str:
        target = [target]

    if source is None:
        # target to all (target 'becomes' source, does not affect anything)
        return [(t, n) for t in target for n in G.nodes() if G.nodes[n]['inttype'] in intertypes and not t == n]
    if target is None:
        # source to all
        return [(s, n) for s in source for n in G.nodes() if G.nodes[n]['inttype'] in intertypes and not s == n]
    # many-to-many
    return [(s, t) for s in source for t in target if not s == t]


'''
facility_sizing.py
'''


def load_elec_cost_state_df():
    # return electricity rate for each node based on state rate in [$/MWh]
    # NUFRIEND
    filename = 'average_electricity_price_state.csv'
    # SF Paper
    # print('SF')
    # filename = 'eia_average_electricity_price_state_2020.csv'

    return pd.read_csv(os.path.join(TEA_DIR, filename), header=0, index_col='State')


def elec_rate_state(G: nx.DiGraph, emissions=False, clean_elec_prem_dolkwh: float = None):
    # return electricity rate for each node based on state rate in [$/MWh] or in [gC02/kWh] if <emissions>=True
    # NUFRIEND
    filename = 'average_electricity_price_state.csv'
    # SF Paper
    print('SF')
    filename = 'eia_average_electricity_price_state_2020.csv'

    if clean_elec_prem_dolkwh is None:
        clean_elec_prem_dolkwh = 0

    if emissions:
        # return emissions by state
        df_state = load_lca_battery_lookup()
        elec_rate_dict = dict()
        for n in G:
            elec_rate_dict[n] = df_state.loc[G.nodes[n]['state'], 'emissions']  # g/kWh
    else:
        # return price by state
        df_state = pd.read_csv(os.path.join(TEA_DIR, filename), header=0, index_col='State')
        elec_rate_dict = dict()
        for n in G:
            # add in clean electricity premium
            elec_rate_dict[n] = 10 * (df_state.loc[G.nodes[n]['state'], 'Commercial'] +
                                      clean_elec_prem_dolkwh * 100)  # 10 * ¢/kWh == $/MWh

    return elec_rate_dict


'''
waybill_data_processing.py
'''


def mmddyyyy_to_datetime(mmddyyyy: str):
    if isinstance(mmddyyyy, date):
        return mmddyyyy

    if isinstance(mmddyyyy, int):
        mmddyyyy = str(mmddyyyy)
        if len(mmddyyyy) == 7:
            mmddyyyy = '0' + mmddyyyy

    y = int(mmddyyyy[-4:])
    m = int(mmddyyyy[:2])
    d = int(mmddyyyy[2:4])

    return date(y, m, d)


def datetime_to_mmddyyyy(dt):
    if isinstance(dt, str):
        return dt
    dt = str(dt.date())  # datetime objects are 'yyyy-mm-dd'
    # mm + dd + yyyy
    return dt[-5:-3] + dt[-2:] + dt[:4]
