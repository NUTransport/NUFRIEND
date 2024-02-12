from util import *
from network_representation import gdfs_from_graph, digraph_from_gdfs, load_simplified_consolidated_graph

'''
INPUT
'''


def load_scenario_df(scenario_code: str):

    filename = scenario_code + '.csv'
    df_scenario = pd.read_csv(os.path.join(SCENARIO_DIR, filename), header=0, index_col='Keyword')['Value']
    # convert str values to float where possible
    df1 = pd.to_numeric(df_scenario, errors='coerce')
    idxs = list(set(df1.dropna().index).difference({'time_window_start', 'time_window_end'}))
    df_scenario.loc[idxs] = df1.loc[idxs]

    return df_scenario


def write_scenario_df(rr: str = None, fuel_type: str = None, deployment_perc: float = None,
                      D: float = None, reroute: bool = None, switch_tech: bool = None, max_reroute_inc: float = None,
                      max_util: float = None, station_type: str = None,
                      clean_energy: bool = None, clean_energy_cost: float = None, emissions_obj: bool = None,
                      CCWS_filename: str = None, perc_ods: float = None, comm_group: str = None,
                      time_window: tuple = None, freq: str = None,
                      eff_energy_p_tender: float = None,
                      suppress_output: bool = None, binary_prog: bool = None,
                      radius: float = None, intertypes: set = None, deployment_table: bool = False):

    if not deployment_table:
        perc_ods = None

    # verify validity of inputs
    # rr
    assert rr in {'BNSF', 'UP', 'NS', 'CSXT', 'KCS', 'CN', 'CP', 'USA1', 'WCAN', 'EAST'} \
           or rr is not None, 'Provide a railroad selection.'
    # fuel_type
    assert fuel_type in {'battery', 'hydrogen', 'diesel', 'e-fuel', 'biodiesel'} or 'hybrid' in fuel_type \
           or fuel_type is not None, 'Provide an energy source selection.'
    # deployment_perc
    assert isinstance(deployment_perc, float) or isinstance(deployment_perc, int) or deployment_perc is None, \
        'Provide a valid input for deployment percentage.'
    if deployment_perc is None:
        deployment_perc = 0
    else:
        assert 0 <= deployment_perc <= 1, 'Deployment percentage must be between zero and one, if provided.'
    # reroute
    assert isinstance(reroute, bool) or reroute == 1 or reroute == 0 or reroute is None, \
        'Provide a valid input for the routing parameter.'
    if reroute is None:
        reroute = True
    reroute = int(reroute)
    # switch_tech
    assert isinstance(switch_tech, bool) or switch_tech in {0, 1} or switch_tech is None, \
        'Provide a valid input for the technology switching parameter.'
    if switch_tech is None:
        switch_tech = False
    switch_tech = int(switch_tech)
    # max_reroute_inc
    assert isinstance(max_reroute_inc, float) or isinstance(max_reroute_inc, int) or max_reroute_inc is None, \
        'Provide a valid input for the maximum rerouting distance increase parameter.'
    if max_reroute_inc is None:
        max_reroute_inc = 0.5
    # for battery case
    if fuel_type == 'battery':
        # max_util
        assert isinstance(max_util, float) or isinstance(max_util, int) or max_util is None, \
            'Provide a valid input for the maximum station utilization.'
        if max_util is None:
            max_util = 0.88
        else:
            assert 0 <= max_util <= 1, 'Maximum station utilization must be a <float> between zero and one.'
            util_rates = np.array([0.15, 0.29, 0.44, 0.58, 0.73, 0.88])
            if max_util <= min(util_rates):
                max_util = min(util_rates)
            else:
                max_util = util_rates[max(np.where(util_rates <= max_util)[0])]
        # D (range)
        assert isinstance(D, float) or isinstance(D, int) or D is None, 'Provide a valid input for energy source range.'
        if D is None:
            D = 1.6 * 400
        # station_type
        # TODO: replace with different station_types once the TEA results are plugged in
        assert station_type == '3MW' or station_type is None, \
            'A station type of ' + station_type + ' is not an adequate station type for the energy source selected.'
        if station_type is None:
            station_type = '3MW'
        # clean_energy_cost
        assert isinstance(clean_energy_cost, float) or isinstance(clean_energy_cost, int) or \
               clean_energy_cost is None, 'Provide a valid input for the cost of clean energy.'
        if clean_energy_cost is None:
            clean_energy_cost = 0.0
        # eff_energy_p_tender
        assert isinstance(eff_energy_p_tender, float) or isinstance(eff_energy_p_tender, int) or \
               eff_energy_p_tender is None, \
            'Provide a valid input for the effective energy storage per tender car.'
        if eff_energy_p_tender is None:
            eff_energy_p_tender = 10000     # kWh/tender car
    # for hydrogen case
    elif fuel_type == 'hydrogen':
        # max_util
        assert isinstance(max_util, float) or isinstance(max_util, int) or max_util is None, \
            'Provide a valid input for the maximum station utilization.'
        if max_util is None:
            max_util = 0.88
        else:
            assert 0 <= max_util <= 1, 'Maximum station utilization must be a <float> between zero and one.'
            util_rates = np.array([0.25, 0.5, 0.83])
            if max_util <= min(util_rates):
                max_util = min(util_rates)
            else:
                max_util = util_rates[max(np.where(util_rates <= max_util)[0])]
        # station_type
        assert station_type in {'Cryo-pump', '700 via pump', '350 via pump'} or station_type is None, \
            'A station type of ' + station_type + ' is not an adequate station type for the energy source selected.'
        if station_type is None:
            station_type = 'Cryo-pump'
        # clean_energy_cost
        assert isinstance(clean_energy_cost, float) or isinstance(clean_energy_cost, int) \
               or clean_energy_cost is None, 'Provide a valid input for the cost of clean energy.'
        if clean_energy_cost is None:
            clean_energy_cost = 0.0
        # eff_energy_p_tender
        assert isinstance(eff_energy_p_tender, float) or isinstance(eff_energy_p_tender, int) \
               or eff_energy_p_tender is None, \
            'Provide a valid input for the effective energy storage per tender car.'
        if eff_energy_p_tender is None:
            eff_energy_p_tender = 4000  # kgh2/tender car
        # D (range) is calculated based on <eff_energy_p_tender>
        # hardcoded based on range_calcs.xls, uses info on avg. locomotive tonnage and diesel energy consumption
        D = 1047 * (1 / KM2MI)
    # for hybrid case
    elif 'hybrid' in fuel_type:
        # max_util
        assert isinstance(max_util, float) or isinstance(max_util, int) or max_util is None, \
            'Provide a valid input for the maximum station utilization.'
        if max_util is None:
            max_util = 0.88
        else:
            assert 0 <= max_util <= 1, 'Maximum station utilization must be a <float> between zero and one.'
            util_rates = np.array([0.15, 0.29, 0.44, 0.58, 0.73, 0.88])
            if max_util <= min(util_rates):
                max_util = min(util_rates)
            else:
                max_util = util_rates[max(np.where(util_rates <= max_util)[0])]
        # D (range)
        assert isinstance(D, float) or isinstance(D, int) or D is None, 'Provide a valid input for energy source range.'
        if D is None:
            D = 1.6 * 400
        # station_type
        assert station_type == '3MW' or station_type is None, \
            'A station type of ' + station_type + ' is not an adequate station type for the energy source selected.'
        if station_type is None:
            station_type = '3MW'
        # clean_energy_cost
        assert isinstance(clean_energy_cost, float) or isinstance(clean_energy_cost, int) or \
               clean_energy_cost is None, 'Provide a valid input for the cost of clean energy.'
        if clean_energy_cost is None:
            clean_energy_cost = 0.0
        # eff_energy_p_tender
        assert isinstance(eff_energy_p_tender, float) or isinstance(eff_energy_p_tender, int)\
               or eff_energy_p_tender is None, \
            'Provide a valid input for the effective energy storage per tender car.'
        if eff_energy_p_tender is None:
            eff_energy_p_tender = 10000  # kWh/tender car
    else:
        # max_util
        assert isinstance(max_util, float) or isinstance(max_util, int) or max_util is None, \
            'Provide a valid input for the maximum station utilization.'
        max_util = 'XXX'
        # D (range); does not matter here
        D = 0
        # station_type; does not matter here
        station_type = 'X'
        # clean_energy_cost; does not matter here
        clean_energy_cost = 0
        # eff_energy_p_tender; does not matter here
        eff_energy_p_tender = 0
    # CCWS_filename
    assert CCWS_filename in {'WB2019_913_Unmasked.csv', 'WB2018_900_Unmasked.csv', 'WB2017_900_Unmasked.csv'} \
           or CCWS_filename is None
    if CCWS_filename is None:
        CCWS_filename = 'WB2019_913_Unmasked.csv'
    # time_window
    assert isinstance(time_window, tuple) or time_window is None, 'Provide a valid time window.'
    if time_window is None:
        time_window_start = '01012019'
        time_window_end = '12312019'
    else:
        time_window_start, time_window_end = time_window
        if len(time_window_start) < 8:
            time_window_start = '0' + time_window_start
        assert len(time_window_start) == 8, 'Provide a valid time window start.'
        if len(time_window_end) < 8:
            time_window_end = '0' + time_window_end
        assert len(time_window_end) == 8, 'Provide a valid time window end.'
    # freq
    assert freq in {'M', 'Q', 'W', '2W', 'D'} or freq is None, 'Provide a valid time window frequency.'
    if freq is None:
        freq = 'M'
    # clean_energy
    assert isinstance(clean_energy, bool) or clean_energy in {0, 1} or clean_energy is None, \
        'Provide a valid parameter for the selection of clean energy.'
    if clean_energy is None:
        clean_energy = False
    clean_energy = int(clean_energy)
    # emissions_obj
    assert isinstance(emissions_obj, bool) or emissions_obj in {0, 1} or emissions_obj is None, \
        'Provide a valid parameter for the selection of the energy sourcing objective.'
    if emissions_obj is None:
        emissions_obj = False
    emissions_obj = int(emissions_obj)
    # perc_ods
    assert (isinstance(perc_ods, float) and 0 <= perc_ods <= 1) or perc_ods is None, \
        'Provide a valid input for percentage of ODs to select.'
    if perc_ods is None:
        perc_ods = 'X'
    # comm_group
    assert (isinstance(comm_group, str) and comm_group in
            {'OTHER', 'TOTAL', 'MO_VEH', 'AG_FOOD', 'IM', 'COAL', 'CHEM_PET', 'NONMET_PRO', 'FOR_PROD', 'MET_ORE'}) \
           or comm_group is None, 'Provide a valid commodity group.'
    if comm_group is None:
        comm_group = 'TOTAL'
    # suppress_output
    assert isinstance(suppress_output, bool) or suppress_output in {0, 1} or suppress_output is None
    if suppress_output is None:
        suppress_output = True
    suppress_output = int(suppress_output)
    # binary_prog
    assert isinstance(binary_prog, bool) or binary_prog in {0, 1} or binary_prog is None
    if binary_prog is None:
        binary_prog = True
    binary_prog = int(binary_prog)
    # radius
    assert isinstance(radius, float) or radius is None
    if radius is None:
        radius = 10000
    # intertypes
    assert isinstance(intertypes, set) or intertypes is None
    if intertypes is None:
        intertypes = {'T', 'P', 'M', 'D'}

    idxs = ['rr', 'fuel_type', 'deployment_perc',
            'D', 'reroute', 'switch_tech', 'max_reroute_inc',
            'max_util', 'station_type',
            'clean_energy', 'clean_energy_cost', 'emissions_obj',
            'CCWS_filename', 'perc_ods', 'comm_group',
            'time_window_start', 'time_window_end', 'freq',
            'eff_energy_p_tender',
            'suppress_output', 'binary_prog',
            'radius', 'intertypes']

    df_scenario = pd.Series(index=idxs,
                            data=[rr, fuel_type, deployment_perc,
                                  D, reroute, switch_tech, max_reroute_inc,
                                  max_util, station_type,
                                  clean_energy, clean_energy_cost, emissions_obj,
                                  CCWS_filename, perc_ods, comm_group,
                                  time_window_start, time_window_end, freq,
                                  eff_energy_p_tender,
                                  suppress_output, binary_prog,
                                  radius, intertypes],
                            name='Value')

    # df_scenario = df_scenario.to_frame()
    # df_scenario['Keyword'] = df_scenario.index
    scenario_code = codify_scenario_output_file(df_scenario=df_scenario, deployment_table=deployment_table)

    df_scenario = df_scenario.to_frame()
    df_scenario['Keyword'] = df_scenario.index
    df_scenario = pd.concat([df_scenario,
                             pd.DataFrame(data=[['scenario_code', scenario_code]],
                                          columns=['Keyword', 'Value'])])
    # df_scenario = df_scenario.append(pd.DataFrame(data=[['scenario_code', scenario_code]],
    #                                               columns=['Keyword', 'Value']))
    df_scenario[['Keyword', 'Value']].to_csv(os.path.join(SCENARIO_DIR, scenario_code + '.csv'), index=False)

    return df_scenario.set_index('Keyword')['Value']


def load_scenario_codification_legend():
    return pd.read_csv(os.path.join(SCENARIO_DIR, 'scenario_codification_legend.csv'), header=0,
                       index_col=['Keyword', 'Value'])


def load_dict_from_json(filepath: str):
    # store dict <d> as json file in filepath
    with open(filepath, 'r') as fr:
        return json.load(fr)


'''
OUTPUT
'''


def get_val_from_code(scenario_code: str, key: str):
    df_legend = load_scenario_codification_legend()
    df_key = df_legend.loc[(key, slice(None))].reset_index()

    if df_key['Value'].values[0] == '0':
        start = int(df_key['Start'].values[0])
        end = int(df_key['Start'].values[0] + df_key['Length'].values[0])

        return scenario_code[start: end]
    else:
        idx = int(df_key['Order'].values[0])
        df_key.index = df_key['Code']

        return df_key.loc[int(scenario_code[idx]), 'Value']


def codify_scenario_output_file(df_scenario, deployment_table=False):

    if isinstance(df_scenario, pd.DataFrame):
        df_scenario = df_scenario['Value'].squeeze()

    # pd.to_numeric(df_scenario, errors='ignore')
    # print(df_scenario)

    df_legend = load_scenario_codification_legend()
    kwds = [i[0] for i in df_legend.index if i[0] in {'rr', 'fuel_type', 'station_type', 'CCWS_filename', 'freq'}]

    order_val_dict = {int(df_legend.loc[(k, df_scenario[k]), 'Order']):
                          str(int(df_legend.loc[(k, df_scenario[k]), 'Code'])) for k in kwds}

    # kwds = [i[0] for i in df_legend.index if i[0] in {'reroute', 'switch_tech', 'clean_energy', 'emissions_obj'}]
    kwds = [i[0] for i in df_legend.index if i[0] in {'reroute', 'switch_tech', 'emissions_obj'}]

    order_val_dict.update({int(df_legend.loc[(k, '0'), 'Order']): str(int(df_scenario[k])) for k in kwds})

    if deployment_table:
        order_val_dict[df_legend.loc[('deployment_perc', '0'), 'Order']] = 'XXX'
    else:
        if df_scenario['deployment_perc'] == 'XXX':
            order_val_dict[df_legend.loc[('deployment_perc', '0'), 'Order']] = 'XXX'
        elif df_scenario['deployment_perc'] < 0.1:
            order_val_dict[df_legend.loc[('deployment_perc', '0'), 'Order']] = \
                str(int(100 * df_scenario['deployment_perc'])) + 'XX'
        elif df_scenario['deployment_perc'] < 1:
            order_val_dict[df_legend.loc[('deployment_perc', '0'), 'Order']] = \
                str(int(100 * df_scenario['deployment_perc'])) + 'X'
        else:
            order_val_dict[df_legend.loc[('deployment_perc', '0'), 'Order']] = '100'
    if df_scenario['max_reroute_inc'] < 0.1:
        order_val_dict[df_legend.loc[('max_reroute_inc', '0'), 'Order']] = \
            str(int(100 * df_scenario['max_reroute_inc'])) + 'XX'
    elif df_scenario['max_reroute_inc'] < 1:
        order_val_dict[df_legend.loc[('max_reroute_inc', '0'), 'Order']] = \
            str(int(100 * df_scenario['max_reroute_inc'])) + 'X'
    else:
        order_val_dict[df_legend.loc[('max_reroute_inc', '0'), 'Order']] = '100'

    # max_util
    if isinstance(df_scenario['max_util'], str) and df_scenario['max_util'] == 'XXX':
        order_val_dict[df_legend.loc[('max_util', '0'), 'Order']] = 'XXX'
    elif df_scenario['max_util'] < 0.1:
        order_val_dict[df_legend.loc[('max_util', '0'), 'Order']] = \
            str(int(100 * df_scenario['max_util'])) + 'XX'
    elif df_scenario['max_util'] < 1:
        order_val_dict[df_legend.loc[('max_util', '0'), 'Order']] = \
            str(int(100 * df_scenario['max_util'])) + 'X'
    else:
        order_val_dict[df_legend.loc[('max_util', '0'), 'Order']] = '100'

    # clean_energy_cost
    # if df_scenario['clean_energy_cost'] < 0.1:
    #     order_val_dict[df_legend.loc[('clean_energy_cost', '0'), 'Order']] = \
    #         str(int(100 * df_scenario['clean_energy_cost'])) + 'XX'
    # elif df_scenario['clean_energy_cost'] < 1:
    #     order_val_dict[df_legend.loc[('clean_energy_cost', '0'), 'Order']] = \
    #         str(int(100 * df_scenario['clean_energy_cost'])) + 'X'
    # elif df_scenario['clean_energy_cost'] < 10:
    #     order_val_dict[df_legend.loc[('clean_energy_cost', '0'), 'Order']] = \
    #         str(int(100 * df_scenario['clean_energy_cost']))

    # eff_energy_p_tender
    # order_val_dict[df_legend.loc[('eff_energy_p_tender', '0'), 'Order']] = \
    #     str(int(df_scenario['eff_energy_p_tender'])) + 'XXXXX'[len(str(int(df_scenario['eff_energy_p_tender']))):]

    # D
    order_val_dict[df_legend.loc[('D', '0'), 'Order']] = \
        str(int(df_scenario['D'])) + 'XXXX'[len(str(int(df_scenario['D']))):]

    # time_window
    order_val_dict[df_legend.loc[('time_window_start', '0'), 'Order']] = df_scenario['time_window_start'][:4]
    order_val_dict[df_legend.loc[('time_window_end', '0'), 'Order']] = df_scenario['time_window_end'][:4]

    keys = list(order_val_dict.keys())
    keys.sort()

    codified_name = ''
    for k in keys:
        codified_name += order_val_dict[k]

    return codified_name


def dict_to_json(d: dict, filepath: str):
    # store dict <d> as json file in filepath
    with open(filepath, 'w') as fw:
        json.dump(d, fw)


'''
CACHE GRAPH
'''


def cache_metrics(G: nx.DiGraph, scenario_code: str):
    metrics = G.graph.copy()
    metrics['crs'] = str(G.graph['crs'])  # replace 'crs' val from crs type to str as crs is not JSON serializable

    filepath = os.path.join(OUTPUT_DIR, 'Metrics', scenario_code + '.json')
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

    if os.path.exists(filepath_nodes):
        # load railroad geometry data
        gdf_nodes, gdf_edges = gdfs_from_graph(load_simplified_consolidated_graph(
            get_val_from_code(scenario_code=scenario_code, key='rr')))
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


'''
CACHE PLOT
'''


def cache_plot(fig: plotly.graph_objs.Figure, scenario_code: str):
    with open(os.path.join(FIG_DIR, scenario_code + '.json'), 'w') as figf:
        figf.write(fig.to_json())
    figf.close()


def load_scenario_plot(scenario_code: str):
    filepath = os.path.join(FIG_DIR, scenario_code + '.json')
    if os.path.exists(filepath):
        return plotly.io.read_json(filepath)
    else:
        return None


def plot_cached_plot(scenario_code: str):
    fig = load_scenario_plot(scenario_code)

    if fig is not None:
        fig.show()

    return fig
