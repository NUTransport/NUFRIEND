from util import *
# MODULES
from network_representation import gdfs_from_graph, load_simplified_consolidated_graph

purple, mid_purple, light_purple, green1, green2, green3, green4, green5, red, light_red, black = ['#512D88', '#DAD0E6',
                                                                                                   '#ECE6F4', '#9AD470',
                                                                                                   '#75D431', '#719B52',
                                                                                                   '#4A871F', '#2E5413',
                                                                                                   '#FF3033', '#FF787A',
                                                                                                   '#18141C']
# c = [purple, green1, mid_purple, green2, green4, green5, light_red, red, 'blue']
c = plotly.colors.qualitative.G10


def plot_battery_facility_location(G, time_horizon, crs='WGS84', additional_plots=False, nested=True, max_flow=False,
                                   colors=False, time_step_label=False, title=None, fig=None):
    if fig is None:
        if additional_plots:
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "table"}, {"type": "scattergeo"}]],
                # rows=2, cols=2,
                # specs=[[{"type": "table", 'rowspan': 2}, {"type": "scattergeo", "rowspan": 2}],
                #        [None, None]],
                column_widths=[0.2, 0.8],
                row_heights=[1],
                horizontal_spacing=0.1,
                # vertical_spacing=0.1,
                subplot_titles=['Scenario', title if title else 'Network']
            )
        else:
            t0 = time.time()
            fig = base_plot(G.graph['railroad'])
            print('\t LOAD BASE PLOT:: ' + str(time.time() - t0))

    # fig = plot_states_bg()

    # G = project_graph(G.copy(), to_crs=crs)
    t0 = time.time()
    # H = selected_subgraph(G=G, time_step=time_step)
    nodes_gdf, edges_gdf = gdfs_from_graph(G, crs=crs, smooth_geometry=False)
    print('GDF EXTRACTION:: ' + str(time.time() - t0))

    if additional_plots:
        # Trace 0: baseline edges
        fig.add_trace(plot_base_edges(edges_gdf), row=1, col=2)
        # Trace 1: edges trace for first time_step
        fig.add_trace(plot_edges(edges_gdf, time_idx=0, time_horizon=time_horizon), row=1, col=2)
        # Trace 2: nodes - facility trace for first time_step
        fig.add_trace(plot_facility_nodes(nodes_gdf, time_idx=0, time_horizon=time_horizon), row=1, col=2)
        # Trace 3: nodes - covered trace for first time_step
        fig.add_trace(plot_covered_nodes(nodes_gdf, time_idx=0, time_horizon=time_horizon), row=1, col=2)
        # Trace 4: scenario summary table
        fig.add_trace(plot_facility_location_summary_table(G=G, time_step=time_horizon[0], max_flow=max_flow),
                      row=1, col=1)

        frames = [go.Frame(data=[plot_edges(edges_gdf=edges_gdf, time_idx=ti, time_horizon=time_horizon,
                                            color=c[ti] if colors else None),
                                 plot_facility_nodes(nodes_gdf=nodes_gdf, time_idx=ti, time_horizon=time_horizon,
                                                     color=c[ti] if colors else None),
                                 plot_covered_nodes(nodes_gdf=nodes_gdf, time_idx=ti, time_horizon=time_horizon,
                                                    color=c[ti] if colors else None),
                                 plot_facility_location_summary_table(G=G, time_step=ts, max_flow=max_flow)],
                           traces=[1, 2, 3, 4], name=f'Time Step {ts}' if time_step_label else f'{ts}')
                  for ti, ts in enumerate(time_horizon)]

        # # START: Color Attempt
        # fig.add_trace(plot_base_edges(edges_gdf), row=1, col=2)

        # for ti, ts in enumerate(time_horizon):
        #     # Trace 3 * ti + 1: edges trace for first time_step
        #     fig.add_trace(plot_edges(edges_gdf, time_idx=ti, time_horizon=time_horizon, color=c[ti]), row=1, col=2)
        #     # Trace 3 * ti + 2: nodes - facility trace for first time_step
        #     fig.add_trace(plot_facility_nodes(nodes_gdf, time_idx=ti, time_horizon=time_horizon, color=c[ti]),
        #                   row=1, col=2)
        #     # Trace 3 * ti + 3: nodes - covered trace for first time_step
        #     fig.add_trace(plot_covered_nodes(nodes_gdf, time_idx=ti, time_horizon=time_horizon, color=c[ti]),
        #                   row=1, col=2)
        #
        # # Trace 3 * len(time_horizon) + 1: scenario summary table
        # fig.add_trace(plot_facility_location_summary_table(G=G, time_step=time_horizon[0], max_flow=max_flow),
        #               row=1, col=1)
        #
        # frames = [go.Frame(data=[plot_edges(edges_gdf=edges_gdf, time_idx=ki, time_horizon=time_horizon,
        #                                     color=c[ki] if colors else None) for ki in range(ti + 1)] +
        #                         [go.Scattergeo(None) for _ in range(ti + 1, len(time_horizon))] +
        #                         [plot_facility_nodes(nodes_gdf=nodes_gdf, time_idx=ki, time_horizon=time_horizon,
        #                                              color=c[ki] if colors else None) for ki in range(ti + 1)] +
        #                         [go.Scattergeo(None) for _ in range(ti + 1, len(time_horizon))] +
        #                         [plot_covered_nodes(nodes_gdf=nodes_gdf, time_idx=ki, time_horizon=time_horizon,
        #                                             color=c[ki] if colors else None) for ki in range(ti + 1)] +
        #                         [go.Scattergeo(None) for _ in range(ti + 1, len(time_horizon))] +
        #                         [plot_facility_location_summary_table(G=G, time_step=ts, max_flow=max_flow)],
        #                    traces=[3 * i + 1 for i in range(len(time_horizon))] +
        #                           [3 * i + 2 for i in range(len(time_horizon))] +
        #                           [3 * i + 3 for i in range(len(time_horizon))] +
        #                           [3 * len(time_horizon) + 1],
        #                    name=f'Time Step {ts}' if time_step_label else f'{ts}')
        #           for ti, ts in enumerate(time_horizon)]

        # frames = [go.Frame(data=[plot_edges(edges_gdf=edges_gdf, time_idx=ti, time_horizon=time_horizon,
        #                                     color=c[ti] if colors else None),
        #                          plot_facility_nodes(nodes_gdf=nodes_gdf, time_idx=ti, time_horizon=time_horizon,
        #                                              color=c[ti] if colors else None),
        #                          plot_covered_nodes(nodes_gdf=nodes_gdf, time_idx=ti, time_horizon=time_horizon,
        #                                             color=c[ti] if colors else None),
        #                          plot_facility_location_summary_table(G=G, time_step=ts, max_flow=max_flow)],
        #                    traces=[1, 2, 3, 4], name=f'Time Step {ts}' if time_step_label else f'{ts}')
        #           for ti, ts in enumerate(time_horizon)]
        # # END: Color Attempt

        fig.update_geos(projection_type="albers usa")
        fig.update(frames=frames)
    else:
        # edges trace for first time_step
        fig.add_trace(plot_edges(edges_gdf, time_idx=0, time_horizon=time_horizon), row=1, col=1)
        # nodes - facility trace for first time_step
        fig.add_trace(plot_facility_nodes(nodes_gdf, time_idx=0, time_horizon=time_horizon), row=1, col=1)
        # nodes - covered trace for first time_step
        fig.add_trace(plot_covered_nodes(nodes_gdf, time_idx=0, time_horizon=time_horizon), row=1, col=1)

        frames = [go.Frame(data=[plot_edges(edges_gdf=edges_gdf, time_idx=ti, time_horizon=time_horizon,
                                            color=c[ti] if colors else None),
                                 plot_facility_nodes(nodes_gdf=nodes_gdf, time_idx=ti, time_horizon=time_horizon,
                                                     color=c[ti] if colors else None),
                                 plot_covered_nodes(nodes_gdf=nodes_gdf, time_idx=ti, time_horizon=time_horizon,
                                                    color=c[ti] if colors else None)])
                  for ti, ts in enumerate(time_horizon)]

        fig.update_geos(projection_type="albers usa")
        fig.update(frames=frames)

    add_slider_animation(fig)

    # fig.show()
    iplot(fig)

    return fig


def plot_battery_facility_location_static(G, time_horizon, title=None, crs='WGS84', fig=None):

    if fig is None:
        t0 = time.time()
        fig = base_plot(G.graph['railroad'])
        print('\t LOAD BASE PLOT:: ' + str(time.time() - t0))

    t0 = time.time()
    # H = selected_subgraph(G=G, time_step=time_step)
    nodes_gdf, edges_gdf = gdfs_from_graph(G, crs=crs, smooth_geometry=False)
    print('GDF EXTRACTION:: ' + str(time.time() - t0))

    for ti, ts in enumerate(time_horizon):
        fig.add_trace(plot_edges(edges_gdf, time_idx=ti, time_horizon=time_horizon, color=c[ti], incremental=True))
        fig.add_trace(plot_covered_nodes(nodes_gdf, time_idx=ti, time_horizon=time_horizon, color=c[ti],
                                         incremental=True))

    for ti, ts in enumerate(time_horizon):
        fig.add_trace(plot_facility_nodes(nodes_gdf, time_idx=ti, time_horizon=time_horizon, color=c[ti],
                                          incremental=True))

    fig.update_geos(projection_type="albers usa")

    fig.update_layout(
        title=title,
        legend=dict(
            orientation='v',
            yanchor='top',
            xanchor='right',
            x=1
        )
    )
    # fig.show()

    # fig.write_image('/Users/adrianhz/Desktop/rollout_image.png', scale=4, width=1600, height=800)

    return fig


def plot_base_edges(edges_gdf):
    edges_gdf = edges_gdf.copy()

    legend_name = 'Baseline Network'
    lg_group = -1

    lats = []
    lons = []
    names = []
    for u, v in edges_gdf.index:
        x, y = edges_gdf.loc[(u, v), 'geometry'].xy
        lats = np.append(lats, y)
        lons = np.append(lons, x)
        name = '{v1} miles'.format(v1=round(edges_gdf.loc[(u, v), 'miles']))
        names = np.append(names, [name] * len(y))
        lats = np.append(lats, None)
        lons = np.append(lons, None)
        names = np.append(names, None)

    g = go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines',
        line=dict(
            width=1,
            color='black'),
        opacity=0.5,
        hoverinfo="text",
        hovertext=names,
        legendgroup=lg_group,
        name=legend_name,
        showlegend=True,
        connectgaps=False,
    )

    return g


def plot_edges(edges_gdf, time_idx, time_horizon, color=None, incremental=False):
    edges_gdf = edges_gdf.copy()

    if incremental:
        if time_idx == 0:
            edges_gdf.drop(index=[i for i in edges_gdf.index
                                  if not edges_gdf.loc[i, 'covered'][time_horizon[time_idx]]],
                           inplace=True)
        else:
            edges_gdf.drop(index=[i for i in edges_gdf.index if edges_gdf.loc[i, 'covered'][time_horizon[time_idx - 1]]
                                  or not edges_gdf.loc[i, 'covered'][time_horizon[time_idx]]],
                           inplace=True)
    else:
        edges_gdf.drop(index=[i for i in edges_gdf.index if not edges_gdf.loc[i, 'covered'][time_horizon[time_idx]]],
                       inplace=True)

    legend_name = 'Battery Network ' + time_horizon[time_idx]
    lg_group = 1

    lats = []
    lons = []
    names = []
    for u, v in edges_gdf.index:
        x, y = edges_gdf.loc[(u, v), 'geometry'].xy
        lats = np.append(lats, y)
        lons = np.append(lons, x)
        name = '{v1} miles'.format(v1=round(edges_gdf.loc[(u, v), 'miles']))
        names = np.append(names, [name] * len(y))
        lats = np.append(lats, None)
        lons = np.append(lons, None)
        names = np.append(names, None)

    g = go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines',
        line=dict(
            width=3,
            color=color if color else green4,
        ),
        opacity=1,
        hoverinfo="text",
        hovertext=names,
        legendgroup=time_idx,
        legendrank=time_idx,
        name=legend_name,
        showlegend=True,
        connectgaps=False,
    )

    return g


def plot_facility_nodes(nodes_gdf, time_idx, time_horizon, color=None, incremental=False):
    legend_name = 'Selected Facility'
    lg_group = 1

    t_step = time_horizon[time_idx]

    lats = []
    lons = []
    names = []
    for n in nodes_gdf.index:
        x, y = nodes_gdf.loc[n, 'geometry'].xy
        if incremental:
            if time_idx == 0:
                if nodes_gdf.loc[n, 'facility'][t_step]:
                    lats = np.append(lats, y)
                    lons = np.append(lons, x)
                    if 'avg' in nodes_gdf.columns:
                        name = '{v1}, {v2} <br> {v3} <br> {v4} MWh/yr'.format(v1=nodes_gdf.loc[n, 'city'],
                                                                              v2=nodes_gdf.loc[n, 'state'],
                                                                              v3=nodes_gdf.loc[n, 'nodeid'],
                                                                              v4=round(nodes_gdf.loc[n, 'avg'][t_step][
                                                                                           'daily_supply_mwh']))
                    else:
                        name = '{v1}, {v2} <br> {v3}'.format(v1=nodes_gdf.loc[n, 'city'],
                                                             v2=nodes_gdf.loc[n, 'state'],
                                                             v3=nodes_gdf.loc[n, 'nodeid'])

                    names = np.append(names, [name] * len(y))
            else:
                if not nodes_gdf.loc[n, 'facility'][time_horizon[time_idx - 1]] and \
                        nodes_gdf.loc[n, 'facility'][t_step]:
                    lats = np.append(lats, y)
                    lons = np.append(lons, x)
                    if 'avg' in nodes_gdf.columns:
                        name = '{v1}, {v2} <br> {v3} <br> {v4} MWh/yr'.format(v1=nodes_gdf.loc[n, 'city'],
                                                                              v2=nodes_gdf.loc[n, 'state'],
                                                                              v3=nodes_gdf.loc[n, 'nodeid'],
                                                                              v4=round(nodes_gdf.loc[n, 'avg'][t_step][
                                                                                           'daily_supply_mwh']))
                    else:
                        name = '{v1}, {v2} <br> {v3}'.format(v1=nodes_gdf.loc[n, 'city'],
                                                             v2=nodes_gdf.loc[n, 'state'],
                                                             v3=nodes_gdf.loc[n, 'nodeid'])

                    names = np.append(names, [name] * len(y))
        else:
            if nodes_gdf.loc[n, 'facility'][t_step]:
                lats = np.append(lats, y)
                lons = np.append(lons, x)
                if 'avg' in nodes_gdf.columns:
                    name = '{v1}, {v2} <br> {v3} <br> {v4} MWh/yr'.format(v1=nodes_gdf.loc[n, 'city'],
                                                                          v2=nodes_gdf.loc[n, 'state'],
                                                                          v3=nodes_gdf.loc[n, 'nodeid'],
                                                                          v4=round(nodes_gdf.loc[n, 'avg'][t_step][
                                                                                       'daily_supply_mwh']))
                else:
                    name = '{v1}, {v2} <br> {v3}'.format(v1=nodes_gdf.loc[n, 'city'],
                                                         v2=nodes_gdf.loc[n, 'state'],
                                                         v3=nodes_gdf.loc[n, 'nodeid'])

                names = np.append(names, [name] * len(y))

        # lats = np.append(lats, None)
        # lons = np.append(lons, None)
        # names = np.append(names, None)

    g = go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers',
        marker=dict(
            size=20,
            color=color if color else green4,
        ),
        opacity=0.8,
        hoverinfo="text",
        hovertext=names,
        legendgroup=time_idx,
        legendrank=time_idx,
        name=legend_name,
        showlegend=True,
        connectgaps=False,
    )

    return g


def plot_covered_nodes(nodes_gdf, time_idx, time_horizon, color=None, incremental=False):
    legend_name = 'Covered Node'
    lg_group = 1

    lats = []
    lons = []
    names = []
    for n in nodes_gdf.index:
        x, y = nodes_gdf.loc[n, 'geometry'].xy
        if incremental:
            if time_idx == 0:
                if not nodes_gdf.loc[n, 'facility'][time_horizon[time_idx]] and \
                        nodes_gdf.loc[n, 'covered'][time_horizon[time_idx]]:
                    lats = np.append(lats, y)
                    lons = np.append(lons, x)
                    name = '{v1}, {v2}'.format(v1=nodes_gdf.loc[n, 'city'], v2=nodes_gdf.loc[n, 'state'])
                    names = np.append(names, [name] * len(y))
            else:
                if not nodes_gdf.loc[n, 'covered'][time_horizon[time_idx - 1]] and \
                        not nodes_gdf.loc[n, 'facility'][time_horizon[time_idx - 1]] and \
                        not nodes_gdf.loc[n, 'facility'][time_horizon[time_idx]] and \
                        nodes_gdf.loc[n, 'covered'][time_horizon[time_idx]]:
                    lats = np.append(lats, y)
                    lons = np.append(lons, x)
                    name = '{v1}, {v2}'.format(v1=nodes_gdf.loc[n, 'city'], v2=nodes_gdf.loc[n, 'state'])
                    names = np.append(names, [name] * len(y))
        else:
            if not nodes_gdf.loc[n, 'facility'][time_horizon[time_idx]] and \
                    nodes_gdf.loc[n, 'covered'][time_horizon[time_idx]]:
                lats = np.append(lats, y)
                lons = np.append(lons, x)
                name = '{v1}, {v2}'.format(v1=nodes_gdf.loc[n, 'city'], v2=nodes_gdf.loc[n, 'state'])
                names = np.append(names, [name] * len(y))

        # lats = np.append(lats, None)
        # lons = np.append(lons, None)
        # names = np.append(names, None)

    g = go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers',
        marker=dict(
            size=5,
            color=color if color else green4,
            symbol='square'
        ),
        opacity=1,
        hoverinfo="text",
        hovertext=names,
        legendgroup=time_idx,
        legendrank=time_idx,
        name=legend_name,
        showlegend=True,
        connectgaps=False,
    )

    return g


def plot_facility_location_summary_table(G: nx.DiGraph, time_step, max_flow=False):
    # plot table of summary of results for battery

    if max_flow:
        # g = go.Table(
        #     header=dict(values=['Variable', 'Value'],
        #                 font=dict(size=14),
        #                 line_color=black,
        #                 fill_color=mid_purple,
        #                 ),
        #     cells=dict(values=[
        #         ['Time Step',
        #          'Railroad',
        #          'Range (km)',
        #          'Number of Facilities',
        #          'Budget',
        #          'Max Flow Capture %',
        #          'Max Flow Capture',
        #          'Max Flow Available',
        #          'Discount Factor',
        #          'Number of Covered Nodes',
        #          'Number of Covered Edges'
        #          ],
        #         ['{v0}'.format(v0=time_step),
        #          '{v0}'.format(v0=G.graph['scenario']['railroad']),
        #          '{v0}'.format(v0=G.graph['scenario']['range_km']),
        #          '{v0}'.format(v0=len(G.graph['framework']['selected_facilities'][time_step])),
        #          '{v0}'.format(v0=G.graph['framework']['budgets'][time_step]),
        #          '{v0}%'.format(v0=round(100 * G.graph['framework']['tm_capt_perc'][time_step], 2)),
        #          '{v0} ton-miles'.format(v0=round(G.graph['framework']['tm_capt'][time_step])),
        #          '{v0} ton-miles'.format(v0=round(G.graph['framework']['tm_available'][time_step])),
        #          '{v0}'.format(v0=round(G.graph['framework']['discount_rates'][time_step], 4)),
        #          '{v0}'.format(v0=(len(set(n for p in G.graph['framework']['covered_path_nodes'][time_step].values()
        #                                    for n in p)))),
        #          '{v0}'.format(v0=len(set((u, v) for p in G.graph['framework']['covered_path_edges'][time_step].values()
        #                                   for u, v in p).union(
        #              set((v, u) for p in G.graph['framework']['covered_path_edges'][time_step].values() for u, v in
        #                  p))))
        #          ]
        #     ],
        #         font=dict(size=12),
        #         line_color=black,
        #         fill_color=light_purple,
        #     )
        # )
        g = go.Table(
            header=dict(values=['Variable', 'Value'],
                        font=dict(size=14),
                        line_color=black,
                        fill_color=mid_purple,
                        ),
            cells=dict(values=[
                ['Time Step',
                 'Cumulative Budget',
                 'Max Flow Capture',
                 'Emissions Reduction' if 'operations' in G.graph.keys() else '--',
                 'Levelized Cost<br>of Emissions',
                 'Cost of Avoided<br>Emissions',
                 ],
                ['{v0}'.format(v0=time_step),
                 '{v0}'.format(v0=G.graph['framework']['cum_budgets'][time_step]),
                 '{v0}%'.format(v0=round(100 * G.graph['framework']['tm_capt_perc'][time_step], 2)),
                 '{v0}%'.format(v0=round(G.graph['operations']['emissions_change'][time_step]['TOTAL'],
                                         2) if 'operations' in G.graph.keys() else '--'),
                 '¢{v0}/ton-mi'.format(v0=round(
                     100 * G.graph['energy_source_TEA'][time_step]['total_scenario_LCO_tonmi']['TOTAL'], 2)),
                 '${v0}/ton CO<sub>2</sub>'.format(v0=round(
                     1e3 * G.graph['operations']['cost_avoided_emissions'][time_step]['TOTAL'])
                 if 'operations' in G.graph.keys() else '--'),
                 ]
            ],
                font=dict(size=12),
                line_color=black,
                fill_color=light_purple,
            )
        )
    else:
        # g = go.Table(
        #     header=dict(values=['Variable', 'Value'],
        #                 font=dict(size=14),
        #                 line_color=black,
        #                 fill_color=mid_purple,
        #                 ),
        #     cells=dict(values=[
        #         ['Time Step',
        #          'Railroad',
        #          'Range (km)',
        #          'Number of Facilities',
        #          'Flow Threshold %',
        #          'Flow Threshold',
        #          'Flow Capture %',
        #          'Flow Capture',
        #          'Max Flow Available',
        #          'Discount Factor',
        #          'Number of Covered Nodes',
        #          'Number of Covered Edges'
        #          ],
        #         ['{v0}'.format(v0=time_step),
        #          '{v0}'.format(v0=G.graph['scenario']['railroad']),
        #          '{v0}'.format(v0=G.graph['scenario']['range_km']),
        #          '{v0}'.format(v0=len(G.graph['framework']['selected_facilities'][time_step])),
        #          '{v0}'.format(v0=round(100 * G.graph['framework']['flow_mins'][time_step] /
        #                                 G.graph['framework']['tm_available'][time_step], 2)),
        #          '{v0}'.format(v0=round(G.graph['framework']['flow_mins'][time_step])),
        #          '{v0}%'.format(v0=round(100 * G.graph['framework']['tm_capt_perc'][time_step], 2)),
        #          '{v0} ton-miles'.format(v0=round(G.graph['framework']['tm_capt'][time_step])),
        #          '{v0} ton-miles'.format(v0=round(G.graph['framework']['tm_available'][time_step])),
        #          '{v0}'.format(v0=round(G.graph['framework']['discount_rates'][time_step], 4)),
        #          '{v0}'.format(v0=(len(set(n for p in G.graph['framework']['covered_path_nodes'][time_step].values()
        #                                    for n in p)))),
        #          '{v0}'.format(v0=len(set((u, v) for p in G.graph['framework']['covered_path_edges'][time_step].values()
        #                                   for u, v in p).union(
        #              set((v, u) for p in G.graph['framework']['covered_path_edges'][time_step].values() for u, v in p))))
        #         ]
        #     ],
        #             font=dict(size=12),
        #             line_color=black,
        #             fill_color=light_purple,
        #         )
        #     )
        g = go.Table(
            header=dict(values=['Variable', 'Value'],
                        font=dict(size=14),
                        line_color=black,
                        fill_color=mid_purple,
                        ),
            cells=dict(values=[
                ['Time Step',
                 'Number of Facilities',
                 'Flow Threshold %',
                 'Flow Capture %',
                 'Emissions Reduction %',
                 'Levelized Cost of <br> Emissions (¢/ton-mi)',
                 ],
                ['{v0}'.format(v0=time_step),
                 '{v0}'.format(v0=len(G.graph['framework']['selected_facilities'][time_step])),
                 '{v0}'.format(v0=round(100 * G.graph['framework']['flow_mins'][time_step] /
                                        G.graph['framework']['tm_available'][time_step], 2)),
                 '{v0}%'.format(v0=round(100 * G.graph['framework']['tm_capt_perc'][time_step], 2)),
                 '{v0}%'.format(v0=round(G.graph['operations']['emissions_change'][time_step]['TOTAL'], 2)),
                 '{v0}%'.format(v0=round(
                     100 * G.graph['energy_source_TEA'][time_step]['total_scenario_LCO_tonmi']['TOTAL'], 2)),
                 ]
            ],
                font=dict(size=12),
                line_color=black,
                fill_color=light_purple,
            )
        )

    return g


def base_plot(rr: str, crs: str = 'WGS84'):
    filepath = os.path.join(FIG_DIR, rr + '_base_map.json')
    if os.path.exists(filepath):
        fig = plotly.io.read_json(filepath)
        return fig
    else:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "scattergeo"}]]
        )
        row = 1
        col = 1

        G = load_simplified_consolidated_graph(rr)
        nodes_gdf, edges_gdf = gdfs_from_graph(G, crs=crs, smooth_geometry=False)
        merged_edges = linemerge([shape(edges_gdf.loc[i, 'geometry']) for i in edges_gdf.index])

        legend_name = 'Diesel Network'
        legend_bool = True
        lg_group = 2
        for e in merged_edges:
            fig.add_trace(
                go.Scattergeo(
                    locationmode='USA-states',
                    lon=e.xy[0].tolist(),
                    lat=e.xy[1].tolist(),
                    hoverinfo='skip',
                    mode='lines',
                    line=dict(
                        width=1,
                        color=purple),
                    opacity=0.5,
                    legendgroup=lg_group,
                    name=legend_name,
                    showlegend=legend_bool
                ),
                row=row, col=col
            )
            legend_bool = False

        # update figure settings
        fig.update_geos(projection_type="albers usa")
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.2,
            ),
            # margin=dict(l=0, r=0, t=0, b=0),
        )

        # cache figure
        with open(filepath, 'w') as figf:
            figf.write(fig.to_json())
        figf.close()

    return fig


def line_size(volume: float) -> float:
    if volume < 1:
        ls = 0.2
    elif volume < 1000:
        ls = 0.5
    elif volume < 5000:
        ls = 1
    elif volume < 10000:
        ls = 2
    elif volume < 50000:
        ls = 3
    else:
        ls = 5
    return ls


def line_label_from_size(line_size: float) -> str:
    if line_size <= 0.2:
        return '0'
    elif line_size <= 0.5:
        return '1 - 1000'
    elif line_size <= 1:
        return '1000 - 5000'
    elif line_size <= 2:
        return '5000 - 10000'
    elif line_size <= 3:
        return '10000 - 50000'
    else:
        return '> 50000'


def line_color(volume: float) -> str:
    if volume < 1000:
        return 'blue'
    elif volume < 5000:
        return 'green'
    elif volume < 10000:
        return 'yellow'
    elif volume < 50000:
        return 'orange'
    return 'red'


def line_group(volume: float) -> str:
    if volume < 1000:
        return 1
    elif volume < 5000:
        return 2
    elif volume < 10000:
        return 3
    elif volume < 50000:
        return 4
    return 5


def line_label(volume: float) -> str:
    if volume < 1000:
        return '< 1000'
    elif volume < 5000:
        return '1000 - 5000'
    elif volume < 10000:
        return '5000 - 10000'
    elif volume < 50000:
        return '10000 - 50000'
    return '> 50000'


def add_slider_animation(fig):
    # duration between frames (ms)
    fr_duration = 500
    sliders = [
        {
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(fr_duration)],
                    "label": f'{f.name}',
                    "method": "animate",
                }
                for _, f in enumerate(fig.frames)
            ],
        }
    ]

    fig.update_layout(sliders=sliders,
                      updatemenus=[
                          {
                              "buttons": [
                                  {
                                      "args": [None, frame_args(fr_duration)],
                                      "label": "&#9654;",  # play symbol
                                      "method": "animate",
                                  },
                                  {
                                      "args": [[None], frame_args(fr_duration)],
                                      "label": "&#9724;",  # pause symbol
                                      "method": "animate",
                                  }],
                              "direction": "left",
                              "pad": {"r": 10, "t": 70},
                              "type": "buttons",
                              "x": 0.1,
                              "y": 0,
                          }])


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "redraw": False,
        "transition": {"duration": duration, "easing": "linear"},
    }
