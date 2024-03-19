import numpy as np

from util import *
# MODULES
from network_representation import project_graph, gdfs_from_graph, load_simplified_consolidated_graph
from helper import load_lca_battery_lookup, load_elec_cost_state_df

'''
TO RUN, USE THESE COMMANDS:
from alpha_framework import *
from GUI_trial import *
G = run_scenario('BNSF', radius=10000, D=250*1.6, freq='Y')     # only need to chance RR name and D for this...
f = plot_facility_nx_size(G)
'''

# global color assignment
# blue, red, orange, green, purple, teal, pink, olive, darkred, darkblue = plotly.colors.qualitative.G10
# '#8E77B2', '#F9F7FB'; mid purple, light purple
[purple, mid_purple, light_purple,
 green1, green2, green3, green4, green5,
 red, light_red, black] = ['#512D88', '#DAD0E6', '#ECE6F4',
                           '#9AD470', '#75D431', '#719B52', '#4A871F', '#2E5413',
                           '#FF3033', '#FF787A', '#18141C']

'''
MASTER PLOT
'''


def plot_scenario(G: nx.DiGraph, fuel_type: str, deployment_perc: float, comm_group: str = None,
                  fig=None, additional_plots=True, figlist=True, legend_show=True):
    if fuel_type == 'battery':
        fig = battery_plot(G, comm_group=comm_group, additional_plots=additional_plots, figlist=figlist,
                           fig=fig, legend_show=legend_show)
    elif fuel_type == 'hydrogen':
        fig = hydrogen_plot(G, comm_group=comm_group, additional_plots=additional_plots, figlist=figlist,
                            fig=fig, legend_show=legend_show)
    elif 'hybrid' in fuel_type:
        fig = hybrid_plot(G, comm_group=comm_group, additional_plots=additional_plots, figlist=figlist,
                          fig=fig, legend_show=legend_show)
    elif fuel_type == 'diesel' or fuel_type == 'biodiesel' or fuel_type == 'e-fuel':
        fig = dropin_plot(G, fuel_type=fuel_type, deployment_perc=deployment_perc, comm_group=comm_group,
                          additional_plots=additional_plots, figlist=figlist, fig=fig,
                          legend_show=legend_show)

    return fig


'''
BATTERY PLOT
'''


def battery_pie_operations_plot(G, comm_group: str, fig=None):
    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "domain"}]],
        )
        row = 1
        col = 1
    else:
        row = 3
        col = 1

    labels = ['Battery', 'Diesel']
    support_diesel_tonmi = G.graph['operations']['support_diesel_total_tonmi'][comm_group]
    battery_tonmi = G.graph['operations']['alt_tech_total_tonmi'][comm_group]
    # use the one below if it is desired to adjust the ton-miles to reflect the baseline ton-miles
    # battery_tonmi = ((1 - G.graph['operations']['perc_tonmi_inc'] / 100) *
    #                  G.graph['operations']['alt_tech_total_tonmi'][comm_group])
    values = [battery_tonmi * 365 / 1e6,
              support_diesel_tonmi * 365 / 1e6]
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=[green4, purple]),
            textinfo='label+percent',
            textposition='inside',
            hovertemplate='<b>%{label}</b> <br> %{value:.0f} [M ton-mi/yr]',
            name='',
            showlegend=False
        ),
        row=row, col=col
    )

    fig.update_layout(font_color=black,
                      autosize=True,
                      margin=dict(l=30, r=30, b=0, t=20, pad=1),
                      legend=dict(
                          orientation='h',
                          yanchor='bottom',
                          y=0,
                          xanchor='center',
                          x=0.5
                      ),
                      title=dict(
                          text=comm_group.capitalize() + ' Ton-Miles',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(t=15, b=5)
                      )
                      )

    return fig


def battery_tea_plot(G, comm_group: str, fig=None, legend_show=True):
    # compute aggregate statistics to plot and be able to compare between battery and diesel

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "bar"}]],
        )
        row = 1
        col = 1
    else:
        row = 2
        col = 2

    fig.add_trace(
        go.Bar(x=['Diesel', 'Battery'],
               y=[0,
                  G.graph['energy_source_TEA']['station_LCO_tonmi'][comm_group] * 100],
               name='Station',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green5)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Battery'],
               y=[0,
                  G.graph['energy_source_TEA']['om_LCO_tonmi'][comm_group] * 100],
               name='Station O&M',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green2)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Battery'],
               y=[0,
                  G.graph['energy_source_TEA']['battery_LCO_tonmi'][comm_group] * 100],
               name='Battery',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green3)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Battery'],
               y=[0,
                  G.graph['energy_source_TEA']['energy_LCO_tonmi'][comm_group] * 100],
               name='Electricity',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green1)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Battery'],
               y=[0,
                  G.graph['energy_source_TEA']['delay_LCO_tonmi'][comm_group] * 100],
               name='Delay',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=red)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Battery'],
               y=[G.graph['diesel_TEA']['fuel_LCO_tonmi'][comm_group] * 100,
                  0],
               name='Fuel',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=purple)
               ),
        row=row, col=col
    )

    # Scenario average
    fig.add_shape(type='line',
                  xref='paper', yref='y',
                  x0=-.5, y0=G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
                  x1=1.5, y1=G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
                  line=dict(color=black, width=2, dash='dash'),
                  opacity=1,
                  name='Scenario Average',
                  row=row, col=col
                  )
    fig.add_trace(
        go.Scatter(x=['Battery'],
                   y=[G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100],
                   mode='markers',
                   marker=dict(symbol='line-ew', size=6, line_width=2, opacity=1, color=black, line_color=black),
                   # opacity=1,
                   name='Scenario <br> Average',
                   hovertemplate='%{y:.2f} [¢/ton-mi]',
                   # showlegend=False
                   ),
        row=row, col=col
    )
    fig.update_layout(barmode='stack',
                      font=dict(color=black, size=12),
                      autosize=True,
                      margin=dict(l=20, r=0, b=0, t=50, pad=0),
                      showlegend=legend_show,
                      legend=dict(
                          orientation='h',
                          yanchor='top',
                          y=1,
                          xanchor='right',
                          x=1.9,
                          font=dict(size=10)
                      ),
                      title=dict(
                          text='Levelized Cost <br> of Operation',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(t=15, b=5)
                      )
                      )

    # fig.add_annotation(hovertext='Scenario Average: ' +
    #                              str(round(G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
    #                                        2)) + ' [¢/ton-mi]',
    #                    # text='Scenario Average',
    #                    text='',
    #                    xref='paper', yref='y',
    #                    x=1.5, y=G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
    #                    xanchor='left', yanchor='middle',
    #                    arrowcolor=black,
    #                    font=dict(color=black),
    #                    showarrow=False,
    #                    row=row, col=col
    #                    )

    # update yaxis properties
    fig.update_yaxes(title=dict(text='[¢ / ton-mi]', standoff=10,
                                font=dict(size=12)), showgrid=False, row=row, col=col)

    return fig


def battery_lca_plot(G, comm_group: str, fig=None, legend_show=True):
    # compute aggregate statistics to plot and be able to compare between battery and diesel

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "xy", 'secondary_y': True}]],
        )
        row = 1
        col = 1
    else:
        row = 2
        col = 1

    # ton CO2
    fig.add_trace(
        go.Bar(x=['100% Diesel', 'Scenario'],
               y=[G.graph['diesel_LCA']['annual_total_emissions_tonco2'][comm_group] / 1e3,
                  G.graph['energy_source_LCA']['annual_support_diesel_total_emissions'][comm_group] / 1e3],
               name='Diesel',
               hovertemplate='%{y:.0f} [kton CO<sub>2</sub>/yr]',
               # showlegend=False,
               marker=dict(color=purple)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['100% Diesel', 'Scenario'],
               y=[0,
                  G.graph['energy_source_LCA']['annual_battery_total_emissions'][comm_group] / 1e3],
               name='Electric Grid',
               hovertemplate='%{y:.0f} [kton CO<sub>2</sub>/yr]',
               # showlegend=False,
               marker=dict(color=green4)
               ),
        secondary_y=False,
        row=row, col=col
    )
    # ton CO2 /tonmile
    fig.add_trace(
        go.Scatter(x=['100% Diesel', 'Scenario'],
                   y=[1e6 * G.graph['diesel_LCA']['emissions_tonco2_tonmi'][comm_group],
                      1e6 * G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'][comm_group]],
                   mode='markers',
                   marker=dict(symbol='diamond', size=10, color=light_red),
                   name='WTW Emissions Rate',
                   hovertemplate='%{y:.2f} [g CO<sub>2</sub>/ton-mi]',
                   showlegend=False
                   ),
        secondary_y=True,
        row=row, col=col
    )

    fig.update_layout(barmode='stack',
                      font_color=black,
                      autosize=True,
                      margin=dict(l=0, r=0, b=0, t=50, pad=1),
                      showlegend=legend_show,
                      legend=dict(
                          orientation='h',
                          yanchor='top',
                          y=1.2,
                          xanchor='center',
                          x=0.5,
                          font=dict(size=10)),
                      title=dict(
                          text='WTW Emissions',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(b=30)
                      )
                      )

    # update yaxis properties
    fig.update_yaxes(title=dict(text='[kton CO<sub>2</sub>]', standoff=10,
                                font=dict(size=12)), secondary_y=False, showgrid=False,
                     row=row, col=col
                     )
    fig.update_yaxes(title=dict(text='[g CO<sub>2</sub> / ton-mi]', standoff=10,
                                font=dict(size=12, color=light_red)),
                     tickfont=dict(color=light_red), showgrid=False,
                     range=[0, np.ceil(1e6 * G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][comm_group] /
                                       10) * 10],
                     secondary_y=True,
                     row=row, col=col
                     )

    return fig


def battery_summary_table(G: nx.DiGraph, comm_group: str, fig=None):
    # plot table of summary of results for battery

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "table"}]]
        )
        row = 1
        col = 1
    else:
        row = 4
        col = 1

    des_tea = G.graph['energy_source_TEA']

    fig.add_trace(
        go.Table(
            header=dict(values=['Scenario Summary', 'Statistics'],
                        font=dict(size=14),
                        line_color=black,
                        fill_color=mid_purple,
                        ),
            cells=dict(values=[[
                'Cost of Avoided Emissions',
                '# of Charging Facilities',
                'Emissions Reduction' if G.graph['operations']['emissions_change'][comm_group] >= 0
                else 'Emissions Increase',
                'Average Route Distance Increase',
                '# of Chargers',
                'Average Charger Utilization',
                'Station Capital Cost',
                'Total Delay Cost',
                'Total Annual Cost',
                'Avg. Charge Time per Loc',
                'Avg. Queue Time',
                'Avg. Queue Length',
                'Peak Queue Time',
                'Peak Queue Length',
                'Avg. Daily Delay Cost per Car'
            ],
                [
                    str(round(G.graph['operations']['cost_avoided_emissions'][comm_group], 2)) +
                    ' [$/kg CO<sub>2</sub>]',
                    str(sum([G.nodes[n]['facility'] for n in G])) + ' out of ' + str(G.number_of_nodes()),
                    str(round(abs(G.graph['operations']['emissions_change'][comm_group]), 2)) + ' %',
                    str(round(G.graph['operations']['perc_mi_inc'][comm_group], 2)) + ' %',
                    des_tea['number_chargers'],
                    str(round(des_tea['actual_utilization'] * 24, 1)) + ' [hrs/day]',
                    str(round(des_tea['station_total'] / 1e6, 2)) + ' [$M]',
                    str(round(des_tea['total_annual_delay_cost'] / 1e6, 3)) + ' [$M]',
                    str(round(des_tea['annual_total_cost'][comm_group] / 1e6, 2)) + ' [$M]',
                    str(round(des_tea['charge_time'], 2)) + ' [hr]',
                    str(round(des_tea['avg_queue_time_p_loc'], 3)) + ' [hr]',
                    str(round(des_tea['avg_queue_length'], 3)) + ' [loc]',
                    str(round(des_tea['peak_queue_time_p_loc'], 3)) + ' [hr]',
                    str(round(des_tea['peak_queue_length'], 3)) + ' [loc]',
                    str(round(des_tea['avg_daily_delay_cost_p_car'], 2)) + ' [$]'
                ]],
                font=dict(size=12),
                line_color=black,
                fill_color=light_purple,
            )
        ),
        row=row, col=col
    )

    fig.update_layout(font_color=black,
                      autosize=True,
                      margin=dict(l=5, r=5, b=5, t=5, pad=1)
                      )

    return fig


def battery_cost_avoided_table(G, comm_group: str, fig=None):
    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "table"}]]
        )
        row = 1
        col = 1
    else:
        row = 3
        col = 2

    fig.add_trace(
        go.Table(
            header=dict(values=['Cost of Avoided Emissions'],
                        font=dict(size=14),
                        line_color=black,
                        fill_color=mid_purple,
                        ),
            cells=dict(values=[[str(round(G.graph['operations']['cost_avoided_emissions'][comm_group], 2)) +
                                ' [$/kg CO<sub>2</sub>]'
                                ]],
                       font=dict(size=12),
                       line_color=black,
                       fill_color=light_purple,
                       )
        ),
        row=row, col=col
    )

    fig.update_layout(font_color=black,
                      autosize=True,
                      # width=400,
                      # height=400,
                      margin=dict(l=5, r=5, b=5, t=5, pad=1)
                      )

    return fig


def battery_plot(G, comm_group: str, additional_plots=True, crs='WGS84', figlist=False, fig=None, legend_show=True):
    # fig.add_trace(go.Scattermapbox(lon=xs, lat=ys, mode='lines', hoverinfo="text", hovertext=caption))

    if fig is None:
        if additional_plots and not figlist:
            fig = make_subplots(
                rows=4, cols=3,
                specs=[[None, None, {"type": "scattergeo", "rowspan": 4}],
                       [{"type": "xy", 'secondary_y': True}, {"type": "bar"}, None],
                       [{"type": "domain"}, {"type": "table"}, None],
                       [{"type": "table", 'colspan': 2}, None, None]],
                column_widths=[0.1, 0.1, 0.8],
                row_heights=[0.02, 0.25, 0.25, 0.48],
                horizontal_spacing=0.1,
                subplot_titles=(None,
                                'WTW Emissions', 'Levelized Cost of Operation',
                                comm_group.capitalize() + ' Ton-Miles', None,
                                None)
            )
            row = 1
            col = 3
        else:
            t0 = time.time()
            fig = base_plot(G.graph['railroad'])
            print('\t LOAD BASE PLOT:: ' + str(time.time() - t0))
            row = 1
            col = 1

    # fig = plot_states_bg()

    # G = project_graph(G.copy(), to_crs=crs)
    t0 = time.time()
    nodes_gdf, edges_gdf = gdfs_from_graph(G, crs=crs, smooth_geometry=False)
    print('GDF EXTRACTION:: ' + str(time.time() - t0))

    t0 = time.time()
    # drop non-covered edges
    edges_gdf.drop(index=edges_gdf[edges_gdf['covered'] == 0].index, inplace=True)
    # keep only these cols
    agg_cols = {'miles': 'first', 'geometry': 'first',
                'battery_avg_ton': 'sum', 'battery_avg_loc': 'sum',
                'support_diesel_avg_ton': 'sum'}
    edges_gdf.drop(columns=set(edges_gdf.columns).difference(set(agg_cols.keys())), inplace=True)
    # convert cols from dict to float values for the <comm_group> provided
    dict_cols = ['battery_avg_ton', 'battery_avg_loc', 'support_diesel_avg_ton']
    for col in dict_cols:
        edges_gdf[col] = edges_gdf[col].apply(lambda x: x[comm_group])

    # create a dict to map {(u, v): (u, v), (v, u): (u, v)}
    edge_mapper = dict()
    for u, v in edges_gdf.index:
        if (u, v) not in edge_mapper.keys():
            edge_mapper[u, v] = (u, v)
            edge_mapper[v, u] = (u, v)
    # map indices
    edges_gdf.rename(index=edge_mapper, inplace=True)
    edges_gdf.fillna(0, inplace=True)
    # groupby (u, v), summing values of 'battery_avg_ton', 'battery_avg_loc', 'support_diesel_avg_ton'
    edges_gdf.groupby(by=['u', 'v']).agg(agg_cols)
    # compute 'share_battery'
    edges_gdf['share_battery'] = 100 * edges_gdf['battery_avg_ton'].div(edges_gdf['support_diesel_avg_ton'] +
                                                                        edges_gdf['battery_avg_ton']).replace(np.inf,
                                                                                                              0.00)
    edges_gdf['share_battery'] = edges_gdf['share_battery'].fillna(0.00)
    # assign line width to each edge based on battery flow tonnage
    edges_gdf['line_width'] = edges_gdf['battery_avg_ton'].apply(lambda x: line_size(x))
    # reset index
    edges_gdf.reset_index(inplace=True)
    # groupby line_width groups and (u, v)
    edges_gdf = edges_gdf.groupby(by=['line_width', 'u', 'v']).first()
    # get line widths
    line_widths = sorted(list(set(edges_gdf.index.get_level_values('line_width'))))

    legend_name = 'Battery Network'
    lg_group = 1

    for lw in line_widths:
        e = edges_gdf.loc[lw, slice(None, None)]
        lats = []
        lons = []
        names = []
        for u, v in e.index:
            x, y = e.loc[(u, v), 'geometry'].xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            name = '{v1} miles <br>{v2} {v3} tons/day <br>{v4} {v5} loc/day <br>' \
                   'Share of {v6} tons moved by battery: {v7}%'.format(v1=round(e.loc[(u, v), 'miles']),
                                                                       v2=round(e.loc[(u, v), 'battery_avg_ton']),
                                                                       v3=comm_group.capitalize(),
                                                                       v4=round(e.loc[(u, v), 'battery_avg_loc']),
                                                                       v5=comm_group.capitalize(),
                                                                       v6=comm_group.lower(),
                                                                       v7=round(e.loc[(u, v), 'share_battery']))
            names = np.append(names, [name] * len(y))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            names = np.append(names, None)

        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(
                    width=lw,
                    color=green4,
                ),
                opacity=1,
                hoverinfo="text",
                hovertext=names,
                legendgroup=lg_group,
                name=legend_name,
                showlegend=lw == 2,
                connectgaps=False,
            )
        )

    print('\t EDGES:: ' + str(time.time() - t0))

    legend_bool = [True, True]
    # od_set = {u for u, _ in G.graph['framework']['ods']}.union({v for _, v in G.graph['framework']['ods']})

    t0 = time.time()
    for i in range(len(nodes_gdf)):
        n = nodes_gdf.loc[nodes_gdf.index[i]]

        if n['facility'] == 1:
            if n['avg']['energy_transfer'] == 1:
                avg_charged_mwh = -n['avg']['daily_demand_mwh']
                peak_charged_mwh = -n['peak']['daily_demand_mwh']
            else:
                avg_charged_mwh = n['avg']['daily_supply_mwh']
                peak_charged_mwh = n['peak']['daily_supply_mwh']
            if (n['energy_source_TEA']['avg_queue_time_p_loc'] is not None) and (
                    n['energy_source_TEA']['peak_queue_time_p_loc'] is not None):
                text = n['city'] + ', ' + n['state'] + '<br>' + \
                       str(round(avg_charged_mwh, 2)) + ' MWh/day <br>' + \
                       str(int(n['avg']['number_loc'])) + ' loc/day <br>' + \
                       str(n['energy_source_TEA']['number_chargers']) + ' chargers <br>' + \
                       'Avg. Queue Time: ' + str(
                    round(n['energy_source_TEA']['avg_queue_time_p_loc'], 2)) + ' hrs <br>' + \
                       'Avg. Queue Length: ' + str(round(n['energy_source_TEA']['avg_queue_length'], 2)) + ' loc <br>' + \
                       'Peak Queue Time: ' + str(round(n['energy_source_TEA']['peak_queue_time_p_loc'], 2)) + \
                       ' hrs <br>' + \
                       'Peak Queue Length: ' + str(
                    round(n['energy_source_TEA']['peak_queue_length'], 2)) + ' loc <br>' + \
                       'Utilized ' + str(
                    round(n['energy_source_TEA']['actual_utilization'] * 24, 1)) + ' hrs/day <br>' + \
                       'Total LCO: ' + str(round(n['energy_source_TEA']['total_LCO'], 3)) + ' $/kWh <br>' + \
                       'WTW Emissions: ' + str(round(n['energy_source_LCA']['emissions_tonco2_kwh'] * 1e6, 3)) + \
                       ' g CO<sub>2</sub>' + '/kWh <br>' + \
                       'Capital Cost: \t $' + str(round(n['energy_source_TEA']['station_total'] / 1e6, 2)) + ' M <br>' + \
                       'Avg. Delay cost per car: \t $' + \
                       str(round(n['energy_source_TEA']['avg_daily_delay_cost_p_car'], 2)) + '<br>' + \
                       'Avg. Daily Delay cost per loc: \t $' + \
                       str(round(n['energy_source_TEA']['avg_daily_delay_cost_p_loc'], 2)) + '<br>' + \
                       'Total Daily Delay Cost: \t $' + \
                       str(round(n['energy_source_TEA']['total_daily_delay_cost'] / 1e3, 2)) + ' K<br>'
            else:
                text = n['city'] + ', ' + n['state'] + '<br>' + \
                       str(round(avg_charged_mwh, 2)) + ' MWh/day <br>' + \
                       str(int(n['avg']['number_loc'])) + ' loc/day <br>' + \
                       str(n['energy_source_TEA']['number_chargers']) + ' chargers <br>' + \
                       'Avg. Queue Time: ' + str(
                    round(0, 2)) + ' hrs <br>' + \
                       'Avg. Queue Length: ' + str(round(0, 2)) + ' loc <br>' + \
                       'Peak Queue Time: ' + str(round(0, 2)) + \
                       ' hrs <br>' + \
                       'Peak Queue Length: ' + str(
                    round(0, 2)) + ' loc <br>' + \
                       'Utilized ' + str(
                    round(n['energy_source_TEA']['actual_utilization'] * 24, 1)) + ' hrs/day <br>' + \
                       'Total LCO: ' + str(round(n['energy_source_TEA']['total_LCO'], 3)) + ' $/kWh <br>' + \
                       'WTW Emissions: ' + str(round(n['energy_source_LCA']['emissions_tonco2_kwh'] * 1e6, 3)) + \
                       ' g CO<sub>2</sub>' + '/kWh <br>' + \
                       'Capital Cost: \t $' + str(round(n['energy_source_TEA']['station_total'] / 1e6, 2)) + ' M <br>' + \
                       'Avg. Delay cost per car: \t $' + \
                       str(round(0, 2)) + '<br>' + \
                       'Avg. Daily Delay cost per loc: \t $' + \
                       str(round(0, 2)) + '<br>' + \
                       'Total Daily Delay Cost: \t $' + \
                       str(round(0, 2)) + ' K<br>'

            fig.add_trace(
                go.Scattergeo(
                    uid=n['nodeid'],
                    locationmode='USA-states',
                    lon=[n['geometry'].x],
                    lat=[n['geometry'].y],
                    hovertemplate=text,
                    # mode='markers+text',
                    # text=str(round(n['energy_source_TEA']['number_chargers'])),
                    # textfont=dict(color='black', size=14),
                    marker=dict(
                        size=5 * np.log(peak_charged_mwh + 10),
                        # size=25,
                        color=green4,
                        sizemode='area',
                    ),
                    # opacity=1,
                    opacity=0.8,
                    legendgroup=lg_group,
                    name='Charging Facility',
                    showlegend=legend_bool[0],
                ),
                # row=row, col=col
            )

            legend_bool[0] = False
    print('\t FACILITY NODES:: ' + str(time.time() - t0))

    t0 = time.time()
    for i in range(len(nodes_gdf)):
        n = nodes_gdf.loc[nodes_gdf.index[i]]
        if n['covered'] == 1:
            text = n['city'] + ', ' + n['state']
            lg_group = 1
            fig.add_trace(
                go.Scattergeo(
                    uid=n['nodeid'],
                    locationmode='USA-states',
                    lon=[n['geometry'].x],
                    lat=[n['geometry'].y],
                    hoverinfo='skip',
                    # hovertemplate=text,
                    marker=dict(size=6,
                                color=green4,
                                symbol='square'),
                    legendgroup=lg_group,
                    name='Covered (Non-Charging) Facility',
                    showlegend=legend_bool[1],
                ),
                # row=row, col=col
            )
            legend_bool[1] = False
    print('\t COVERED NODES:: ' + str(time.time() - t0))

    if additional_plots:
        if figlist:
            fig.update_geos(projection_type="albers usa")
            fig.update_layout(
                autosize=True,
                margin=dict(l=0, r=0, b=300, t=0, pad=1),
                showlegend=legend_show,
                legend=dict(
                    itemsizing='trace',
                    orientation='h',
                    yanchor="top",
                    y=0.95,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, color=black)
                ),
            )
            for annotation in fig.layout.annotations:
                annotation.font.size = 12

            figs = [fig]
            # table for summary statistics
            figs.append(battery_summary_table(G=G, comm_group=comm_group))
            # bar charts for LCA stats
            figs.append(battery_lca_plot(G=G, comm_group=comm_group, legend_show=legend_show))
            # bar charts for TEA stats
            figs.append(battery_tea_plot(G=G, comm_group=comm_group, legend_show=legend_show))
            # pie charts for operational stats
            figs.append(battery_pie_operations_plot(G=G, comm_group=comm_group))
            # table for cost of avoided emissions
            figs.append(battery_cost_avoided_table(G=G, comm_group=comm_group))
            fig = figs
        else:
            # pie charts for operational stats
            fig = battery_pie_operations_plot(G=G, comm_group=comm_group, fig=fig)
            # bar charts for LCA stats
            fig = battery_lca_plot(G=G, comm_group=comm_group, fig=fig, legend_show=legend_show)
            # bar charts for TEA stats
            fig = battery_tea_plot(G=G, comm_group=comm_group, fig=fig, legend_show=legend_show)
            # table for summary statistics
            fig = battery_summary_table(G=G, comm_group=comm_group, fig=fig)
            # table for cost of avoided emissions
            fig = battery_cost_avoided_table(G=G, comm_group=comm_group, fig=fig)

            fig.update_geos(projection_type="albers usa")
            fig.update_layout(
                showlegend=legend_show,
                legend=dict(
                    itemsizing='trace',
                    yanchor='middle',
                    xanchor='right',
                    orientation='v',
                    x=.9,
                    y=0.5,
                    font=dict(size=12, color=black)
                ),
            )
            for annotation in fig.layout.annotations:
                annotation.font.size = 12

            labels = []
    else:
        fig.update_geos(projection_type="albers usa")
        fig_title = ''
        fig.update_layout(title=dict(text=fig_title, font=dict(color='black', size=6)), font=dict(color='black'),
                          showlegend=legend_show,
                          legend=dict(
                              yanchor='middle',
                              xanchor='right',
                              orientation='v',
                              x=.9,
                              y=0.5),
                          )
        for annotation in fig.layout.annotations:
            annotation.font.size = 12

    return fig


def hybrid_pie_operations_plot(G, comm_group: str, fig=None):
    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "domain"}]],
        )
        row = 1
        col = 1
    else:
        row = 3
        col = 1

    labels = ['Hybrid', 'Diesel']
    support_diesel_tonmi = G.graph['operations']['support_diesel_total_tonmi'][comm_group]
    hybrid_tonmi = G.graph['operations']['alt_tech_total_tonmi'][comm_group]
    # use the one below if it is desired to adjust the ton-miles to reflect the baseline ton-miles
    # battery_tonmi = ((1 - G.graph['operations']['perc_tonmi_inc'] / 100) *
    #                  G.graph['operations']['alt_tech_total_tonmi'][comm_group])
    values = [hybrid_tonmi * 365 / 1e6,
              support_diesel_tonmi * 365 / 1e6]
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=[green4, purple]),
            textinfo='label+percent',
            textposition='inside',
            hovertemplate='<b>%{label}</b> <br> %{value:.0f} [M ton-mi/yr]',
            name='',
            showlegend=False
        ),
        row=row, col=col
    )

    fig.update_layout(font_color=black,
                      autosize=True,
                      margin=dict(l=30, r=30, b=0, t=20, pad=1),
                      legend=dict(
                          orientation='h',
                          yanchor='bottom',
                          y=0,
                          xanchor='center',
                          x=0.5
                      ),
                      title=dict(
                          text=comm_group.capitalize() + ' Ton-Miles',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(t=15, b=5)
                      )
                      )

    return fig


def hybrid_tea_plot(G, comm_group: str, fig=None, legend_show=True):
    # compute aggregate statistics to plot and be able to compare between battery and diesel

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "bar"}]],
        )
        row = 1
        col = 1
    else:
        row = 2
        col = 2

    fig.add_trace(
        go.Bar(x=['Diesel', 'Hybrid'],
               y=[0,
                  G.graph['energy_source_TEA']['station_LCO_tonmi'][comm_group] * 100],
               name='Station',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green5)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Hybrid'],
               y=[0,
                  G.graph['energy_source_TEA']['om_LCO_tonmi'][comm_group] * 100],
               name='Station O&M',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green2)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Hybrid'],
               y=[0,
                  G.graph['energy_source_TEA']['battery_LCO_tonmi'][comm_group] * 100],
               name='Battery',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green3)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Hybrid'],
               y=[0,
                  G.graph['energy_source_TEA']['energy_LCO_tonmi'][comm_group] * 100],
               name='Electricity',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green1)
               ),
        row=row, col=col
    )
    # fig.add_trace(
    #     go.Bar(x=['Diesel', 'Hybrid'],
    #            y=[0,
    #               G.graph['energy_source_TEA']['delay_LCO_tonmi'][comm_group] * 100],
    #            name='Delay',
    #            hovertemplate='%{y:.2f} [¢/ton-mi]',
    #            # showlegend=False,
    #            marker=dict(color=red)
    #            ),
    #     row=row, col=col
    # )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Hybrid'],
               y=[G.graph['diesel_TEA']['fuel_LCO_tonmi'][comm_group] * 100,
                  G.graph['energy_source_TEA']['fuel_LCO_tonmi'][comm_group] * 100],
               name='Fuel',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=purple)
               ),
        row=row, col=col
    )

    # Scenario average
    fig.add_shape(type='line',
                  xref='paper', yref='y',
                  x0=-.5, y0=G.graph['energy_source_TEA']['total_scenario_nodelay_LCO_tonmi'][comm_group] * 100,
                  x1=1.5, y1=G.graph['energy_source_TEA']['total_scenario_nodelay_LCO_tonmi'][comm_group] * 100,
                  line=dict(color=black, width=2, dash='dash'),
                  opacity=1,
                  name='Scenario Average',
                  row=row, col=col
                  )
    fig.add_trace(
        go.Scatter(x=['Hybrid'],
                   y=[G.graph['energy_source_TEA']['total_scenario_nodelay_LCO_tonmi'][comm_group] * 100],
                   mode='markers',
                   marker=dict(symbol='line-ew', size=6, line_width=2, opacity=1, color=black, line_color=black),
                   # opacity=1,
                   name='Scenario <br> Average',
                   hovertemplate='%{y:.2f} [¢/ton-mi]',
                   # showlegend=False
                   ),
        row=row, col=col
    )
    fig.update_layout(barmode='stack',
                      font=dict(color=black, size=12),
                      autosize=True,
                      margin=dict(l=20, r=0, b=0, t=50, pad=0),
                      showlegend=legend_show,
                      legend=dict(
                          orientation='h',
                          yanchor='top',
                          y=1,
                          xanchor='right',
                          x=1.9,
                          font=dict(size=10)
                      ),
                      title=dict(
                          text='Levelized Cost <br> of Operation',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(t=15, b=5)
                      )
                      )

    # fig.add_annotation(hovertext='Scenario Average: ' +
    #                              str(round(G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
    #                                        2)) + ' [¢/ton-mi]',
    #                    # text='Scenario Average',
    #                    text='',
    #                    xref='paper', yref='y',
    #                    x=1.5, y=G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
    #                    xanchor='left', yanchor='middle',
    #                    arrowcolor=black,
    #                    font=dict(color=black),
    #                    showarrow=False,
    #                    row=row, col=col
    #                    )

    # update yaxis properties
    fig.update_yaxes(title=dict(text='[¢ / ton-mi]', standoff=10,
                                font=dict(size=12)), showgrid=False, row=row, col=col)

    return fig


def hybrid_lca_plot(G, comm_group: str, fig=None, legend_show=True):
    # compute aggregate statistics to plot and be able to compare between battery and diesel

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "xy", 'secondary_y': True}]],
        )
        row = 1
        col = 1
    else:
        row = 2
        col = 1

    # ton CO2
    # pure diesel
    fig.add_trace(
        go.Bar(x=['100% Diesel', 'Scenario'],
               y=[G.graph['diesel_LCA']['annual_total_emissions_tonco2'][comm_group] / 1e3,
                  G.graph['energy_source_LCA']['annual_support_diesel_total_emissions'][comm_group] / 1e3],
               name='Diesel',
               hovertemplate='%{y:.0f} [kton CO<sub>2</sub>/yr]',
               # showlegend=False,
               marker=dict(color=purple)
               ),
        row=row, col=col
    )
    # electric
    fig.add_trace(
        go.Bar(x=['100% Diesel', 'Scenario'],
               y=[0,
                  G.graph['energy_source_LCA']['annual_hybrid_battery_total_emissions'][comm_group] / 1e3],
               name='Electric Grid',
               hovertemplate='%{y:.0f} [kton CO<sub>2</sub>/yr]',
               # showlegend=False,
               marker=dict(color=green4)
               ),
        secondary_y=False,
        row=row, col=col
    )
    # hybrid diesel
    fig.add_trace(
        go.Bar(x=['100% Diesel', 'Scenario'],
               y=[0,
                  G.graph['energy_source_LCA']['annual_hybrid_diesel_total_emissions'][comm_group] / 1e3],
               name='Hybrid Diesel',
               hovertemplate='%{y:.0f} [kton CO<sub>2</sub>/yr]',
               # showlegend=False,
               marker=dict(color=green1)
               ),
        secondary_y=False,
        row=row, col=col
    )
    # ton CO2 /tonmile
    fig.add_trace(
        go.Scatter(x=['100% Diesel', 'Scenario'],
                   y=[1e6 * G.graph['diesel_LCA']['emissions_tonco2_tonmi'][comm_group],
                      1e6 * G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'][comm_group]],
                   mode='markers',
                   marker=dict(symbol='diamond', size=10, color=light_red),
                   name='WTW Emissions Rate',
                   hovertemplate='%{y:.2f} [g CO<sub>2</sub>/ton-mi]',
                   showlegend=False
                   ),
        secondary_y=True,
        row=row, col=col
    )

    fig.update_layout(barmode='stack',
                      font_color=black,
                      autosize=True,
                      margin=dict(l=0, r=0, b=0, t=50, pad=1),
                      showlegend=legend_show,
                      legend=dict(
                          orientation='h',
                          yanchor='top',
                          y=1.2,
                          xanchor='center',
                          x=0.5,
                          font=dict(size=10)),
                      title=dict(
                          text='WTW Emissions',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(b=30)
                      )
                      )

    # update yaxis properties
    fig.update_yaxes(title=dict(text='[kton CO<sub>2</sub>]', standoff=10,
                                font=dict(size=12)), secondary_y=False, showgrid=False,
                     row=row, col=col
                     )
    fig.update_yaxes(title=dict(text='[g CO<sub>2</sub> / ton-mi]', standoff=10,
                                font=dict(size=12, color=light_red)),
                     tickfont=dict(color=light_red), showgrid=False,
                     range=[0, np.ceil(1e6 * G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][comm_group] /
                                       10) * 10],
                     secondary_y=True,
                     row=row, col=col
                     )

    return fig


def hybrid_summary_table(G: nx.DiGraph, comm_group: str, fig=None):
    # plot table of summary of results for hybrid

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "table"}]]
        )
        row = 1
        col = 1
    else:
        row = 4
        col = 1

    des_tea = G.graph['energy_source_TEA']

    fig.add_trace(
        go.Table(
            header=dict(values=['Scenario Summary', 'Statistics'],
                        font=dict(size=14),
                        line_color=black,
                        fill_color=mid_purple,
                        ),
            cells=dict(values=[[
                'Cost of Avoided Emissions with Delay',
                '# of Charging Facilities',
                'Emissions Reduction' if G.graph['operations']['emissions_change'][comm_group] >= 0
                else 'Emissions Increase',
                'Total Levelized Cost of Operation with Delay',
                '# of Chargers',
                'Average Charger Utilization',
                'Station Capital Cost',
                'Total Delay Cost',
                'Total Annual Cost',
                'Avg. Charge Time per Loc',
                'Avg. Queue Time',
                'Avg. Queue Length',
                'Peak Queue Time',
                'Peak Queue Length',
                'Avg. Daily Delay Cost per Car'
            ],
                [
                    str(round(G.graph['operations']['cost_avoided_emissions'][comm_group], 2)) +
                    ' [$/kg CO<sub>2</sub>]',
                    str(sum([G.nodes[n]['facility'] for n in G])) + ' out of ' + str(G.number_of_nodes()),
                    str(round(abs(G.graph['operations']['emissions_change'][comm_group]), 2)) + ' %',
                    str(round(100 * G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group],
                              2)) + ' [¢/ton-mi]',
                    des_tea['number_chargers'],
                    str(round(des_tea['actual_utilization'] * 24, 1)) + ' [hrs/day]',
                    str(round(des_tea['station_total'] / 1e6, 2)) + ' [$M]',
                    str(round(des_tea['total_annual_delay_cost'] / 1e6, 3)) + ' [$M]',
                    str(round(des_tea['annual_total_cost'][comm_group] / 1e6, 2)) + ' [$M]',
                    str(round(des_tea['charge_time'], 2)) + ' [hr]',
                    str(round(des_tea['avg_queue_time_p_loc'], 3)) + ' [hr]',
                    str(round(des_tea['avg_queue_length'], 3)) + ' [loc]',
                    str(round(des_tea['peak_queue_time_p_loc'], 3)) + ' [hr]',
                    str(round(des_tea['peak_queue_length'], 3)) + ' [loc]',
                    str(round(des_tea['avg_daily_delay_cost_p_car'], 2)) + ' [$]'
                ]],
                font=dict(size=12),
                line_color=black,
                fill_color=light_purple,
            )
        ),
        row=row, col=col
    )

    fig.update_layout(font_color=black,
                      autosize=True,
                      margin=dict(l=5, r=5, b=5, t=5, pad=1)
                      )

    return fig


def hybrid_cost_avoided_table(G, comm_group: str, fig=None):
    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "table"}]]
        )
        row = 1
        col = 1
    else:
        row = 3
        col = 2

    fig.add_trace(
        go.Table(
            header=dict(values=['Cost of Avoided Emissions'],
                        font=dict(size=14),
                        line_color=black,
                        fill_color=mid_purple,
                        ),
            cells=dict(values=[[str(round(G.graph['operations']['cost_avoided_emissions_no_delay'][comm_group], 2)) +
                                ' [$/kg CO<sub>2</sub>]'
                                ]],
                       font=dict(size=12),
                       line_color=black,
                       fill_color=light_purple,
                       )
        ),
        row=row, col=col
    )

    fig.update_layout(font_color=black,
                      autosize=True,
                      # width=400,
                      # height=400,
                      margin=dict(l=5, r=5, b=5, t=5, pad=1)
                      )

    return fig


def hybrid_plot(G, comm_group: str, additional_plots=True, crs='WGS84', figlist=False, fig=None, legend_show=True):
    # fig.add_trace(go.Scattermapbox(lon=xs, lat=ys, mode='lines', hoverinfo="text", hovertext=caption))

    if fig is None:
        if additional_plots and not figlist:
            fig = make_subplots(
                rows=4, cols=3,
                specs=[[None, None, {"type": "scattergeo", "rowspan": 4}],
                       [{"type": "xy", 'secondary_y': True}, {"type": "bar"}, None],
                       [{"type": "domain"}, {"type": "table"}, None],
                       [{"type": "table", 'colspan': 2}, None, None]],
                column_widths=[0.1, 0.1, 0.8],
                row_heights=[0.02, 0.25, 0.25, 0.48],
                horizontal_spacing=0.1,
                subplot_titles=(None,
                                'WTW Emissions', 'Levelized Cost of Operation',
                                comm_group.capitalize() + ' Ton-Miles', None,
                                None)
            )
            row = 1
            col = 3
        else:
            t0 = time.time()
            fig = base_plot(G.graph['railroad'])
            print('\t LOAD BASE PLOT:: ' + str(time.time() - t0))
            row = 1
            col = 1

    # fig = plot_states_bg()

    fuel_type = G.graph['scenario']['fuel_type']
    fuel_type_battery = G.graph['scenario']['fuel_type'] + '_battery'
    fuel_type_diesel = G.graph['scenario']['fuel_type'] + '_diesel'

    # G = project_graph(G.copy(), to_crs=crs)
    t0 = time.time()
    nodes_gdf, edges_gdf = gdfs_from_graph(G, crs=crs, smooth_geometry=False)
    print('GDF EXTRACTION:: ' + str(time.time() - t0))

    t0 = time.time()
    # drop non-covered edges
    edges_gdf.drop(index=edges_gdf[edges_gdf['covered'] == 0].index, inplace=True)
    # keep only these cols
    agg_cols = {'miles': 'first', 'geometry': 'first',
                fuel_type + '_avg_ton': 'sum', fuel_type_battery + '_avg_loc': 'sum',
                fuel_type_diesel + '_avg_loc': 'sum',
                'support_diesel_avg_ton': 'sum'}
    edges_gdf.drop(columns=set(edges_gdf.columns).difference(set(agg_cols.keys())), inplace=True)
    # convert cols from dict to float values for the <comm_group> provided
    dict_cols = [fuel_type + '_avg_ton', fuel_type_battery + '_avg_loc', fuel_type_diesel + '_avg_loc',
                 'support_diesel_avg_ton']
    for col in dict_cols:
        edges_gdf[col] = edges_gdf[col].apply(lambda x: x[comm_group])

    # create a dict to map {(u, v): (u, v), (v, u): (u, v)}
    edge_mapper = dict()
    for u, v in edges_gdf.index:
        if (u, v) not in edge_mapper.keys():
            edge_mapper[u, v] = (u, v)
            edge_mapper[v, u] = (u, v)
    # map indices
    edges_gdf.rename(index=edge_mapper, inplace=True)
    edges_gdf.fillna(0, inplace=True)
    # groupby (u, v), summing values of 'battery_avg_ton', 'battery_avg_loc', 'support_diesel_avg_ton'
    edges_gdf.groupby(by=['u', 'v']).agg(agg_cols)
    # compute 'share_hybrid'
    edges_gdf['share_' + fuel_type] = 100 * edges_gdf[fuel_type + '_avg_ton'].div(
        edges_gdf['support_diesel_avg_ton'] + edges_gdf[fuel_type + '_avg_ton']).replace(np.inf, 0.00)
    edges_gdf['share_' + fuel_type] = edges_gdf['share_' + fuel_type].fillna(0.00)
    # assign line width to each edge based on battery flow tonnage
    edges_gdf['line_width'] = edges_gdf[fuel_type + '_avg_ton'].apply(lambda x: line_size(x))
    # reset index
    edges_gdf.reset_index(inplace=True)
    # groupby line_width groups and (u, v)
    edges_gdf = edges_gdf.groupby(by=['line_width', 'u', 'v']).first()
    # get line widths
    line_widths = sorted(list(set(edges_gdf.index.get_level_values('line_width'))))

    legend_name = 'Hybrid Network'
    lg_group = 1

    for lw in line_widths:
        e = edges_gdf.loc[lw, slice(None, None)]
        lats = []
        lons = []
        names = []
        for u, v in e.index:
            x, y = e.loc[(u, v), 'geometry'].xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            name = '{v1} miles <br>{v2} {v3} tons/day <br>{v4} {v5} loc/day <br>' \
                   'Share of {v6} tons moved by hybrid: {v7}%'.format(v1=round(e.loc[(u, v), 'miles']),
                                                                      v2=round(e.loc[(u, v), fuel_type + '_avg_ton']),
                                                                      v3=comm_group.capitalize(),
                                                                      v4=round(e.loc[(u, v), fuel_type_battery +
                                                                                     '_avg_loc'] +
                                                                               e.loc[(u, v), fuel_type_diesel +
                                                                                     '_avg_loc']),
                                                                      v5=comm_group.capitalize(),
                                                                      v6=comm_group.lower(),
                                                                      v7=round(e.loc[(u, v), 'share_' + fuel_type]))
            names = np.append(names, [name] * len(y))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            names = np.append(names, None)

        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(
                    width=lw,
                    color=green4,
                ),
                opacity=1,
                hoverinfo="text",
                hovertext=names,
                legendgroup=lg_group,
                name=legend_name,
                showlegend=lw == 2,
                connectgaps=False,
            )
        )

    print('\t EDGES:: ' + str(time.time() - t0))

    legend_bool = [True, True]
    # od_set = {u for u, _ in G.graph['framework']['ods']}.union({v for _, v in G.graph['framework']['ods']})

    t0 = time.time()
    for i in range(len(nodes_gdf)):
        n = nodes_gdf.loc[nodes_gdf.index[i]]
        if n['facility'] == 1:
            if n['avg']['energy_transfer'] == 1:
                avg_charged_mwh = -n['avg']['daily_demand_mwh']
                peak_charged_mwh = -n['peak']['daily_demand_mwh']
            else:
                avg_charged_mwh = n['avg']['daily_supply_mwh']
                peak_charged_mwh = n['peak']['daily_supply_mwh']
            if (n['energy_source_TEA']['avg_queue_time_p_loc'] is not None) and (
                    n['energy_source_TEA']['peak_queue_time_p_loc'] is not None):
                text = n['city'] + ', ' + n['state'] + '<br>' + \
                       str(round(avg_charged_mwh, 2)) + ' MWh/day <br>' + \
                       str(int(n['avg']['number_loc'])) + ' loc/day <br>' + \
                       str(n['energy_source_TEA']['number_chargers']) + ' chargers <br>' + \
                       'Avg. Queue Time: ' + str(
                    round(n['energy_source_TEA']['avg_queue_time_p_loc'], 2)) + ' hrs <br>' + \
                       'Avg. Queue Length: ' + str(round(n['energy_source_TEA']['avg_queue_length'], 2)) + ' loc <br>' + \
                       'Peak Queue Time: ' + str(round(n['energy_source_TEA']['peak_queue_time_p_loc'], 2)) + \
                       ' hrs <br>' + \
                       'Peak Queue Length: ' + str(
                    round(n['energy_source_TEA']['peak_queue_length'], 2)) + ' loc <br>' + \
                       'Utilized ' + str(
                    round(n['energy_source_TEA']['actual_utilization'] * 24, 1)) + ' hrs/day <br>' + \
                       'Total LCO: ' + str(round(n['energy_source_TEA']['total_LCO'], 3)) + ' $/kWh <br>' + \
                       'WTW Emissions: ' + str(round(n['energy_source_LCA']['emissions_tonco2_kwh'] * 1e6, 3)) + \
                       ' g CO<sub>2</sub>' + '/kWh <br>' + \
                       'Capital Cost: \t $' + str(round(n['energy_source_TEA']['station_total'] / 1e6, 2)) + ' M <br>' + \
                       'Avg. Delay cost per car: \t $' + \
                       str(round(n['energy_source_TEA']['avg_daily_delay_cost_p_car'], 2)) + '<br>' + \
                       'Avg. Daily Delay cost per loc: \t $' + \
                       str(round(n['energy_source_TEA']['avg_daily_delay_cost_p_loc'], 2)) + '<br>' + \
                       'Total Daily Delay Cost: \t $' + \
                       str(round(n['energy_source_TEA']['total_daily_delay_cost'] / 1e3, 2)) + ' K<br>'
            else:
                text = n['city'] + ', ' + n['state'] + '<br>' + \
                       str(round(avg_charged_mwh, 2)) + ' MWh/day <br>' + \
                       str(int(n['avg']['number_loc'])) + ' loc/day <br>' + \
                       str(n['energy_source_TEA']['number_chargers']) + ' chargers <br>' + \
                       'Avg. Queue Time: ' + str(
                    round(0, 2)) + ' hrs <br>' + \
                       'Avg. Queue Length: ' + str(round(0, 2)) + ' loc <br>' + \
                       'Peak Queue Time: ' + str(round(0, 2)) + \
                       ' hrs <br>' + \
                       'Peak Queue Length: ' + str(
                    round(0, 2)) + ' loc <br>' + \
                       'Utilized ' + str(
                    round(n['energy_source_TEA']['actual_utilization'] * 24, 1)) + ' hrs/day <br>' + \
                       'Total LCO: ' + str(round(n['energy_source_TEA']['total_LCO'], 3)) + ' $/kWh <br>' + \
                       'WTW Emissions: ' + str(round(n['energy_source_LCA']['emissions_tonco2_kwh'] * 1e6, 3)) + \
                       ' g CO<sub>2</sub>' + '/kWh <br>' + \
                       'Capital Cost: \t $' + str(round(n['energy_source_TEA']['station_total'] / 1e6, 2)) + ' M <br>' + \
                       'Avg. Delay cost per car: \t $' + \
                       str(round(0, 2)) + '<br>' + \
                       'Avg. Daily Delay cost per loc: \t $' + \
                       str(round(0, 2)) + '<br>' + \
                       'Total Daily Delay Cost: \t $' + \
                       str(round(0, 2)) + ' K<br>'

            fig.add_trace(
                go.Scattergeo(
                    uid=n['nodeid'],
                    locationmode='USA-states',
                    lon=[n['geometry'].x],
                    lat=[n['geometry'].y],
                    hovertemplate=text,
                    # mode='markers+text',
                    # text=str(round(n['energy_source_TEA']['number_chargers'])),
                    # textfont=dict(color='black', size=14),
                    marker=dict(
                        size=5 * np.log(peak_charged_mwh + 10),
                        # size=25,
                        color=green4,
                        sizemode='area',
                    ),
                    # opacity=1,
                    opacity=0.8,
                    legendgroup=lg_group,
                    name='Charging Facility',
                    showlegend=legend_bool[0],
                ),
                # row=row, col=col
            )

            legend_bool[0] = False
    print('\t FACILITY NODES:: ' + str(time.time() - t0))

    t0 = time.time()
    for i in range(len(nodes_gdf)):
        n = nodes_gdf.loc[nodes_gdf.index[i]]
        if n['covered'] == 1:
            text = n['city'] + ', ' + n['state']
            lg_group = 1
            fig.add_trace(
                go.Scattergeo(
                    uid=n['nodeid'],
                    locationmode='USA-states',
                    lon=[n['geometry'].x],
                    lat=[n['geometry'].y],
                    hoverinfo='skip',
                    # hovertemplate=text,
                    marker=dict(size=6,
                                color=green4,
                                symbol='square'),
                    legendgroup=lg_group,
                    name='Covered (Non-Charging) Facility',
                    showlegend=legend_bool[1],
                ),
                # row=row, col=col
            )
            legend_bool[1] = False
    print('\t COVERED NODES:: ' + str(time.time() - t0))

    if additional_plots:
        if figlist:
            fig.update_geos(projection_type="albers usa")
            fig.update_layout(
                autosize=True,
                margin=dict(l=0, r=0, b=300, t=0, pad=1),
                showlegend=legend_show,
                legend=dict(
                    itemsizing='trace',
                    orientation='h',
                    yanchor="top",
                    y=0.95,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, color=black)
                ),
            )
            for annotation in fig.layout.annotations:
                annotation.font.size = 12

            figs = [fig]
            # table for summary statistics
            figs.append(hybrid_summary_table(G=G, comm_group=comm_group))
            # bar charts for LCA stats
            figs.append(hybrid_lca_plot(G=G, comm_group=comm_group, legend_show=legend_show))
            # bar charts for TEA stats
            figs.append(hybrid_tea_plot(G=G, comm_group=comm_group, legend_show=legend_show))
            # pie charts for operational stats
            figs.append(hybrid_pie_operations_plot(G=G, comm_group=comm_group))
            # table for cost of avoided emissions
            figs.append(hybrid_cost_avoided_table(G=G, comm_group=comm_group))
            fig = figs
        else:
            # pie charts for operational stats
            fig = hybrid_pie_operations_plot(G=G, comm_group=comm_group, fig=fig)
            # bar charts for LCA stats
            fig = hybrid_lca_plot(G=G, comm_group=comm_group, fig=fig, legend_show=legend_show)
            # bar charts for TEA stats
            fig = hybrid_tea_plot(G=G, comm_group=comm_group, fig=fig, legend_show=legend_show)
            # table for summary statistics
            fig = hybrid_summary_table(G=G, comm_group=comm_group, fig=fig)
            # table for cost of avoided emissions
            fig = hybrid_cost_avoided_table(G=G, comm_group=comm_group, fig=fig)

            fig.update_geos(projection_type="albers usa")
            fig.update_layout(
                showlegend=legend_show,
                legend=dict(
                    itemsizing='trace',
                    yanchor='middle',
                    xanchor='right',
                    orientation='v',
                    x=.9,
                    y=0.5,
                    font=dict(size=12, color=black)
                ),
                # legend=dict(
                #     orientation='h',
                #     yanchor="top",
                #     y=0.99,
                #     xanchor="right",
                #     x=0.2,
                # ),
                # margin=dict(l=0, r=0, t=0, b=0),
            )
            for annotation in fig.layout.annotations:
                annotation.font.size = 12

    else:
        fig.update_geos(projection_type="albers usa")
        fig_title = ''
        fig.update_layout(title=dict(text=fig_title, font=dict(color='black', size=6)), font=dict(color='black'),
                          showlegend=legend_show,
                          # legend=dict(
                          #     yanchor="top",
                          #     y=0.99,
                          #     xanchor="right",
                          #     x=0.99,
                          # ),
                          legend=dict(
                              yanchor='middle',
                              xanchor='right',
                              orientation='v',
                              x=.9,
                              y=0.5),
                          # margin=dict(l=0, r=0, t=0, b=0),
                          )
        for annotation in fig.layout.annotations:
            annotation.font.size = 12

    return fig


def plot_states_bg():
    # $ cost
    dfc = load_elec_cost_state_df()
    dfc['cost'] = dfc['Commercial']
    title = 'State <br> [$/MWh]'
    dfc.drop(index=['HI', 'AK'], inplace=True)
    # emissions
    # dfc = load_lca_battery_lookup()
    # dfc['cost'] = dfc['emissions']
    # title = 'State <br> [g CO2/kWh]'

    dfc['code'] = dfc.index

    fig = go.Figure(data=go.Choropleth(
        locations=dfc['code'],  # Spatial coordinates
        z=dfc['cost'].astype(float),  # Data to be color-coded
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='Reds',
        colorbar_title=title,
    ))

    fig.update_layout(
        # title_text='2011 US Agriculture Exports by State',
        geo_scope='usa',  # limit map scope to USA
    )

    # fig.show()
    # load states gdf
    # states_path = os.path.join(GEN_DIR, 'cb_2018_us_state_500k')
    # states_df = gpd.read_file(states_path)
    # extract continguous states
    # states_df = states_df[(states_df['STATEFP'].astype(int) < 60) & (states_df['STATEFP'].astype(int) != 15)
    #                       & (states_df['STATEFP'].astype(int) != 2)].copy()
    # states_df.to_crs(crs=crs, inplace=True)
    # # plot
    # fig, ax = plt.subplots(figsize=(9, 6))
    # states_df.boundary.plot(ax=ax, alpha=0.5, linewidth=0.5)
    # # remove tick marks
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # ax.set_title('Contiguous U.S.')

    return fig


'''
HYDROGEN PLOT
'''


def hydrogen_pie_operations_plot(G, comm_group: str, fig=None):
    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "domain"}]]
        )
        row = 1
        col = 1
    else:
        row = 3
        col = 1

    labels = ['Hydrogen', 'Diesel']
    support_diesel_tonmi = G.graph['operations']['support_diesel_total_tonmi'][comm_group]
    hydrogen_tonmi = G.graph['operations']['alt_tech_total_tonmi'][comm_group]
    # use the one below if it is desired to adjust the ton-miles to reflect the baseline ton-miles
    # hydrogen_tonmi = ((1 - G.graph['operations']['perc_tonmi_inc'] / 100) *
    #                   G.graph['operations']['alt_tech_total_tonmi'][comm_group])
    values = [hydrogen_tonmi * 365 / 1e6,
              support_diesel_tonmi * 365 / 1e6]
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=[green4, purple]),
            textinfo='label+percent',
            textposition='inside',
            hovertemplate='<b>%{label}</b> <br> %{value:.0f} [M ton-mi/yr]',
            name='',
            showlegend=False
        ),
        row=row, col=col
    )

    fig.update_layout(font_color=black,
                      autosize=True,
                      margin=dict(l=30, r=30, b=0, t=20, pad=1),
                      legend=dict(
                          orientation='h',
                          yanchor='bottom',
                          y=0,
                          xanchor='center',
                          x=0.5
                      ),
                      title=dict(
                          text=comm_group.capitalize() + ' Ton-Miles',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(t=15, b=5)
                      )
                      )

    return fig


def hydrogen_tea_plot(G, comm_group: str, fig=None, legend_show=True):
    # compute aggregate statistics to plot and be able to compare between battery and diesel

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "bar"}]]
        )
        row = 1
        col = 1
    else:
        row = 2
        col = 2

    fig.add_trace(
        go.Bar(x=['Diesel', 'Hydrogen'],
               y=[G.graph['diesel_TEA']['fuel_LCO_tonmi'][comm_group] * 100,
                  G.graph['energy_source_TEA']['fuel_LCO_tonmi'][comm_group] * 100],
               name='Fuel',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=purple)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Hydrogen'],
               y=[0,
                  G.graph['energy_source_TEA']['station_LCO_tonmi'][comm_group] * 100],
               name='Station',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green5)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Hydrogen'],
               y=[0,
                  G.graph['energy_source_TEA']['terminal_LCO_tonmi'][comm_group] * 100],
               name='Terminal',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green1)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Hydrogen'],
               y=[0,
                  G.graph['energy_source_TEA']['liquefier_LCO_tonmi'][comm_group] * 100],
               name='Liquefier',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green2)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Hydrogen'],
               y=[0,
                  G.graph['energy_source_TEA']['tender_LCO_tonmi'][comm_group] * 100],
               name='Tender Car',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=green4)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['Diesel', 'Hydrogen'],
               y=[0,
                  G.graph['energy_source_TEA']['delay_LCO_tonmi'][comm_group] * 100],
               name='Delay',
               hovertemplate='%{y:.2f} [¢/ton-mi]',
               # showlegend=False,
               marker=dict(color=red)
               ),
        row=row, col=col
    )

    # fig.add_shape(type='line',
    #               xref='paper', yref='y',
    #               x0=-0.50, y0=G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
    #               x1=1.5, y1=G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
    #               line=dict(color=black, width=2, dash='dash'),
    #               opacity=1,
    #               name='Scenario Average',
    #               row=row, col=col
    #               )
    # fig.add_annotation(hovertext=str(round(G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
    #                                        2)) + ' [¢/ton-mi]',
    #                    text='Scenario Average',
    #                    xref='paper', yref='y',
    #                    x=1.5, y=G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
    #                    xanchor='left', yanchor='middle',
    #                    arrowcolor=black,
    #                    font=dict(color=black),
    #                    showarrow=False,
    #                    row=row, col=col
    #                    )

    # Scenario average
    fig.add_shape(type='line',
                  xref='paper', yref='y',
                  x0=-.5, y0=G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
                  x1=1.5, y1=G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100,
                  line=dict(color=black, width=2, dash='dash'),
                  opacity=1,
                  name='Scenario Average',
                  row=row, col=col
                  )
    fig.add_trace(
        go.Scatter(x=['Hydrogen'],
                   y=[G.graph['energy_source_TEA']['total_scenario_LCO_tonmi'][comm_group] * 100],
                   mode='markers',
                   marker=dict(symbol='line-ew', size=6, line_width=2, opacity=1, color=black, line_color=black),
                   # opacity=1,
                   name='Scenario <br> Average',
                   hovertemplate='%{y:.2f} [¢/ton-mi]',
                   # showlegend=False
                   ),
        row=row, col=col
    )

    fig.update_layout(barmode='stack',
                      font=dict(color=black, size=12),
                      autosize=True,
                      margin=dict(l=20, r=0, b=0, t=50, pad=0),
                      showlegend=legend_show,
                      legend=dict(
                          orientation='h',
                          yanchor='top',
                          y=1,
                          xanchor='right',
                          x=1.9,
                          font=dict(size=10)
                      ),
                      title=dict(
                          text='Levelized Cost <br> of Operation',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(t=15, b=5)
                      )
                      )

    # update yaxis properties
    fig.update_yaxes(title=dict(text='[¢ / ton-mi]', standoff=10,
                                font=dict(size=12)), showgrid=False, row=row, col=col)

    return fig


def hydrogen_lca_plot(G, comm_group: str, fig=None, legend_show=True):
    # compute aggregate statistics to plot and be able to compare between battery and diesel

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "xy", 'secondary_y': True}]]
        )
        row = 1
        col = 1
    else:
        row = 2
        col = 1

    # ton CO2
    fig.add_trace(
        go.Bar(x=['100% Diesel', 'Scenario'],
               y=[G.graph['diesel_LCA']['annual_total_emissions_tonco2'][comm_group] / 1000,
                  G.graph['energy_source_LCA']['annual_support_diesel_total_emissions'][comm_group] / 1000],
               name='Diesel',
               hovertemplate='%{y:.0f} [kton CO<sub>2</sub>/yr]',
               # showlegend=False,
               marker=dict(color=purple)
               ),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(x=['100% Diesel', 'Scenario'],
               y=[0,
                  G.graph['energy_source_LCA']['annual_hydrogen_total_emissions'][comm_group] / 1000],
               name='Hydrogen',
               hovertemplate='%{y:.0f} [kton CO<sub>2</sub>/yr]',
               # showlegend=False,
               marker=dict(color=green4)
               ),
        secondary_y=False,
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=['100% Diesel', 'Scenario'],
                   y=[1e6 * G.graph['diesel_LCA']['emissions_tonco2_tonmi'][comm_group],
                      1e6 * G.graph['energy_source_LCA']['avg_emissions_tonco2_tonmi'][comm_group]],
                   mode='markers',
                   marker=dict(symbol='diamond', size=10, color=light_red),
                   name='WTW Emissions Rate',
                   hovertemplate='%{y:.2f} [g CO<sub>2</sub>/ton-mi]',
                   showlegend=False
                   ),
        secondary_y=True,
        row=row, col=col
    )

    fig.update_layout(barmode='stack',
                      font_color=black,
                      autosize=True,
                      margin=dict(l=0, r=0, b=0, t=50, pad=1),
                      showlegend=legend_show,
                      legend=dict(
                          orientation='h',
                          yanchor='top',
                          y=1.2,
                          xanchor='center',
                          x=0.5,
                          font=dict(size=10)),
                      title=dict(
                          text='WTW Emissions',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(b=30)
                      )
                      )

    # update yaxis properties
    fig.update_yaxes(title=dict(text='[kton CO<sub>2</sub>]', standoff=10,
                                font=dict(size=12)), secondary_y=False, showgrid=False, row=row, col=col)
    fig.update_yaxes(title=dict(text='[g CO<sub>2</sub> / ton-mi]', standoff=10,
                                font=dict(size=12, color=light_red)),
                     tickfont=dict(color=light_red), showgrid=False,
                     range=[0, np.ceil(1e6 * G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][comm_group] /
                                       10) * 10],
                     secondary_y=True, row=row, col=col)

    return fig


def hydrogen_summary_table(G: nx.DiGraph, comm_group: str, fig=None):
    # plot table of summary of results for battery

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "table"}]]
        )
        row = 1
        col = 1
    else:
        row = 4
        col = 1

    des_tea = G.graph['energy_source_TEA']

    fig.add_trace(
        go.Table(
            header=dict(values=['Scenario Summary', 'Statistics'],
                        font=dict(size=14),
                        line_color=black,
                        fill_color=mid_purple,
                        ),
            cells=dict(values=[[
                'Cost of Avoided Emissions',
                '# of Refueling Facilities',
                comm_group.capitalize() + ' Emissions Reduction'
                if G.graph['operations']['emissions_change'][comm_group] >= 0
                else ' Emissions Increase',
                'Average Route Distance Increase',
                '# of Pumps',
                'Average Pump Utilization',
                'Station Capital Cost',
                'Total Delay Cost',
                'Total Annual Cost',
                'Avg. Refuel Time per Loc',
                'Avg. Queue Time',
                'Avg. Queue Length',
                'Peak Queue Time',
                'Peak Queue Length',
                'Avg. Daily Delay Cost per Car'
            ],
                [
                    str(round(G.graph['operations']['cost_avoided_emissions'][comm_group], 2)) +
                    ' [$/kg CO<sub>2</sub>]',
                    str(sum([G.nodes[n]['facility'] for n in G])) + ' out of ' + str(G.number_of_nodes()),
                    str(round(abs(G.graph['operations']['emissions_change'][comm_group]), 2)) + ' %',
                    str(round(G.graph['operations']['perc_mi_inc'][comm_group], 2)) + ' %',
                    des_tea['number_pumps'],
                    str(round(des_tea['actual_utilization'] * 24, 1)) + ' [hrs/day]',
                    str(round(des_tea['station_total'] / 1e6, 2)) + ' [$M]',
                    str(round(des_tea['total_annual_delay_cost'] / 1e6, 3)) + ' [$M]',
                    str(round(des_tea['annual_total_cost'][comm_group] / 1e6, 2)) + ' [$M]',
                    str(round(des_tea['pump_time'], 2)) + ' [hr]',
                    str(round(des_tea['avg_queue_time_p_loc'], 3)) + ' [hr]',
                    str(round(des_tea['avg_queue_length'], 3)) + ' [loc]',
                    str(round(des_tea['peak_queue_time_p_loc'], 3)) + ' [hr]',
                    str(round(des_tea['peak_queue_length'], 3)) + ' [loc]',
                    str(round(des_tea['avg_daily_delay_cost_p_car'], 2)) + ' [$]'
                ]],
                font=dict(size=12),
                line_color=black,
                fill_color=light_purple,
            )
        ),
        row=row, col=col
    )

    fig.update_layout(font_color=black,
                      autosize=True,
                      margin=dict(l=5, r=5, b=5, t=5, pad=1)
                      )

    return fig


def hydrogen_cost_avoided_table(G, comm_group: str, fig=None):
    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "table"}]]
        )
        row = 1
        col = 1
    else:
        row = 3
        col = 2

    fig.add_trace(
        go.Table(
            header=dict(values=['Cost of Avoided Emissions'],
                        font=dict(size=14),
                        line_color=black,
                        fill_color=mid_purple,
                        ),
            cells=dict(values=[[str(round(G.graph['operations']['cost_avoided_emissions'][comm_group], 2)) +
                                ' [$/kg CO<sub>2</sub>]'
                                ]],
                       font=dict(size=12),
                       line_color=black,
                       fill_color=light_purple,
                       )
        ),
        row=row, col=col
    )

    fig.update_layout(font_color=black,
                      autosize=True,
                      margin=dict(l=5, r=5, b=5, t=5, pad=1)
                      )

    return fig


# def hydrogen_plot(G, comm_group: str, additional_plots=True, crs='WGS84', figlist=False, fig=None, figshow=True):
#     # plotly.colors.qualitative G10
#     # blue,  # blue     plotly.colors.qualitative.G10[0]
#     # green,  # green    plotly.colors.qualitative.G10[3]
#     # '#ff7f0e',  # safety orange
#     # '#2ca02c',  # cooked asparagus green
#
#     if fig is None:
#         if additional_plots and not figlist:
#             fig = make_subplots(
#                 rows=4, cols=3,
#                 specs=[[None, None, {"type": "scattergeo", "rowspan": 4}],
#                        [{"type": "xy", 'secondary_y': True}, {"type": "bar"}, None],
#                        [{"type": "domain"}, {"type": "table"}, None],
#                        [{"type": "table", 'colspan': 2}, None, None]],
#                 column_widths=[0.1, 0.1, 0.8],
#                 row_heights=[0.02, 0.25, 0.25, 0.48],
#                 horizontal_spacing=0.1,
#                 subplot_titles=(None,
#                                 'WTW Emissions', 'Levelized Cost of Operation',
#                                 comm_group.capitalize() + ' Ton-Miles', None,
#                                 None)
#             )
#             row = 1
#             col = 3
#         else:
#             fig = make_subplots(
#                 rows=1, cols=1,
#                 specs=[[{"type": "scattergeo"}]]
#             )
#             row = 1
#             col = 1
#
#     G = project_graph(G.copy(), to_crs=crs)
#     nodes_gdf, edges_gdf = gdfs_from_graph(G, smooth_geometry=True)
#
#     edges_gdf['lons'] = edges_gdf['geometry'].apply(lambda c: list(c.xy[0]))
#     edges_gdf['lats'] = edges_gdf['geometry'].apply(lambda c: list(c.xy[1]))
#
#     legend_names = ['Hydrogen Network', 'Diesel Network']
#     legend_bool = [True, True]
#     for i in range(len(edges_gdf)):
#         # plot baseline (all edges) first
#         if edges_gdf.loc[edges_gdf.index[i], 'covered'] == 0:
#             lg_group = 2
#             fig.add_trace(
#                 go.Scattergeo(
#                     locationmode='USA-states',
#                     lon=edges_gdf['lons'][i],
#                     lat=edges_gdf['lats'][i],
#                     hoverinfo='skip',
#                     mode='lines',
#                     line=dict(width=1,
#                               color=purple),
#                     opacity=0.5,
#                     legendgroup=lg_group,
#                     name=legend_names[lg_group - 1],
#                     showlegend=legend_bool[lg_group - 1]
#                 ),
#                 row=row, col=col
#             )
#             legend_bool[lg_group - 1] = False
#     # sum up flows from both directions for plotting on undirected edges
#     edge_set_undirected = {(u, v) for u, v in edges_gdf.index}.union({(v, u) for u, v in edges_gdf.index})
#     edges_gdf.fillna(0, inplace=True)
#     for u, v in edge_set_undirected:
#         tot_tons = edges_gdf.loc[(u, v), 'hydrogen_avg_ton'][comm_group] + \
#                    edges_gdf.loc[(v, u), 'hydrogen_avg_ton'][comm_group]
#         edges_gdf.loc[(u, v), 'total_tons'] = tot_tons
#         edges_gdf.loc[(v, u), 'total_tons'] = tot_tons
#     edges_gdf.sort_values(by=['total_tons'], inplace=True)
#     for u, v in edges_gdf.index:
#         e = edges_gdf.loc[(u, v)]
#         e_rev = edges_gdf.loc[(v, u)]
#         e_tot_hydro_avg_ton = round(e['hydrogen_avg_ton'][comm_group] + e_rev['hydrogen_avg_ton'][comm_group])
#         e_tot_hydro_avg_loc = round(e['hydrogen_avg_loc'][comm_group] + e_rev['hydrogen_avg_loc'][comm_group])
#         e_tot_all_avg_ton = (e['hydrogen_avg_ton'][comm_group] + e['support_diesel_avg_ton'][comm_group] +
#                              e_rev['hydrogen_avg_ton'][comm_group] + e['support_diesel_avg_ton'][comm_group])
#         if np.floor(e_tot_all_avg_ton) == 0:
#             e_weight_hydrogen_perc_ton = 0
#         else:
#             e_weight_hydrogen_perc_ton = round(100 * e_tot_hydro_avg_ton / e_tot_all_avg_ton, 2)
#         if e['covered'] == 1:
#             text = str(round(e['miles'])) + ' miles' + '<br>' + \
#                    str(e_tot_hydro_avg_ton) + ' ' + comm_group.capitalize() + ' tons/day' + '<br>' + \
#                    str(e_tot_hydro_avg_loc) + ' ' + comm_group.capitalize() + ' loc/day' + '<br>' + \
#                    'Share of ' + comm_group.lower() + ' tons moved by hydrogen: ' + \
#                    str(e_weight_hydrogen_perc_ton) + '%'
#             lg_group = 1
#             fig.add_trace(
#                 go.Scattergeo(
#                     locationmode='USA-states',
#                     lon=e['lons'],
#                     lat=e['lats'],
#                     hovertemplate=text,
#                     mode='lines',
#                     line=dict(
#                         width=line_size(e_tot_all_avg_ton),
#                         color=green4,
#                     ),
#                     # opacity=1,
#                     legendgroup=lg_group,
#                     name=legend_names[lg_group - 1],
#                     showlegend=legend_bool[lg_group - 1]
#                 ),
#                 row=row, col=col
#             )
#             legend_bool[lg_group - 1] = False
#
#     legend_bool = [True, True, True]
#     for i in range(len(nodes_gdf)):
#         n = nodes_gdf.loc[nodes_gdf.index[i]]
#         # plot baseline (all nodes) first
#         lg_group = 2
#         fig.add_trace(
#             go.Scattergeo(
#                 locationmode='USA-states',
#                 lon=[n['geometry'].x],
#                 lat=[n['geometry'].y],
#                 hoverinfo='skip',
#                 marker=dict(size=1,
#                             color=purple,
#                             symbol='triangle-up'),
#                 legendgroup=lg_group,
#                 name=legend_names[lg_group - 1],
#                 showlegend=legend_bool[1]
#             ),
#             row=row, col=col
#         )
#         legend_bool[lg_group - 1] = False
#         if n['facility'] == 1:
#             if n['avg']['energy_transfer'] == 1:
#                 avg_pumped_kgh2 = -n['avg']['daily_demand_kgh2']
#                 peak_pumped_kgh2 = -n['peak']['daily_demand_kgh2']
#             else:
#                 avg_pumped_kgh2 = n['avg']['daily_supply_kgh2']
#                 peak_pumped_kgh2 = n['peak']['daily_supply_kgh2']
#             text = n['city'] + ', ' + n['state'] + '<br>' + \
#                    str(round(avg_pumped_kgh2, 2)) + ' kgh2/day <br>' + \
#                    str(int(n['avg']['number_loc'])) + ' loc/day <br>' + \
#                    str(n['energy_source_TEA']['number_pumps']) + ' pumps <br>' + \
#                    'Avg. Queue Time: ' + str(round(n['energy_source_TEA']['avg_queue_time_p_loc'], 2)) + ' hrs <br>' + \
#                    'Avg. Queue Length: ' + str(round(n['energy_source_TEA']['avg_queue_length'], 2)) + ' loc <br>' + \
#                    'Peak Queue Time: ' + str(round(n['energy_source_TEA']['peak_queue_time_p_loc'], 2)) + \
#                    ' hrs <br>' + \
#                    'Peak Queue Length: ' + str(round(n['energy_source_TEA']['peak_queue_length'], 2)) + ' loc <br>' + \
#                    'Utilized ' + str(round(n['energy_source_TEA']['actual_utilization'] * 24, 1)) + ' hrs/day <br>' + \
#                    'Total LCO: ' + str(round(n['energy_source_TEA']['total_LCO'], 3)) + ' $/kgh2 <br>' + \
#                    'WTW Emissions: ' + str(round(n['energy_source_LCA']['emissions_tonco2_kgh2'] * 1e6, 3)) + \
#                    ' g CO<sub>2</sub>' + '/kgh2 <br>' + \
#                    'Capital Cost: \t $' + str(round(n['energy_source_TEA']['station_total'] / 1e6, 2)) + ' M <br>' + \
#                    'Avg. Delay cost per car: \t $' + \
#                    str(round(n['energy_source_TEA']['avg_daily_delay_cost_p_car'], 2)) + '<br>' + \
#                    'Avg. Daily Delay cost per loc: \t $' + \
#                    str(round(n['energy_source_TEA']['avg_daily_delay_cost_p_loc'], 2)) + '<br>' + \
#                    'Total Daily Delay Cost: \t $' + \
#                    str(round(n['energy_source_TEA']['total_daily_delay_cost'] / 1e3, 2)) + ' K<br>'
#             #  + str(round(1e-3 * (n['energy_source_TEA']['total_LCO'] -
#             #                   n['diesel_TEA']['total_LCO_tonmi']) /
#             #           (n['energy_source_LCA']['avg_emissions_tonco2_tonmi'] -
#             #            n['diesel_LCA']['total_emissions_tonco2_tonmi']), 3)) + \
#
#             # str(round(n['peak']['daily_supply_mwh'], 2)) + ' MWh/day' + '<br>' + \
#             # str(int(n['peak']['number_loc'])) + ' loc/day at peak' + '<br>' + \
#             # str(round(-n['peak']['daily_demand_mwh'], 2)) + ' MWh at peak demand' + '<br>' + \
#             # str(round(-n['avg']['daily_demand_mwh'], 2)) + ' MWh at avg demand'
#             lg_group = 1
#             fig.add_trace(
#                 go.Scattergeo(
#                     locationmode='USA-states',
#                     lat=[n['geometry'].y],
#                     lon=[n['geometry'].x],
#                     # hoverinfo='text',
#                     # text=text,
#                     hovertemplate=text,
#                     marker=dict(
#                         # want no bigger than size 50; simple scaling (since min is always 0 here)
#                         # size=50 * (n['facility_size'] -
#                         #            min(nodes_gdf['facility_size'])) / (max(nodes_gdf['facility_size']) -
#                         #                                                min(nodes_gdf['facility_size'])),
#                         size=5 * np.log(peak_pumped_kgh2 + 10),
#                         # size=30,
#                         color=green4,
#                         sizemode='area',
#                     ),
#                     opacity=0.8,
#                     legendgroup=lg_group,
#                     name='Refueling Facility',
#                     showlegend=legend_bool[2]
#                 ),
#                 row=row, col=col
#             )
#             legend_bool[2] = False
#         elif n['covered'] == 1:
#             text = n['city'] + ', ' + n['state']
#             lg_group = 1
#             fig.add_trace(
#                 go.Scattergeo(
#                     locationmode='USA-states',
#                     lon=[n['geometry'].x],
#                     lat=[n['geometry'].y],
#                     hovertemplate=text,
#                     marker=dict(size=4,
#                                 color=green4,
#                                 symbol='square'),
#                     legendgroup=lg_group,
#                     name='Covered (Non-Refueling) Facility',
#                     showlegend=legend_bool[lg_group - 1]
#                 ),
#                 row=row, col=col
#             )
#             legend_bool[lg_group - 1] = False
#
#     if additional_plots:
#         if figlist:
#             fig.update_geos(projection_type="albers usa")
#             fig_title = ''
#             fig.update_layout(
#                 autosize=True,
#                 margin=dict(l=0, r=0, b=300, t=0, pad=1),
#                 showlegend=True,
#                 legend=dict(
#                     itemsizing='trace',
#                     orientation='h',
#                     yanchor="top",
#                     y=0.95,
#                     xanchor="center",
#                     x=0.5,
#                     font=dict(size=12, color=black)
#                 ),
#             )
#             for annotation in fig.layout.annotations:
#                 annotation.font.size = 12
#
#             figs = [fig]
#             # table for summary statistics
#             figs.append(hydrogen_summary_table(G=G, comm_group=comm_group))
#             # bar charts for LCA stats
#             figs.append(hydrogen_lca_plot(G=G, comm_group=comm_group))
#             # bar charts for TEA stats
#             figs.append(hydrogen_tea_plot(G=G, comm_group=comm_group))
#             # pie charts for operational stats
#             figs.append(hydrogen_pie_operations_plot(G=G, comm_group=comm_group))
#             # table for cost of avoided emissions
#             figs.append(hydrogen_cost_avoided_table(G=G, comm_group=comm_group))
#             fig = figs
#             labels = ['Map', 'Table', 'Emissions', 'Cost', 'Ton-Miles', 'CAE']
#         else:
#             # pie charts for operational stats
#             fig = hydrogen_pie_operations_plot(G, comm_group, fig)
#             # bar charts for LCA stats
#             fig = hydrogen_lca_plot(G, comm_group, fig)
#             # bar charts for TEA stats
#             fig = hydrogen_tea_plot(G, comm_group=comm_group, fig=fig)
#             # table for summary statistics
#             fig = hydrogen_summary_table(G, comm_group, fig)
#             # table for cost of avoided emissions
#             fig = hydrogen_cost_avoided_table(G, comm_group, fig)
#
#             fig.update_geos(projection_type="albers usa")
#             fig_title = ''
#             fig.update_layout(title=dict(text=fig_title, font=dict(color='black', size=6)), font=dict(color='black'),
#                               showlegend=True,
#                               legend=dict(
#                                   yanchor="top",
#                                   y=0.99,
#                                   xanchor="right",
#                                   x=0.99,
#                               ),
#                               # margin=dict(l=0, r=0, t=0, b=0),
#                               )
#             for annotation in fig.layout.annotations:
#                 annotation.font.size = 12
#
#             labels = []
#     else:
#         labels = []
#
#     # if figshow:
#     #     fig.show()
#
#     return fig, labels


def hydrogen_plot(G, comm_group: str, additional_plots=True, crs='WGS84', figlist=False, fig=None, legend_show=True):
    if fig is None:
        if additional_plots and not figlist:
            fig = make_subplots(
                rows=4, cols=3,
                specs=[[None, None, {"type": "scattergeo", "rowspan": 4}],
                       [{"type": "xy", 'secondary_y': True}, {"type": "bar"}, None],
                       [{"type": "domain"}, {"type": "table"}, None],
                       [{"type": "table", 'colspan': 2}, None, None]],
                column_widths=[0.1, 0.1, 0.8],
                row_heights=[0.02, 0.25, 0.25, 0.48],
                horizontal_spacing=0.1,
                subplot_titles=(None,
                                'WTW Emissions', 'Levelized Cost of Operation',
                                comm_group.capitalize() + ' Ton-Miles', None,
                                None)
            )
            row = 1
            col = 3
        else:
            t0 = time.time()
            fig = base_plot(G.graph['railroad'])
            print('LOAD BASE PLOT:: ' + str(time.time() - t0))
            row = 1
            col = 1

    nodes_gdf, edges_gdf = gdfs_from_graph(G, crs=crs, smooth_geometry=False)

    t0 = time.time()
    # drop non-covered edges
    edges_gdf.drop(index=edges_gdf[edges_gdf['covered'] == 0].index, inplace=True)
    # keep only these cols
    agg_cols = {'miles': 'first', 'geometry': 'first',
                'hydrogen_avg_ton': 'sum', 'hydrogen_avg_loc': 'sum',
                'support_diesel_avg_ton': 'sum'}
    edges_gdf.drop(columns=set(edges_gdf.columns).difference(set(agg_cols.keys())), inplace=True)
    # convert cols from dict to float values for the <comm_group> provided
    dict_cols = ['hydrogen_avg_ton', 'hydrogen_avg_loc', 'support_diesel_avg_ton']
    for col in dict_cols:
        edges_gdf[col] = edges_gdf[col].apply(lambda x: x[comm_group])

    # create a dict to map {(u, v): (u, v), (v, u): (u, v)}
    edge_mapper = dict()
    for u, v in edges_gdf.index:
        if (u, v) not in edge_mapper.keys():
            edge_mapper[u, v] = (u, v)
            edge_mapper[v, u] = (u, v)
    # map indices
    edges_gdf.rename(index=edge_mapper, inplace=True)
    edges_gdf.fillna(0, inplace=True)
    # groupby (u, v), summing values of 'hydrogen_avg_ton', 'hydrogen_avg_loc', 'support_diesel_avg_ton'
    edges_gdf.groupby(by=['u', 'v']).agg(agg_cols)
    # compute 'share_hydrogen'
    edges_gdf['share_hydrogen'] = 100 * edges_gdf['hydrogen_avg_ton'].div(edges_gdf['support_diesel_avg_ton'] +
                                                                          edges_gdf['hydrogen_avg_ton']).replace(np.inf,
                                                                                                                 0.00)
    edges_gdf['share_hydrogen'] = edges_gdf['share_hydrogen'].replace(np.NAN, 0.00)

    # assign line width to each edge based on hydrogen flow tonnage
    edges_gdf['line_width'] = edges_gdf['hydrogen_avg_ton'].apply(lambda x: line_size(x))
    # reset index
    edges_gdf.reset_index(inplace=True)
    # groupby line_width groups and (u, v)
    edges_gdf = edges_gdf.groupby(by=['line_width', 'u', 'v']).first()
    # get line widths
    line_widths = sorted(list(set(edges_gdf.index.get_level_values('line_width'))))

    legend_name = 'Hydrogen Network'
    lg_group = 1

    for lw in line_widths:
        e = edges_gdf.loc[lw, slice(None, None)]
        lats = []
        lons = []
        names = []
        for u, v in e.index:
            x, y = e.loc[(u, v), 'geometry'].xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            name = '{v1} miles <br>{v2} {v3} tons/day <br>{v4} {v5} loc/day <br>' \
                   'Share of {v6} tons moved by hydrogen: {v7}%'.format(v1=round(e.loc[(u, v), 'miles']),
                                                                        v2=round(e.loc[(u, v), 'hydrogen_avg_ton']),
                                                                        v3=comm_group.capitalize(),
                                                                        v4=round(e.loc[(u, v), 'hydrogen_avg_loc']),
                                                                        v5=comm_group.capitalize(),
                                                                        v6=comm_group.lower(),
                                                                        v7=round(e.loc[(u, v), 'share_hydrogen']))
            names = np.append(names, [name] * len(y))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            names = np.append(names, None)

        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(
                    width=lw,
                    color=green4,
                ),
                opacity=1,
                hoverinfo="text",
                hovertext=names,
                legendgroup=lg_group,
                name=legend_name,
                showlegend=lw == 2,
                connectgaps=False,
            )
        )

    print('\t EDGES:: ' + str(time.time() - t0))

    legend_bool = [True, True]
    for i in range(len(nodes_gdf)):
        n = nodes_gdf.loc[nodes_gdf.index[i]]
        if n['facility'] == 1:
            if n['avg']['energy_transfer'] == 1:
                avg_pumped_kgh2 = -n['avg']['daily_demand_kgh2']
                peak_pumped_kgh2 = -n['peak']['daily_demand_kgh2']
            else:
                avg_pumped_kgh2 = n['avg']['daily_supply_kgh2']
                peak_pumped_kgh2 = n['peak']['daily_supply_kgh2']
            text = n['city'] + ', ' + n['state'] + '<br>' + \
                   str(round(avg_pumped_kgh2, 2)) + ' kgh2/day <br>' + \
                   str(int(n['avg']['number_loc'])) + ' loc/day <br>' + \
                   str(n['energy_source_TEA']['number_pumps']) + ' pumps <br>' + \
                   'Avg. Queue Time: ' + str(
                round(n['energy_source_TEA']['avg_queue_time_p_loc'], 2)) + ' hrs <br>' + \
                   'Avg. Queue Length: ' + str(round(n['energy_source_TEA']['avg_queue_length'], 2)) + ' loc <br>' + \
                   'Peak Queue Time: ' + str(round(n['energy_source_TEA']['peak_queue_time_p_loc'], 2)) + \
                   ' hrs <br>' + \
                   'Peak Queue Length: ' + str(
                round(n['energy_source_TEA']['peak_queue_length'], 2)) + ' loc <br>' + \
                   'Utilized ' + str(
                round(n['energy_source_TEA']['actual_utilization'] * 24, 1)) + ' hrs/day <br>' + \
                   'Total LCO: ' + str(round(n['energy_source_TEA']['total_LCO'], 3)) + ' $/kgh2 <br>' + \
                   'WTW Emissions: ' + str(round(n['energy_source_LCA']['emissions_tonco2_kgh2'] * 1e6, 3)) + \
                   ' g CO<sub>2</sub>' + '/kgh2 <br>' + \
                   'Capital Cost: \t $' + str(round(n['energy_source_TEA']['station_total'] / 1e6, 2)) + ' M <br>' + \
                   'Avg. Delay cost per car: \t $' + \
                   str(round(n['energy_source_TEA']['avg_daily_delay_cost_p_car'], 2)) + '<br>' + \
                   'Avg. Daily Delay cost per loc: \t $' + \
                   str(round(n['energy_source_TEA']['avg_daily_delay_cost_p_loc'], 2)) + '<br>' + \
                   'Total Daily Delay Cost: \t $' + \
                   str(round(n['energy_source_TEA']['total_daily_delay_cost'] / 1e3, 2)) + ' K<br>'

            fig.add_trace(
                go.Scattergeo(
                    uid=n['nodeid'],
                    locationmode='USA-states',
                    lon=[n['geometry'].x],
                    lat=[n['geometry'].y],
                    hovertemplate=text,
                    marker=dict(
                        size=5 * np.log(peak_pumped_kgh2 + 10),
                        # size=30,
                        color=green4,
                        sizemode='area',
                    ),
                    opacity=0.8,
                    legendgroup=lg_group,
                    name='Refueling Facility',
                    showlegend=legend_bool[0],
                ),
                # row=row, col=col
            )

            legend_bool[0] = False
        elif n['covered'] == 1:
            text = n['city'] + ', ' + n['state']
            lg_group = 1
            fig.add_trace(
                go.Scattergeo(
                    uid=n['nodeid'],
                    locationmode='USA-states',
                    lon=[n['geometry'].x],
                    lat=[n['geometry'].y],
                    hovertemplate=text,
                    marker=dict(size=4,
                                color=green4,
                                symbol='square'),
                    legendgroup=lg_group,
                    name='Covered (Non-Refueling) Facility',
                    showlegend=legend_bool[1],
                ),
                # row=row, col=col
            )
            legend_bool[1] = False

    if additional_plots:
        if figlist:
            fig.update_geos(projection_type="albers usa")
            fig.update_layout(
                autosize=True,
                margin=dict(l=0, r=0, b=300, t=0, pad=1),
                showlegend=legend_show,
                legend=dict(
                    itemsizing='trace',
                    orientation='h',
                    yanchor="top",
                    y=0.95,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, color=black)
                ),
            )
            for annotation in fig.layout.annotations:
                annotation.font.size = 12

            figs = [fig]
            # table for summary statistics
            figs.append(hydrogen_summary_table(G=G, comm_group=comm_group))
            # bar charts for LCA stats
            figs.append(hydrogen_lca_plot(G=G, comm_group=comm_group, legend_show=legend_show))
            # bar charts for TEA stats
            figs.append(hydrogen_tea_plot(G=G, comm_group=comm_group, legend_show=legend_show))
            # pie charts for operational stats
            figs.append(hydrogen_pie_operations_plot(G=G, comm_group=comm_group))
            # table for cost of avoided emissions
            figs.append(hydrogen_cost_avoided_table(G=G, comm_group=comm_group))
            fig = figs
        else:
            # pie charts for operational stats
            fig = hydrogen_pie_operations_plot(G=G, comm_group=comm_group, fig=fig)
            # bar charts for LCA stats
            fig = hydrogen_lca_plot(G=G, comm_group=comm_group, fig=fig, legend_show=legend_show)
            # bar charts for TEA stats
            fig = hydrogen_tea_plot(G=G, comm_group=comm_group, fig=fig, legend_show=legend_show)
            # table for summary statistics
            fig = hydrogen_summary_table(G=G, comm_group=comm_group, fig=fig)
            # table for cost of avoided emissions
            fig = hydrogen_cost_avoided_table(G=G, comm_group=comm_group, fig=fig)

            fig.update_geos(projection_type="albers usa")
            fig_title = ''
            fig.update_layout(title=dict(text=fig_title, font=dict(color='black', size=6)), font=dict(color='black'),
                              showlegend=legend_show,
                              legend=dict(
                                  yanchor="top",
                                  y=0.99,
                                  xanchor="right",
                                  x=0.99,
                              ),
                              # margin=dict(l=0, r=0, t=0, b=0),
                              )
            for annotation in fig.layout.annotations:
                annotation.font.size = 12

    return fig


'''
DROP-IN PLOT
'''


def dropin_pie_operations_plot(G, fuel_type: str, deployment_perc: float, comm_group: str, fig=None):
    f_mapper = {'biodiesel': 'Biofuel', 'e-fuel': 'e-fuel'}

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "domain"}]]
        )
        row = 1
        col = 1
    else:
        row = 3
        col = 1

    if fuel_type == 'diesel':
        # labels = ['Diesel']
        # values = [sum([G.edges[u, v]['baseline_avg_ton']['TOTAL'] for u, v in G.edges]) * 365 / 1e6]
        # fig.add_trace(
        #     go.Pie(
        #         labels=labels,
        #         values=values,
        #         marker=dict(colors=[green, blue]),
        #         textinfo='label+percent',
        #         textposition='inside',
        #         hovertemplate='<b>%{label}</b> <br> %{value:.0f} [M ton/yr]',
        #         name='',
        #         showlegend=False
        #     ),
        #     row=2, col=1
        # )

        labels = ['Diesel']
        values = [G.graph['operations']['baseline_total_annual_tonmi'][comm_group] / 1e6]
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=[purple]),
                textinfo='label+percent',
                textposition='inside',
                hovertemplate='<b>%{label}</b> <br> %{value:.0f} [M ton/yr]',
                name='',
                showlegend=False
            ),
            row=row, col=col
        )
    else:
        # labels = ['Diesel', fuel_type.capitalize()]
        # values = [sum([G.edges[u, v]['baseline_avg_ton']['TOTAL'] *
        #                (1 - deployment_perc) for u, v in G.edges]) * 365 / 1e6,
        #           sum([G.edges[u, v]['baseline_avg_ton']['TOTAL'] * deployment_perc for u, v in G.edges]) * 365 / 1e6]
        # fig.add_trace(
        #     go.Pie(
        #         labels=labels,
        #         values=values,
        #         marker=dict(colors=[blue, green]),
        #         textinfo='label+percent',
        #         textposition='inside',
        #         hovertemplate='<b>%{label}</b> <br> %{value:.0f} [M ton/yr]',
        #         name='',
        #         showlegend=False
        #     ),
        #     row=2, col=1
        # )

        labels = ['Diesel', f_mapper[fuel_type]]
        values = [G.graph['operations']['baseline_total_annual_tonmi'][comm_group] * (1 - deployment_perc) / 1e6,
                  G.graph['operations']['baseline_total_annual_tonmi'][comm_group] * deployment_perc / 1e6]
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=[purple, green4]),
                textinfo='label+percent',
                textposition='inside',
                hovertemplate='<b>%{label}</b> <br> %{value:.0f} [M ton/yr]',
                name='',
                showlegend=False
            ),
            row=row, col=col
        )

    fig.update_layout(font_color=black,
                      autosize=True,
                      margin=dict(l=30, r=30, b=0, t=20, pad=1),
                      legend=dict(
                          orientation='h',
                          yanchor='bottom',
                          y=0,
                          xanchor='center',
                          x=0.5
                      ),
                      title=dict(
                          text=comm_group.capitalize() + ' Ton-Miles',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(t=15, b=5)
                      )
                      )

    return fig


def dropin_tea_plot(G, fuel_type: str, comm_group: str, fig=None, legend_show=True):
    # compute aggregate statistics to plot and be able to compare between battery and diesel
    f_mapper = {'biodiesel': 'Biofuel', 'e-fuel': 'e-fuel'}

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "bar"}]]
        )
        row = 1
        col = 1
    else:
        row = 2
        col = 2

    if fuel_type == 'diesel':
        fig.add_trace(
            go.Bar(x=['Diesel'],
                   y=[G.graph['energy_source_TEA']['fuel_LCO_tonmi'][comm_group] * 100],
                   name='Fuel',
                   hovertemplate='%{y:.2f} [¢/ton-mi]',
                   # showlegend=False,
                   marker=dict(color=purple)
                   ),
            row=row, col=col
        )

    else:
        fig.add_trace(
            go.Bar(x=['Diesel', f_mapper[fuel_type]],
                   y=[G.graph['diesel_TEA']['fuel_LCO_tonmi'][comm_group] * 100,
                      G.graph['energy_source_TEA']['fuel_LCO_tonmi'][comm_group] * 100],
                   name='Fuel',
                   hovertemplate='%{y:.2f} [¢/ton-mi]',
                   # showlegend=False,
                   marker=dict(color=green4)
                   ),
            row=row, col=col
        )

        # Scenario average
        fig.add_shape(type='line',
                      xref='paper', yref='y',
                      x0=-.5, y0=G.graph['energy_source_TEA']['total_LCO_tonmi'][comm_group] * 100,
                      x1=1.5, y1=G.graph['energy_source_TEA']['total_LCO_tonmi'][comm_group] * 100,
                      line=dict(color=black, width=2, dash='dash'),
                      opacity=1,
                      name='Scenario Average',
                      row=row, col=col
                      )
        fig.add_trace(
            go.Scatter(x=[f_mapper[fuel_type]],
                       y=[G.graph['energy_source_TEA']['total_LCO_tonmi'][comm_group] * 100],
                       mode='markers',
                       marker=dict(symbol='line-ew', size=6, line_width=2, opacity=1, color=black, line_color=black),
                       # opacity=1,
                       name='Scenario <br> Average',
                       hovertemplate='%{y:.2f} [¢/ton-mi]',
                       # showlegend=False
                       ),
            row=row, col=col
        )
        # fig.add_shape(type='line',
        #               xref='paper', yref='y',
        #               x0=-0.50, y0=G.graph['energy_source_TEA']['total_LCO_tonmi'][comm_group] * 100,
        #               x1=1.5, y1=G.graph['energy_source_TEA']['total_LCO_tonmi'][comm_group] * 100,
        #               line=dict(color=black, width=2, dash='dash'),
        #               opacity=1,
        #               row=row, col=col
        #               )
        # fig.add_annotation(hovertext=str(round(G.graph['energy_source_TEA']['total_LCO_tonmi'][comm_group] * 100, 2)) +
        #                              ' [¢/ton-mi]',
        #                    text='Scenario Average',
        #                    xref='paper', yref='y',
        #                    x=1.5, y=G.graph['energy_source_TEA']['total_LCO_tonmi'][comm_group] * 100,
        #                    xanchor='left', yanchor='middle',
        #                    arrowcolor=black,
        #                    font=dict(color=black),
        #                    showarrow=False,
        #                    row=row, col=col
        #                    )

    fig.update_layout(barmode='stack',
                      font=dict(color=black, size=12),
                      autosize=True,
                      margin=dict(l=20, r=0, b=0, t=50, pad=0),
                      showlegend=legend_show,
                      legend=dict(
                          orientation='h',
                          yanchor='top',
                          y=1,
                          xanchor='right',
                          x=1.9,
                          font=dict(size=10)
                      ),
                      title=dict(
                          text='Levelized Cost <br> of Operation',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(t=15, b=5)
                      )
                      )
    # update yaxis properties
    fig.update_yaxes(title=dict(text='[¢ / ton-mi]', standoff=10,
                                font=dict(size=10)), showgrid=False, row=row, col=col)

    return fig


def dropin_lca_plot(G, fuel_type: str, comm_group: str, fig=None, legend_show=True):
    # compute aggregate statistics to plot and be able to compare between battery and diesel

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "xy", 'secondary_y': True}]]
        )
        row = 1
        col = 1
    else:
        row = 2
        col = 1

    if fuel_type == 'diesel':
        # ton CO2
        fig.add_trace(
            go.Bar(x=['Diesel'],
                   y=[G.graph['energy_source_LCA']['annual_total_emissions_tonco2'][comm_group] / 1000],
                   name='Diesel',
                   hovertemplate='%{y:.2f} [kton CO<sub>2</sub>/yr]',
                   # showlegend=False,
                   marker=dict(color=purple)
                   ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=['Diesel'],
                       y=[1e6 * G.graph['energy_source_LCA']['total_emissions_tonco2_tonmi'][comm_group]],
                       mode='markers',
                       marker=dict(symbol='diamond', size=10, color=light_red),
                       name='WTW Emissions',
                       hovertemplate='%{y:.2f} [g CO<sub>2</sub>/ton-mi]',
                       showlegend=False
                       ),
            secondary_y=True,
            row=row, col=col
        )

        # update yaxis properties
        fig.update_yaxes(title=dict(text='[kton CO<sub>2</sub> / yr]', standoff=10,
                                    font=dict(size=10)), secondary_y=False, showgrid=False, row=row, col=col)
        fig.update_yaxes(title=dict(text='[g CO<sub>2</sub> / ton-mi]', standoff=10,
                                    font=dict(size=10, color=light_red)),
                         tickfont=dict(color=light_red), showgrid=False,
                         range=[0, np.ceil(1e6 *
                                           G.graph['energy_source_LCA']['total_emissions_tonco2_tonmi'][comm_group]
                                           / 10) * 10],
                         secondary_y=True, row=row, col=col)
    else:
        # ton CO2
        fig.add_trace(
            go.Bar(x=['100% Diesel', 'Scenario'],
                   y=[G.graph['diesel_LCA']['annual_total_emissions_tonco2'][comm_group] / 1000,
                      (G.graph['energy_source_LCA']['annual_total_emissions_tonco2'][comm_group] -
                       G.graph['energy_source_LCA']['annual_fuel_emissions_tonco2'][comm_group]) / 1000],
                   name='Diesel',
                   hovertemplate='%{y:.2f} [kton CO<sub>2</sub>/yr]',
                   # showlegend=False,
                   marker=dict(color=purple)
                   ),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(x=['100% Diesel', 'Scenario'],
                   y=[0,
                      G.graph['energy_source_LCA']['annual_fuel_emissions_tonco2'][comm_group] / 1000],
                   name=fuel_type,
                   hovertemplate='%{y:.2f} [kton CO<sub>2</sub>/yr]',
                   # showlegend=False,
                   marker=dict(color=green4)
                   ),
            secondary_y=False,
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=['100% Diesel', 'Scenario'],
                       y=[1e6 * G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][comm_group],
                          1e6 * G.graph['energy_source_LCA']['total_emissions_tonco2_tonmi'][comm_group]],
                       mode='markers',
                       marker=dict(symbol='diamond', size=10, color=light_red),
                       name='WTW Emissions Rate',
                       hovertemplate='%{y:.2f} [g CO<sub>2</sub>/ton-mi]',
                       showlegend=False
                       ),
            secondary_y=True,
            row=row, col=col,
        )

        # update yaxis properties
        fig.update_yaxes(title=dict(text='[kton CO<sub>2</sub> / yr]', standoff=10,
                                    font=dict(size=10)), secondary_y=False, showgrid=False, row=row, col=col)
        fig.update_yaxes(title=dict(text='[g CO<sub>2</sub> / ton-mi]', standoff=10,
                                    font=dict(size=10, color=light_red)),
                         tickfont=dict(color=light_red), showgrid=False,
                         range=[0, np.ceil(1e6 * G.graph['diesel_LCA']['total_emissions_tonco2_tonmi'][comm_group]
                                           / 10) * 10],
                         secondary_y=True, row=row, col=col)

    fig.update_layout(barmode='stack',
                      font_color=black,
                      autosize=True,
                      margin=dict(l=0, r=0, b=0, t=50, pad=1),
                      showlegend=legend_show,
                      legend=dict(
                          orientation='h',
                          yanchor='top',
                          y=1.2,
                          xanchor='center',
                          x=0.5,
                          font=dict(size=10)),
                      title=dict(
                          text='WTW Emissions',
                          font=dict(size=16, color=black),
                          y=1,
                          x=0.5,
                          yanchor='top',
                          xanchor='center',
                          pad=dict(b=30)
                      )
                      )

    return fig


def dropin_summary_table(G: nx.DiGraph, fuel_type: str, comm_group: str, fig=None):
    # plot table of summary of results for battery

    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "table"}]]
        )
        row = 1
        col = 1
    else:
        row = 4
        col = 1

    if fuel_type == 'diesel':
        fig.add_trace(
            go.Table(
                header=dict(values=['Scenario Summary', 'Statistics'],
                            font=dict(size=14),
                            line_color=black,
                            fill_color=mid_purple,
                            ),
                cells=dict(values=[['Total Annual Operating Cost',
                                    ],
                                   [str(round(G.graph['energy_source_TEA']['annual_total_cost'][comm_group] / 1e6, 2)) +
                                    ' [$M]',
                                    ]],
                           font=dict(size=12),
                           line_color=black,
                           fill_color=light_purple,
                           )
            ),
            row=row, col=col
        )

    else:

        fig.add_trace(
            go.Table(
                header=dict(values=['Scenario Summary', 'Statistics'],
                            font=dict(size=14),
                            line_color=black,
                            fill_color=mid_purple,
                            ),
                cells=dict(values=[['Total Annual Operating Cost',
                                    comm_group.capitalize() + ' Emissions Reduction'
                                    if G.graph['operations']['emissions_change'][comm_group] >= 0
                                    else ' Emissions Increase',
                                    ],
                                   [str(round(G.graph['energy_source_TEA']['annual_total_cost'][comm_group] / 1e6, 2)) +
                                    ' [$M]',
                                    str(round(abs(G.graph['operations']['emissions_change'][comm_group]), 2)) + ' %',
                                    ]],
                           font=dict(size=12),
                           line_color=black,
                           fill_color=light_purple,
                           )
            ),
            row=row, col=col
        )

    fig.update_layout(font_color=black,
                      autosize=True,
                      margin=dict(l=5, r=5, b=5, t=5, pad=1)
                      )

    return fig


def dropin_cost_avoided_table(G, fuel_type: str, comm_group: str, fig=None):
    if fig is None:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "table"}]]
        )
        row = 1
        col = 1
    else:
        row = 3
        col = 2

    if fuel_type != 'diesel':
        fig.add_trace(
            go.Table(
                header=dict(values=['Cost of Avoided Emissions'],
                            font=dict(size=14),
                            line_color=black,
                            fill_color=mid_purple,
                            ),
                cells=dict(values=[[str(round(G.graph['operations']['cost_avoided_emissions'][comm_group], 2)) +
                                    ' [$/kg CO<sub>2</sub>]'
                                    ]],
                           font=dict(size=12),
                           line_color=black,
                           fill_color=light_purple,
                           )
            ),
            row=row, col=col
        )
        fig.update_layout(font_color=black,
                          autosize=True,
                          margin=dict(l=5, r=5, b=5, t=5, pad=1)
                          )

    return fig


def dropin_plot(G, fuel_type: str, deployment_perc: float, comm_group: str, additional_plots=True,
                crs='WGS84', figlist=False, fig=None, legend_show=True):
    # perform plotting fdor dropin fuels, showing emissions by thickness and
    # aggregate plots displaying relevant summary of results, make commodity specific

    if fig is None:
        if additional_plots and not figlist:
            if fuel_type == 'diesel':
                fig = make_subplots(
                    rows=4, cols=3,
                    specs=[[None, None, {"type": "scattergeo", "rowspan": 4}],
                           [{"type": "xy", 'secondary_y': True}, {"type": "bar"}, None],
                           [{"type": "domain"}, None, None],
                           [{"type": "table", 'colspan': 2}, None, None]],
                    column_widths=[0.1, 0.1, 0.8],
                    row_heights=[0.02, 0.25, 0.25, 0.48],
                    horizontal_spacing=0.1,
                    subplot_titles=(None,
                                    'WTW Emissions', 'Levelized Cost of Operation',
                                    'Ton-Miles', None,
                                    None)
                )
            else:
                fig = make_subplots(
                    rows=4, cols=3,
                    specs=[[None, None, {"type": "scattergeo", "rowspan": 4}],
                           [{"type": "xy", 'secondary_y': True}, {"type": "bar"}, None],
                           [{"type": "domain"}, {"type": "table"}, None],
                           [{"type": "table", 'colspan': 2}, None, None]],
                    column_widths=[0.1, 0.1, 0.8],
                    row_heights=[0.02, 0.25, 0.25, 0.48],
                    horizontal_spacing=0.1,
                    subplot_titles=(None,
                                    'WTW Emissions', 'Levelized Cost of Operation',
                                    'Ton-Miles', None,
                                    None)
                )
            row = 1
            col = 3
        else:
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"type": "scattergeo"}]]
            )
            row = 1
            col = 1

    nodes_gdf, edges_gdf = gdfs_from_graph(G, crs=crs, smooth_geometry=False)

    t0 = time.time()
    # keep only these cols
    agg_cols = {'miles': 'first', 'geometry': 'first',
                'baseline_avg_ton': 'sum', 'baseline_avg_loc': 'sum'}
    edges_gdf.drop(columns=set(edges_gdf.columns).difference(set(agg_cols.keys())), inplace=True)
    # convert cols from dict to float values for the <comm_group> provided
    dict_cols = ['baseline_avg_ton', 'baseline_avg_loc']
    for col in dict_cols:
        edges_gdf[col] = edges_gdf[col].apply(lambda x: x[comm_group])

    # create a dict to map {(u, v): (u, v), (v, u): (u, v)}
    edge_mapper = dict()
    for u, v in edges_gdf.index:
        if (u, v) not in edge_mapper.keys():
            edge_mapper[u, v] = (u, v)
            edge_mapper[v, u] = (u, v)
    # map indices
    edges_gdf.rename(index=edge_mapper, inplace=True)
    edges_gdf.fillna(0, inplace=True)
    # groupby (u, v), summing values of 'baseline_avg_ton', 'baseline_avg_loc'
    edges_gdf.groupby(by=['u', 'v']).agg(agg_cols)
    # assign line width to each edge based on flow tonnage
    edges_gdf['line_width'] = edges_gdf['baseline_avg_ton'].apply(lambda x: line_size(x))
    # reset index
    edges_gdf.reset_index(inplace=True)
    # groupby line_width groups and (u, v)
    edges_gdf = edges_gdf.groupby(by=['line_width', 'u', 'v']).first()
    # get line widths
    line_widths = sorted(list(set(edges_gdf.index.get_level_values('line_width'))))

    lg_group = 1

    for lw in line_widths:
        e = edges_gdf.loc[lw, slice(None, None)]
        lats = []
        lons = []
        names = []
        for u, v in e.index:
            x, y = e.loc[(u, v), 'geometry'].xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            name = '{v1} miles <br>{v2} {v3} tons/day <br>{v4} {v5} loc/day'.format(
                v1=round(e.loc[(u, v), 'miles']),
                v2=round(e.loc[(u, v), 'baseline_avg_ton']),
                v3=comm_group.capitalize(),
                v4=round(e.loc[(u, v), 'baseline_avg_loc']),
                v5=comm_group.capitalize())
            names = np.append(names, [name] * len(y))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            names = np.append(names, None)

        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(
                    width=lw,
                    color=green4,
                ),
                opacity=1,
                hoverinfo="text",
                hovertext=names,
                legendgroup=lw if lw > 0.2 else 0.5,
                name=line_label_from_size(lw),
                showlegend=True,
                connectgaps=False,
            )
        )

    print('\t EDGES:: ' + str(time.time() - t0))

    # END NEW

    if additional_plots:
        if figlist:
            fig.update_geos(projection_type="albers usa")
            fig.update_layout(
                autosize=True,
                margin=dict(l=0, r=0, b=300, t=0, pad=1),
                showlegend=legend_show,
                legend=dict(
                    title_text='Tons per Day: ',
                    itemsizing='trace',
                    orientation='h',
                    yanchor="top",
                    y=0.95,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, color=black)
                ),
            )

            for annotation in fig.layout.annotations:
                annotation.font.size = 12

            figs = [fig]
            # table for summary statistics
            figs.append(dropin_summary_table(G=G, fuel_type=fuel_type, comm_group=comm_group))
            # bar charts for LCA stats
            figs.append(dropin_lca_plot(G=G, fuel_type=fuel_type, comm_group=comm_group, legend_show=legend_show))
            # bar charts for TEA stats
            figs.append(dropin_tea_plot(G=G, fuel_type=fuel_type, comm_group=comm_group, legend_show=legend_show))
            # pie charts for operational stats
            figs.append(dropin_pie_operations_plot(G=G, fuel_type=fuel_type, deployment_perc=deployment_perc,
                                                   comm_group=comm_group))
            # table for cost of avoided emissions
            figs.append(dropin_cost_avoided_table(G=G, fuel_type=fuel_type, comm_group=comm_group))
            fig = figs
        else:
            # pie charts for operational stats
            fig = dropin_pie_operations_plot(G, fuel_type, deployment_perc, comm_group, fig)
            # bar charts for LCA stats
            fig = dropin_lca_plot(G, fuel_type, comm_group, fig, legend_show=legend_show)
            # bar charts for TEA stats
            fig = dropin_tea_plot(G, fuel_type, comm_group, fig, legend_show=legend_show)
            # table for summary statistics
            fig = dropin_summary_table(G, fuel_type, comm_group, fig)
            # table for cost of avoided emissions
            fig = dropin_cost_avoided_table(G, fuel_type, comm_group, fig)

            fig.update_geos(projection_type="albers usa")
            fig_title = ''
            fig.update_layout(title=dict(text=fig_title, font=dict(color='black', size=6)), font=dict(color='black'),
                              showlegend=legend_show,
                              legend_title_text='Tons per Day [2019]')

            for annotation in fig.layout.annotations:
                annotation.font.size = 12

    else:
        fig.update_geos(projection_type="albers usa")
        fig_title = ''
        fig.update_layout(title=dict(text=fig_title, font=dict(color='black', size=6)), font=dict(color='black'),
                          showlegend=legend_show,
                          legend_title_text='Tons per Day [2019]')

        for annotation in fig.layout.annotations:
            annotation.font.size = 12

    return fig


'''
BASE PLOT
'''


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


'''
FORMATTING
'''


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
