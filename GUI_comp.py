from input_output import load_scenario_metrics
from util import *
import plotly.express as px

# global color assignment
blue, red, orange, green, purple, teal, pink, oli33ve, darkred, darkblue = plotly.colors.qualitative.G10
color_plate = [blue, red, orange, green, purple, teal, pink, oli33ve, darkred, darkblue]

# Default plot groups
plot_group = ["energy_source_LCA", "energy_source_TEA", "operations"]

# Default plot lists
plot_list = {"energy_source_LCA": ["annual_total_emissions_tonco2"],
             "energy_source_TEA": ["total_LCO_tonmi", "annual_total_cost"],
             "operations": ["cost_avoided_emissions"],
             }

# Default plot names
plot_list_name = {"annual_total_emissions_tonco2": "Annual Total Emission <br>(ton CO2 eqv)",
                  "total_LCO_tonmi": "Total LCO ($/ton-mile)",
                  "annual_total_cost": "Annual Total Cost ($)",
                  "cost_avoided_emissions": "Cost of Avoided Emission <br>($/kg CO2 eqv)",
                  "daily_fuel_cost": "Daily Fuel Cost ($)",
                  "annual_fuel_cost": "Annual Fuel Cost ($)",
                  }


# Return metrics for the corresponding comp_list
def comparison_chart_metrics(comp_list) -> list:
    metrics = []
    for comp_scen in comp_list:
        metrics.append(load_scenario_metrics(comp_scen))
    return metrics


# Return scenario details
def return_comparison_scneario_detail(comp_list):
    metrics = comparison_chart_metrics(comp_list)
    return [m["scenario"] for m in metrics]


# plot table of summary of results for battery
def plot_scneario_table(comp_list: list):
    metrics = comparison_chart_metrics(comp_list)
    metric_scenario = return_comparison_scneario_detail(comp_list)

    header = pd.DataFrame([str(i + 1) for i in range(len(comp_list))])
    header = pd.concat([pd.DataFrame(['Scenario']), header])

    scenario_label = pd.DataFrame([['Railroad'],
                                   ['Range (mi)'],
                                   ['Energy Source'],
                                   ['Deployment %'],
                                   # ['Re-routing'],
                                   # ['Switching'],
                                   # ['Re-routing Max %'],
                                   # ['Max utilization %'],
                                   ],
                                  )
    scenario_list = pd.DataFrame([[str(s["railroad"]) for s in metric_scenario],
                                  [str(s["range_mi"]) for s in metric_scenario],
                                  [str(s["fuel_type"]) for s in metric_scenario],
                                  # TODO: to update to commodity-specific
                                  [str(int(m["scenario"]["desired_deployment_perc"]["TOTAL"] * 100)) + ' %' \
                                       if m["scenario"]["fuel_type"] != "battery" \
                                       else str(int(m["operations"]["deployment_perc"]["TOTAL"] * 100)) + ' %' \
                                   for m in metrics
                                   ],
                                  # [str(s["reroute"]) for s in metric_scenario],
                                  # [str(s["switch_tech"]) for s in metric_scenario],
                                  # [str(int(s["max_reroute_inc"] * 100)) + ' %' for s in metric_scenario],
                                  # [str(int(s["max_util"] * 100)) + ' %' for s in metric_scenario],
                                  ])
    cells = pd.concat([scenario_label, scenario_list], axis=1).transpose()

    # Define table colors to match with charts
    fill_color = []
    n = len(comp_list)
    for col in range(n + 1):
        if col == 0:
            fill_color.append("#000000")
        else:
            fill_color.append(color_plate[col - 1])

    fig = go.Figure(data=[
        go.Table(
            header=dict(values=header,
                        font=dict(color="white", size=11),
                        fill_color=fill_color
                        ),
            cells=dict(values=cells,
                       font=dict(size=11),
                       # fill_color=fill_color
                       )
        )
    ])

    # fig.update_layout(width=300, height=500)

    return fig


# Return figures
# TODO: Handle no plot_list_name situation
def plot_comparison_chart(comp_list,
                          plot_group: list = plot_group,
                          plot_field: list = plot_list,
                          plot_list_name: list = plot_list_name,
                          print_all: bool = False):
    metrics = comparison_chart_metrics(comp_list)
    if not metrics:
        return None
    else:
        if not plot_group:
            plot_group_check = metrics[0].keys()
        else:
            plot_group_check = plot_group
        output_fig = []
        for pg in plot_group_check:
            # check if it is on all metrics list
            check_pg_in_all = True
            for m in metrics:
                if pg not in m.keys():
                    check_pg_in_all = False
                    break
            if check_pg_in_all:
                if plot_field:
                    plot_field_list = plot_field[pg]
                else:
                    plot_field_list = metrics[0][pg]

                if isinstance(plot_field_list, list):
                    for pf in plot_field_list:
                        # check if it is on all metrics list
                        check_pf_in_all = True
                        for m in metrics:
                            if pf not in m[pg]:
                                check_pf_in_all = False
                                break
                        if check_pf_in_all:
                            # print(pg,pf)
                            if pf in plot_list_name.keys():
                                pf_name = plot_list_name[pf]
                                output_fig.append(plot_bar_metric(metrics, pg, pf, pf_name))
                            else:
                                if print_all:
                                    pf_name = pf
                                    output_fig.append(plot_bar_metric(metrics, pg, pf, pf_name))

    return output_fig


# Return a figure plotting plot_field across the metrics
def plot_bar_metric(metrics, plot_group: str, plot_field: str, plot_name: str):
    to_plot = []
    N = len(metrics)
    for m in metrics:
        # print(plot_group,plot_field)
        to_plot.append(m[plot_group][plot_field]["TOTAL"])
    # TODO: to update to commodity-specific

    fig = go.Figure([
        go.Bar(x=[i + 1 for i in range(N)],
               y=to_plot,
               name='plot_field',
               # hovertemplate='%{y:.2f} [¢/ton-mi]',
               showlegend=False,
               # marker=dict(color=red),
               marker_color=color_plate
               ),
    ])
    fig.update_xaxes(dtick=1)
    fig.update_layout(title_text=plot_name, xaxis={'tickformat': ',d'})
    return fig
