from plotting import *
from run import run_scenario
from input_output import write_scenario_df
from GUI_comp import *

# This code file is for all dash components; for plotly components refer to plotting.py

# Whether this version is for public access
public_access = False

# Whether this allows remote access
remote_access = False

# Set up logging
if public_access:
    logging.basicConfig(filename='GUI_exe_pub.log', encoding='utf-8', level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
else:
    logging.basicConfig(filename='GUI_exe.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')

# Setting up template
# Load template explicitly for graph plotting
template_code = "pulse"
# Update customized codes in assets\custom.css
external_stylesheets = [
    "assets\\bootstrap.min.css",
    "assets\\custom.css",
    # dbc.themes.PULSE
]
load_figure_template(template_code)  # override plotly default template

# Google Analytics
if public_access:
    external_scripts = ["assets\\gtag.js", "https://www.googletagmanager.com/gtag/js?id=G-JQYB04RJK0"]
else:
    external_scripts = []

# General Dashboard
if public_access:
    RR_list = ['WCAN', 'EAST', 'USA1']
    RR_list_value = "WCAN"
    RR_name_selected = RR_list_value
    RR_list_show = ['Western & Canadian', 'Eastern', 'Transcontinental']
else:
    RR_list = ['BNSF', 'CN', 'CP', 'CSXT', 'KCS', 'NS', 'UP', 'WCAN', 'EAST', 'USA1']
    RR_list_value = "KCS"
    RR_name_selected = RR_list_value
    RR_list_show = ['BNSF', 'CN', 'CP', 'CSXT', 'KCS', 'NS', 'UP', 'Western & Canadian', 'Eastern', 'USA']
range_list = [200, 400, 800, 1000, 1500]
range_list_value = 400
RR_range_selected = range_list_value
ES_list = ['diesel', 'biodiesel', 'e-fuel', "battery", "hydrogen"]
ES_list_value = 'battery'
ES_name_selected = ES_list_value
ES_list_show = ['Diesel', 'Biofuel', 'e-fuel', "Battery-electric", "Hydrogen"]
# db_list = ['General', 'Comparison', 'Parameters']
db_list = ['General', 'Parameters']
db_list_value = 'General'
legend_switch_value = True
legend_switch_selected = legend_switch_value

comm_list = ['TOTAL', 'AG_FOOD', 'CHEM_PET', 'COAL', 'FOR_PROD', 'MET_ORE', 'MO_VEH', 'NONMET_PRO', 'IM', 'OTHER']
comm_list_value = 'TOTAL'
comm_name_selected = comm_list_value
comm_list_show = ['All', 'Agricultural & Food', "Chemicals & Petroleum", "Coal", "Forest Product", "Metal and Ores",
                  "Motor Vehicles", "Non-metallic Products", 'Intermodal', 'Others']
deploy_pct_list = [0.2, 0.4, 0.6, 0.8, 1]
deploy_pct_value = 0.6
deploy_pct_selected = deploy_pct_value
current_scenario = ()
last_n_run_click = 0  # to track how many times run button is clicked
last_n_rerun_click = 0  # to track how many times rerun button is clicked
last_n_comp_click = 0  # to track how many times add to compare button is clicked
last_n_comp_clear_click = 0  # to track how many times clear comparison button is clicked

# Parameter Dashboard
charger_speed_list = [3]
charger_speed_value = 3
charger_speed_list_value = charger_speed_value
charger_no_hour_list = [20]
charger_no_hour_value = 20
charger_no_hour_list_value = charger_no_hour_value
battery_capa_list = [10]
battery_capa_value = 10
battery_capa_list_value = battery_capa_value
battery_PPA_switch_value: bool = False
battery_ppa_cost_box_value = 0
year_list = [2035]
year_value = 2035
year_list_value = year_value
discount_rate_list = [7]
discount_rate_value = 7
discount_rate_list_value = discount_rate_value
hydrogen_no_hour_list = [20]
hydrogen_no_hour_value = 20
hydrogen_no_hour_list_value = hydrogen_no_hour_value
hydrogen_dispense_list = ['Cryo-pump', '700 via pump', '350 via pump']
hydrogen_dispense_list_show = [
    '10 bar LH2 dispensing',
    '700 bar via LH2 pump/vaporization',
    '350 bar via LH2 pump/vaporization',
]
hydrogen_PPA_cost_box_value = 0
hydrogen_fuel_list = ['Natural Gas', 'NG with CO2 Sequestration', 'PEM Electrolysis - Solar',
                      'PEM Electrolysis - Nuclear']
hydrogen_fuel_list_value = 'Natural Gas'
hydrogen_dispense_value = 'Cryo-pump'
hydrogen_dispense_list_value = hydrogen_dispense_value
re_routing_switch_value: bool = True
loco_switching_switch_value: bool = False

# Comparison dashboard
comp_list = []

# Define dash object with template
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)
app.title = 'NUFRIEND Dashboard'

# Authentication
if remote_access and not public_access:
    # Loading login
    login_filename = "locologin.csv"
    VALID_USERNAME_PASSWORD_PAIRS = {}

    with open(login_filename, 'r') as login_data:
        for l in csv.DictReader(login_data):
            VALID_USERNAME_PASSWORD_PAIRS.update(l)

    auth = dash_auth.BasicAuth(
        app,
        VALID_USERNAME_PASSWORD_PAIRS
    )

app.config.suppress_callback_exceptions = True

# banner
header = html.Img(src=app.get_asset_url('header_light.png'), style={'width': '100%'})

# initial screen
initial_text = dcc.Markdown('''NUFRIEND is a
comprehensive industry-oriented tool to simulate the deployment of new energy technologies across U.S. freight
rail networks. Scenario-specific simulation and optimization modules provide estimates for carbon reductions,
capital investments, costs of carbon reductions, and operational impacts for any given deployment profile.
\n
For more details, please refer to https://www.transportation.northwestern.edu/research/featured-reports/locomotives.html.
\n 
Please select the scenario and click \"Run\". Initial run may take up to 30 seconds.
''')

# disclaimer
disclaimer = dcc.Markdown('''
    This dashboard is developed by Northwestern University Transportation Center 
    (NUTC: Hani S. Mahmassani, Pablo Durango-Cohen, Adrian Hernandez, Max Ng) 
    and Argonne National Laboratory (ANL: Amgad Elgowainy, Michael Wang, Joann Zhou, Nazib Siddique)
    for Advanced Research Projects Agency - Energy (ARPA-E)  LOwering CO2: Models to Optimize Train Infrastructure, Vehicles, and Energy Storage (LOCOMOTIVES) project. \n
    All results shown here are preliminary and for demonstration only. They are based on pre-defined sets of scenarios and assumptions. The parameter pane is currently for illustration only.\n
    The user guide is available here: https://www.transportation.northwestern.edu/docs/research/featured-reports/nufriend_dashboard_user_guide.pdf. \n
    For more details, please refer to https://www.transportation.northwestern.edu/research/featured-reports/locomotives.html. \n
    For questions regarding the dashboard, please contact Max Ng (maxng@u.northwestern.edu) and Adrian Hernandez (AdrianHernandez2025@u.northwestern.edu).
''')

RR_dropdown = html.Div(
    [
        dbc.Label("Railroad"),
        dcc.Dropdown([{'value': x, 'label': y}
                      for x, y in zip(RR_list, RR_list_show)],
                     RR_list_value,
                     id='RR_dropdown',
                     persistence=True, persistence_type="memory", ),
    ], className="dash-bootstrap"
)

ES_dropdown = html.Div(
    [
        dbc.Label("Energy"),
        dcc.Dropdown([{'value': x, 'label': y}
                      for x, y in zip(ES_list, ES_list_show)],
                     ES_list_value,
                     id='ES_dropdown',
                     persistence=True, persistence_type="memory"),
    ]
)

# Slider for range selection
range_slider = html.Div(
    [
        dbc.Collapse(
            dbc.Row([
                dbc.Label("Battery Range (mi)", html_for="slider"),
                dcc.Slider(0, 1500,
                           step=None,
                           marks={
                               x: str(x) for x in range_list
                           },
                           value=range_list_value,
                           id="range_slider",
                           persistence=True, persistence_type="memory",
                           )
            ]),
            id="battery_range_collapse",
            is_open=True,
        )
    ],
    className="mb-3",
)

# Slider for deployment % selection
deploy_pct_slider = html.Div(
    [
        dbc.Label("Target Deployment % (Overall)", html_for="slider"),
        dcc.Slider(0, 1,
                   step=None,
                   marks={
                       x: str(int(x * 100)) + '%' for x in deploy_pct_list
                   },
                   value=deploy_pct_value,
                   id="deploy_pct_slider",
                   persistence=True, persistence_type="memory",
                   )
    ],
    className="mb-3",
)

comm_dropdown = html.Div(
    [
        dbc.Label("Commodity"),
        dcc.Dropdown([{'value': x, 'label': y}
                      for x, y in zip(comm_list, comm_list_show)],
                     comm_list_value,
                     id='comm_dropdown',
                     persistence=True, persistence_type="memory",
                     ),
    ]
)

# Legend switch
legend_switch = html.Div(
    [
        dbc.Switch(
            # options=[
            #     {"label": "Re-routing allowed", "value": 1},
            # ],
            label="Show legend",
            value=legend_switch_value,
            id="legend_switch",
            persistence=True, persistence_type="memory",
            # switch=True,
        ),
    ]
)

# Loading spinner
loading_spinner = html.Div(
    [
        dbc.Spinner(html.Div(id="loading-output")),
    ]
)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
RR_map = dls.Hash(
    dbc.Row(children="", id="RR_basemap"),
    color="#435278",
    speed_multiplier=2,
    size=100,
)

# Radio buttons for RR selection
db_radio = html.Div([
    dbc.Label("Dashboard", html_for="radio"),
    dbc.RadioItems(
        options=[{'value': x, 'label': x}
                 for x in db_list],
        value=db_list_value,
        id="db_menu",
        className="btn-group",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary",
        labelCheckedClassName="active",
        persistence=True, persistence_type="memory",
    ),
]
)

# re-load button
run_button = html.Div(
    [
        dbc.Button("Run",
                   color="primary",
                   className="me-1",
                   id="run_button",
                   n_clicks=0,
                   ),
        dls.Hash(
            html.Div(children="", id="cache_output"),
            color="#435278",
            speed_multiplier=2,
            size=100,
        )
    ]
)

# re-load button
rerun_button = html.Div(
    dbc.Collapse(
        dbc.Row([
            dbc.Button("Re-run",
                       color="primary",
                       className="me-1",
                       id="rerun_button",
                       n_clicks=0,
                       ),
            dls.Hash(
                html.Div(children="", id="cache_output"),
                color="#435278",
                speed_multiplier=2,
                size=100,
            )
        ]),
        id="rerun_button_collapse",
        is_open=not public_access,
    )

)

# cache everything button
cache_everything_button = html.Div(
    dbc.Collapse(
        dbc.Row([
            dbc.Button("Cache Everything!",
                       color="primary",
                       className="me-1",
                       id="cache_everything_button",
                       n_clicks=0,
                       ),
            dls.Hash(
                html.Div(children="", id="cache_everything_output"),
                color="#435278",
                speed_multiplier=2,
                size=100,
                fullscreen=True,
            )
        ]),
        id="cache_everything_button_collapse",
        is_open=not remote_access,
    )
)

# add to comparison button
compare_button = html.Div(
    [
        dbc.Button("Add to compare",
                   color="primary",
                   className="me-1",
                   id="compare_button",
                   n_clicks=0,
                   ),
        html.Div(children="", id="compare_output")
    ]
)


# Return card rows with figure of cards for charts
def return_chart_fig_card(fig_list):
    card_list = []
    if not fig_list:
        return dbc.Row()
    else:
        n_card = len(fig_list) - 2
        no_col = 2  # Number of columns to show the cards (excluding map and metrics)
        width_total = 12
        width_map = 8
        height_total = 50

        card_list = []
        for i, fig in enumerate(fig_list[2:]):
            card = dbc.Card(
                [
                    dbc.CardBody(
                        [
                            dcc.Graph(figure=fig,
                                      id="card_" + str(i + 2),
                                      style={
                                          # 'width': '90vh',
                                          'height': str(height_total / no_col) + 'vh',
                                          #     'height': '20%',
                                      },
                                      # className="h-100",
                                      config={
                                          'displayModeBar': False
                                      }
                                      ),
                        ]
                    ),
                ],
                style={"border": "none"},
            )
            card_list.append(card)

        card_row_list = []

        i = 0
        temp_row = []
        while i < len(card_list):
            temp_row.append(
                dbc.Col(card_list[i], width=int((width_total) / int(n_card / no_col)))
            )
            if i % no_col == 1:
                card_row_list.append(temp_row)
                temp_row = []
            i += 1
        if temp_row:
            card_row_list.append(temp_row)

        # Combine cards of charts with the RR map
        metric_row_list = [
            dbc.Row(card_row, className="g-0") for card_row in card_row_list
        ]

        # Add metric table
        metric_row_list.append(dbc.Row(dbc.Card(
            [
                dbc.CardBody(
                    [
                        dcc.Graph(figure=fig_list[1],
                                  id="card_1",
                                  style={
                                      'height': str(height_total) + 'vh',
                                  },
                                  config={
                                      'displayModeBar': False
                                  }
                                  ),
                    ]
                ),
            ],
            className="g-0",
            style={"border": "none"},
        )))

        # Add card of main RR map
        output_row_list = [dbc.Col(metric_row_list, width=width_total - width_map),
                           dbc.Col(
                               dbc.Card([
                                   dbc.CardBody(
                                       [
                                           dcc.Graph(figure=fig_list[0],
                                                     id="card_0",
                                                     style={
                                                         'width': '100vh',
                                                         'height': '100vh',
                                                     }
                                                     ),
                                       ]
                                   ),
                               ]
                                   , style={"border": "none"},
                               )
                               , width=width_map

                           )
                           ]
        output_row = dbc.Row(output_row_list, className="g-0", )

        return output_row


# Return corresponding station_type for battery and hydrogen
def return_station_type(ES_name):
    if ES_name == "battery":
        return str(charger_speed_list_value) + "MW"
    elif ES_name == "hydrogen":
        return hydrogen_dispense_list_value
    else:
        return None


# @cache.memoize(timeout=timeout)
def return_RR_range_map(RR_name, ES_name, comm_name, RR_range, deploy_pct, load_scenario=True, cache_scenario=True,
                        plot=True, legend_show=True, station_type=None,
                        clean_energy=False, clean_energy_cost=0, h2_fuel_type=None,
                        reroute=re_routing_switch_value, switch_tech=loco_switching_switch_value, ):
    # TODO: <load_plot> currently useless, must decide whether se want this or not
    # <load_plot> = True - load the fig from the cached file if it exists, = False - rerun scenario and cache plot file
    _, fig, label = run_scenario(rr=RR_name, fuel_type=ES_name, comm_group=comm_name, D=RR_range * 1.6,
                                 deployment_perc=deploy_pct, cache_scenario=cache_scenario,
                                 reroute=reroute, switch_tech=switch_tech,
                                 clean_energy=clean_energy, clean_energy_cost=clean_energy_cost,
                                 h2_fuel_type=h2_fuel_type,
                                 plot=plot, load_scenario=load_scenario, legend_show=legend_show,
                                 station_type=station_type)

    return fig, label


# Read parameter changes
@app.callback(
    [Output("battery_range_collapse", "is_open"),
     ],
    [Input("RR_dropdown", "value"),
     Input("ES_dropdown", "value"),
     Input("comm_dropdown", "value"),
     Input("range_slider", "value"),
     Input("deploy_pct_slider", "value"),
     Input("legend_switch", "value"),
     ]
)
def read_parameter(RR_name, ES_name, comm_name, RR_range, deploy_pct, legend_show):
    global RR_name_selected, ES_name_selected, comm_name_selected, RR_range_selected, deploy_pct_selected, legend_switch_selected
    RR_name_selected, ES_name_selected, comm_name_selected, RR_range_selected, deploy_pct_selected, legend_switch_selected = RR_name, ES_name, comm_name, RR_range, deploy_pct, legend_show
    # Expand/Collapse battery range slider depending on ES
    if ES_name == 'battery':
        battery_range_collapse_open = True
    else:
        battery_range_collapse_open = False
    return [battery_range_collapse_open]


# Load / reload a scenario
@app.callback(
    [Output("RR_basemap", "children"),
     Output("cache_output", "children"),
     ],
    [
        Input("run_button", "n_clicks"),
        Input("rerun_button", "n_clicks"),
    ]
)
def display_RR_range_map(n_run, n_rerun):
    RR_name, ES_name, comm_name, RR_range, deploy_pct, legend_show = [RR_name_selected, ES_name_selected,
                                                                      comm_name_selected, RR_range_selected,
                                                                      deploy_pct_selected, legend_switch_selected]
    global last_n_run_click, last_n_rerun_click
    update_graph = False  # Flag whether to update a graph

    # Check run
    if (n_run <= last_n_run_click) or (n_run == 0):
        output = ""
    else:
        last_n_run_click = n_run
        update_graph = True
        output = "Run successful"

    # Check re-run
    if (n_rerun <= last_n_rerun_click) or (n_rerun == 0):
        load_scenario = True  # set to load from cache
        output = ""
    else:
        last_n_rerun_click = n_rerun
        load_scenario = False  # set NOT to load from cache
        update_graph = True
        output = "Re-run successful"

    if update_graph:
        current_scenario = RR_name, ES_name, comm_name, RR_range, deploy_pct, re_routing_switch_value, loco_switching_switch_value
        print(request.remote_addr, current_scenario)
        logging.info(request.remote_addr)  # IP address
        logging.info(current_scenario)  # Scenario
        station_type = return_station_type(ES_name)

        if ES_name == 'battery':
            clean_energy = battery_PPA_switch_value
            fig_list, label = return_RR_range_map(RR_name, ES_name, comm_name, RR_range, deploy_pct,
                                                  load_scenario=load_scenario, legend_show=legend_show,
                                                  clean_energy=clean_energy,
                                                  clean_energy_cost=battery_ppa_cost_box_value,
                                                  station_type=station_type, reroute=re_routing_switch_value,
                                                  switch_tech=loco_switching_switch_value)
        elif ES_name == 'hydrogen':
            clean_energy = hydrogen_fuel_list_value != "Natural Gas"
            fig_list, label = return_RR_range_map(RR_name, ES_name, comm_name, 0, deploy_pct,
                                                  load_scenario=load_scenario, legend_show=legend_show,
                                                  clean_energy=clean_energy,
                                                  clean_energy_cost=hydrogen_PPA_cost_box_value,
                                                  h2_fuel_type=hydrogen_fuel_list_value,
                                                  station_type=station_type, reroute=re_routing_switch_value,
                                                  switch_tech=loco_switching_switch_value)
        elif ES_name == 'diesel':
            fig_list, label = return_RR_range_map(RR_name, ES_name, comm_name, 0, 1, load_scenario=load_scenario,
                                                  legend_show=legend_show)
        else:
            fig_list, label = return_RR_range_map(RR_name, ES_name, comm_name, 0, deploy_pct,
                                                  load_scenario=load_scenario,
                                                  legend_show=legend_show, station_type=station_type,
                                                  reroute=re_routing_switch_value,
                                                  switch_tech=loco_switching_switch_value)

        return return_chart_fig_card(fig_list), output
    else:
        print(request.remote_addr)
        logging.info(request.remote_addr)  # IP address
        return html.Div(initial_text), output


# Add to comparison
@app.callback(
    Output("compare_output", "children"),
    [
        Input("compare_button", "n_clicks"),
    ]
)
def add_to_comparison(n):
    RR_name, ES_name, comm_name, RR_range, deploy_pct = RR_name_selected, ES_name_selected, comm_name_selected, RR_range_selected, deploy_pct_selected
    global last_n_comp_click
    if n > last_n_comp_click:
        if ES_name == 'battery':
            RR_range_0 = RR_range
            deploy_pct_0 = deploy_pct
        elif ES_name == 'diesel':
            RR_range_0 = 0
            deploy_pct_0 = 1
        else:
            RR_range_0 = 0
            deploy_pct_0 = deploy_pct
        last_n_comp_click = n

        scenario_code = write_scenario_df(rr=RR_name, fuel_type=ES_name, deployment_perc=deploy_pct_0,
                                          D=1.6 * RR_range_0, reroute=re_routing_switch_value,
                                          switch_tech=loco_switching_switch_value, max_reroute_inc=0.5,
                                          max_util=0.88, station_type=None,
                                          CCWS_filename=None,
                                          time_window=None, freq='M')

        if scenario_code not in comp_list:
            comp_list.append(scenario_code)
            print(comp_list)

        return "Added"
    else:
        return ""


error_list = []


# Caching every options
@app.callback(
    [Output("cache_everything_output", "children")],
    Input("cache_everything_button", "n_clicks"))
def cache_everything(n):
    if n == 1:
        for ES_name in ['hydrogen', "battery", 'diesel', 'biodiesel', 'e-fuel']:
            # for RR_name in ["BNSF", "UP", "NS", "KCS", "CSXT", "CN", "CP"]:
            for RR_name in RR_list:  # cache every RR
                if ES_name == 'battery':
                    for reroute in [True, False]:
                        for switch_tech in [True, False]:
                            for RR_range in range_list:  # iterate among all battery ranges
                                for deploy_pct in deploy_pct_list:
                                    print(RR_name, ES_name, RR_range, deploy_pct, reroute, switch_tech)
                                    try:
                                        return_RR_range_map(RR_name, ES_name, "TOTAL", RR_range, deploy_pct, plot=False,
                                                            reroute=reroute, switch_tech=switch_tech)
                                    except Exception as e:
                                        error_list.append(
                                            [RR_name, ES_name, "TOTAL", RR_range, deploy_pct, reroute, switch_tech,
                                             traceback.format_exc()])
                                        pd.DataFrame(data=error_list).to_csv(os.path.join(OUTPUT_DIR, 'error_list.csv'))
                                        continue
                elif ES_name == 'hydrogen':
                    for reroute in [True, False]:
                        for switch_tech in [True, False]:
                            for hydrogen_dispense in hydrogen_dispense_list:  # iterate among all H2 dispenser
                                for deploy_pct in deploy_pct_list:
                                    station_type = hydrogen_dispense
                                    print(RR_name, ES_name, 0, deploy_pct, station_type)
                                    try:
                                        return_RR_range_map(RR_name, ES_name, "TOTAL", 0, deploy_pct, plot=False,
                                                            station_type=station_type, reroute=reroute,
                                                            switch_tech=switch_tech)
                                    except Exception as e:
                                        error_list.append(
                                            [RR_name, ES_name, "TOTAL", RR_range, deploy_pct, station_type, reroute,
                                             switch_tech,
                                             traceback.format_exc()])
                                        pd.DataFrame(data=error_list).to_csv(os.path.join(OUTPUT_DIR, 'error_list.csv'))
                                        continue
                elif ES_name == 'diesel':
                    print(RR_name, ES_name, 0, 1)
                    try:
                        return_RR_range_map(RR_name, ES_name, "TOTAL", 0, 1, plot=False)
                    except Exception as e:
                        error_list.append([RR_name, ES_name, "TOTAL", 0, 1, traceback.format_exc()])
                        pd.DataFrame(data=error_list).to_csv(os.path.join(OUTPUT_DIR, 'error_list.csv'))
                        continue
                else:
                    for deploy_pct in deploy_pct_list:
                        print(RR_name, ES_name, 0, deploy_pct)
                        try:
                            return_RR_range_map(RR_name, ES_name, "TOTAL", 0, deploy_pct, plot=False)
                        except Exception as e:
                            error_list.append([RR_name, ES_name, "TOTAL", 0, deploy_pct, traceback.format_exc()])
                            pd.DataFrame(data=error_list).to_csv(os.path.join(OUTPUT_DIR, 'error_list.csv'))
                            continue
        return ["Cached!"]
    elif n == 0:
        return [""]
    else:
        return ["Cached!"]


general_dashboard = [
    dbc.Row([
        dbc.Col([dbc.Row(RR_dropdown),
                 dbc.Row(style={"height": "1vh"}),
                 dbc.Row(ES_dropdown),
                 dbc.Row(style={"height": "1vh"}),
                 dbc.Row(range_slider),
                 dbc.Row(style={"height": "1vh"}),
                 dbc.Row(deploy_pct_slider),
                 dbc.Row(style={"height": "1vh"}),
                 dbc.Row(comm_dropdown),
                 dbc.Row(style={"height": "1vh"}),
                 dbc.Row(legend_switch),
                 dbc.Row(style={"height": "1vh"}),
                 dbc.Row(run_button),
                 dbc.Row(style={"height": "1vh"}),
                 dbc.Row(rerun_button),
                 dbc.Row(style={"height": "1vh"}),
                 dbc.Row(cache_everything_button),
                 dbc.Row(style={"height": "1vh"}),
                 html.Div(children="Preliminary Results"),
                 ], width=2),
        dbc.Col([
            dbc.Row(RR_map),
            dbc.Row(loading_spinner),
        ], width=10),
    ],
    ),
    dbc.Row(style={"height": "1vh"}),
]

comp_action_button = html.Div(
    [
        dbc.Button("Compare!",
                   color="primary",
                   className="me-1",
                   id="compare_action_button",
                   n_clicks=0,
                   ),
        html.Div(children="", id="compare_action_output")
    ]
)

# add to comparison button
comp_clear_button = html.Div(
    [
        dbc.Button("Clear Comparison",
                   color="primary",
                   className="me-1",
                   id="comp_clear_button",
                   n_clicks=0,
                   ),
        html.Div(children="", id="comp_clear_output")
    ]
)


# TODO: Refresh comparison cards after cleared => combined function with comparison button
# Add to comparison
@app.callback(
    Output("comp_clear_output", "children"),
    Input("comp_clear_button", "n_clicks"),
)
def comp_clear_comparison(n):
    global last_n_comp_clear_click, comp_list
    if n > last_n_comp_clear_click:
        last_n_comp_clear_click = n
        comp_list = []
        return "Cleared"
    else:
        return ""


comp_scenario_table = dbc.Row(children=None, id="comp_scenario_text")

## Comparison Dashboard
comp_dashboard = [
    dbc.Row([
        dbc.Col([
            dbc.Row(comp_scenario_table),
            dbc.Row(comp_action_button),
            dbc.Row(comp_clear_button),
        ], width=3),
        dbc.Col([
            dbc.Row(children=None, id="card_row"),
        ], width=9),
    ]),
    dbc.Row(style={"height": "1vh"}),
]


# Return card rows with figure of cards
def return_fig_card(fig_list):
    card_list = []
    if not fig_list:
        return dbc.Row()
    else:
        for i, fig in enumerate(fig_list):
            card = dbc.Card(
                [
                    dbc.CardBody(
                        [
                            dcc.Graph(figure=fig,
                                      id="card_" + str(i),
                                      # style={
                                      # 'width': '90vh',
                                      # 'height': str(fig_height / 10) + 'vh',
                                      # }
                                      ),
                        ]
                    ),
                ],
                style={"width": "18rem"},
            )
            card_list.append(card)
        card_row = dbc.Row([
            card for card in card_list
        ])
        return card_row


# Triggered to show figures in card_row
@app.callback(
    [Output("card_row", "children"),
     Output("comp_scenario_text", "children")],
    Input("compare_action_button", "n_clicks")
)
def show_comp_card_row(n_clicks):
    scenario_table = dbc.Row(dcc.Graph(figure=plot_scneario_table(comp_list)))
    return return_fig_card(plot_comparison_chart(comp_list)), scenario_table


## Parameter Dashboard

# Slider for year
year_slider = html.Div(
    [
        dbc.Label("Year", html_for="slider"),
        dcc.Slider(2020, 2050,
                   step=None,
                   marks={
                       x: str(x) for x in year_list
                   },
                   value=year_list_value,
                   id="year_slider",
                   persistence=True, persistence_type="memory",
                   )
    ],
    className="mb-3",
)

# Slider for discount rate
discount_rate_slider = html.Div(
    [
        dbc.Label("Discount rate", html_for="slider"),
        dcc.Slider(0, 15,
                   step=None,
                   marks={
                       x: str(x) + '%' for x in discount_rate_list
                   },
                   value=discount_rate_list_value,
                   id="discount_rate_slider",
                   persistence=True, persistence_type="memory",
                   )
    ],
    className="mb-3",
)

# Slider for charger speed
charger_speed_slider = html.Div(
    [
        dbc.Label("Charger Speed", html_for="slider"),
        dcc.Slider(0, 5,
                   step=None,
                   marks={
                       x: str(x) + 'MW' for x in charger_speed_list
                   },
                   value=charger_speed_list_value,
                   id="charger_speed_slider",
                   persistence=True, persistence_type="memory",
                   )
    ],
    className="mb-3",
)

# Slider for charger number of hours
charger_no_hour_slider = html.Div(
    [
        dbc.Label("Max. Number of Charger Operating Hours", html_for="slider"),
        dcc.Slider(0, 24,
                   step=None,
                   marks={
                       x: str(x) for x in charger_no_hour_list
                   },
                   value=charger_no_hour_list_value,
                   id="charger_no_hour_slider",
                   persistence=True, persistence_type="memory",
                   )
    ],
    className="mb-3",
)

# Slider for battery capacity
battery_capa_slider = html.Div(
    [
        dbc.Label("Battery Capacity", html_for="slider"),
        dcc.Slider(0, 20,
                   step=None,
                   marks={
                       x: str(int(x)) + 'MWh' for x in battery_capa_list
                   },
                   value=battery_capa_list_value,
                   id="battery_capa_slider",
                   persistence=True, persistence_type="memory",
                   )
    ],
    className="mb-3",
)

# Battery clean energy premium
battery_PPA_box = html.Div(
    [
        dbc.Label("Electricity Clean Source Premium ($/kWh)"),
        dbc.Input(
            value=battery_ppa_cost_box_value,
            id="battery_PPA_box",
            type="number",
            persistence=True, persistence_type="memory",
        ),
    ]
)

# Battery clean energy switch
battery_PPA_switch = html.Div(
    [
        dbc.Switch(
            label="Electricity Clean Source",
            value=battery_PPA_switch_value,
            id="battery_PPA_switch",
            persistence=True, persistence_type="memory",
        ),
    ]
)

# Slider for hydrogen refueling number of hours
hydrogen_no_hour_slider = html.Div(
    [
        dbc.Label("Max. Number of Refueling Operating Hours", html_for="slider"),
        dcc.Slider(0, 24,
                   step=None,
                   marks={
                       x: str(x) for x in hydrogen_no_hour_list
                   },
                   value=hydrogen_no_hour_list_value,
                   id="hydrogen_no_hour_slider",
                   persistence=True, persistence_type="memory",
                   )
    ],
    className="mb-3",
)

# Hydrgen dispenser type dropdown
hydrogen_dispense_dropdown = html.Div(
    [
        dbc.Label("Dispensing Type"),
        dcc.Dropdown([{'value': x, 'label': y}
                      for x, y in zip(hydrogen_dispense_list, hydrogen_dispense_list_show)],
                     hydrogen_dispense_list_value,
                     id='hydrogen_dispense_dropdown',
                     persistence=True, persistence_type="memory",
                     ),
    ]
)

# Hydrogen clean energy premium
hydrogen_PPA_cost_box = html.Div(
    [
        dbc.Label("Fuel Premium ($/kg H2)"),
        dbc.Input(
            value=hydrogen_PPA_cost_box_value,
            id="hydrogen_PPA_cost_box",
            type="number",
            persistence=True, persistence_type="memory",
        ),
    ]
)

# Hydrogen fuel type dropdown
hydrogen_fuel_dropdown = html.Div(
    [
        dbc.Label("Fuel Type"),
        dcc.Dropdown([{'value': x, 'label': y}
                      for x, y in zip(hydrogen_fuel_list, hydrogen_fuel_list)],
                     hydrogen_fuel_list_value,
                     id='hydrogen_fuel_dropdown',
                     persistence=True, persistence_type="memory",
                     ),
    ]
)

# Re-routing allowed switch
re_routing_switch = html.Div(
    [
        dbc.Switch(
            # options=[
            #     {"label": "Re-routing allowed", "value": 1},
            # ],
            label="Re-routing allowed",
            value=re_routing_switch_value,
            id="re_routing_switch",
            persistence=True, persistence_type="memory",
            # switch=True,
        ),
    ]
)

# Loco-switching allowed switch
loco_switching_switch = html.Div(
    [
        dbc.Switch(
            # options=[
            #     {"label": "Locomotive switching allowed", "value": 1},
            # ],
            label="Locomotive switching allowed",
            value=loco_switching_switch_value,
            id="loco_switching_switch",
            persistence=True, persistence_type="memory",
            # switch=True,
        ),
    ]
)

dummy_output = dbc.Row(id="dummy_output")

para_dashboard = [
    dbc.Row(html.Strong("General")),
    dbc.Row([dbc.Col(year_slider, width=4),
             dbc.Col(discount_rate_slider, width=4),
             ]),
    dbc.Row(style={"height": "1vh"}),
    dbc.Row(html.Strong("Battery")),
    dbc.Row([dbc.Col(charger_speed_slider, width=3),
             dbc.Col(charger_no_hour_slider, width=3),
             dbc.Col(battery_capa_slider, width=2),
             dbc.Col(battery_PPA_switch, width=2),
             dbc.Col(battery_PPA_box, width=2),
             ]),
    dbc.Row(style={"height": "1vh"}),
    dbc.Row(html.Strong("Hydrogen")),
    dbc.Row([dbc.Col(hydrogen_no_hour_slider, width=4),
             dbc.Col(hydrogen_dispense_dropdown, width=3),
             dbc.Col(hydrogen_fuel_dropdown, width=3),
             dbc.Col(hydrogen_PPA_cost_box, width=2),
             ]),
    dbc.Row(html.Strong("Routing")),
    dbc.Row([dbc.Col(re_routing_switch, width=4),
             dbc.Col(loco_switching_switch, width=4),
             ]),

    dbc.Row(style={"height": "10vh"}),
    dbc.Row(dummy_output),
]


# Set parameters according to param dashboard
@app.callback(
    Output("dummy_output", 'children'),
    [Input("year_slider", "value"),
     Input("discount_rate_slider", "value"),
     Input("charger_speed_slider", "value"),
     Input("charger_no_hour_slider", "value"),
     Input("battery_capa_slider", "value"),
     Input("battery_PPA_switch", "value"),
     Input("battery_PPA_box", "value"),
     Input("hydrogen_no_hour_slider", "value"),
     Input("hydrogen_dispense_dropdown", "value"),
     Input("hydrogen_PPA_cost_box", "value"),
     Input("hydrogen_fuel_dropdown", "value"),
     Input("re_routing_switch", "value"),
     Input("loco_switching_switch", "value"),
     ],
)
def set_param(year_value, discount_rate_value, charger_speed_value, charger_no_hour_value,
              battery_capa_value, battery_PPA_switch, battery_PPA_value, hydrogen_no_hour_value,
              hydrogen_dispense_value,
              hydrogen_PPA_cost_value, hydrogen_fuel_value,
              re_routing_value, loco_switching_value):
    global year_list_value, discount_rate_list_value, charger_speed_list_value, charger_no_hour_list_value, \
        battery_capa_list_value, battery_PPA_switch_value, battery_ppa_cost_box_value, hydrogen_no_hour_list_value, hydrogen_dispense_list_value, \
        hydrogen_PPA_cost_box_value, hydrogen_fuel_list_value, \
        re_routing_switch_value, loco_switching_switch_value

    year_list_value, discount_rate_list_value, charger_speed_list_value, charger_no_hour_list_value, \
    battery_capa_list_value, battery_PPA_switch_value, battery_ppa_cost_box_value, hydrogen_no_hour_list_value, hydrogen_dispense_list_value, \
    hydrogen_PPA_cost_box_value, hydrogen_fuel_list_value, \
    re_routing_switch_value, loco_switching_switch_value = \
        year_value, discount_rate_value, charger_speed_value, charger_no_hour_value, \
        battery_capa_value, battery_PPA_switch, battery_PPA_value, hydrogen_no_hour_value, hydrogen_dispense_value, \
        hydrogen_PPA_cost_value, hydrogen_fuel_value, \
        re_routing_value, loco_switching_value

    return ""


# Dashboard tabs
db_tab = html.Div([
    dbc.Tabs([dbc.Tab(tab_id=x, label=x) for x in db_list],
             id="tabs", active_tab='General',
             ),
    dbc.Row(style={"height": "1vh"}),
    html.Div(id='tabs-content'),
])

# Combine all for display
app.layout = html.Div(
    [
        dbc.Row(dbc.Col(header)),
        dbc.Row(db_tab,
                # style={"height": "100%"}
                ),
        dbc.Row(disclaimer),
    ]
)


# Dashboard tab interaction
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'active_tab'))
def render_content(tab):
    if tab == 'General':
        return general_dashboard
    elif tab == "Comparison":
        return comp_dashboard
    elif tab == "Parameters":
        return para_dashboard
    else:
        return None


# Return flask app for external calling e.g., waitress
def GUI_app():
    global remote_access
    remote_access = True
    return app.server


def GUI_app_pub():
    global remote_access, public_access
    remote_access = True
    public_access = True
    return app.server


# Start dash
if __name__ == '__main__':
    if remote_access:
        # specify host='0.0.0.0' to allow remote connection
        if public_access:
            app.run_server(host='0.0.0.0', port=8000, debug=False)
        else:
            app.run_server(host='0.0.0.0', port=8080, debug=True)
    else:
        app.run_server(debug=True)


# To tell flask it is behind a proxy (to use with nginx)
@app.server.route("/")
def flask_proxy():
    app.server.wsgi_app = ProxyFix(
        app.server.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
    )


if remote_access:
    flask_proxy()
