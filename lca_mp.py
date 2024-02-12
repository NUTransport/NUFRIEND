"""
LCA
===

Life-cycle assessment calculations module. Conducts life-cycle assessment for an input networkx graph representing the
rail network and deployment scenario. All calculations are stored in updates made to the provided graph's attributes.
"""

from util import *
from helper import load_lca_battery_lookup, load_lca_hydrogen_lookup, load_lca_battery_lookup_mp

'''
BATTERY
'''


def lca_battery_mp(G: nx.DiGraph, time_horizon: list, clean_energy=False):
    for n in G:
        G.nodes[n]['energy_source_LCA'] = {t: dict() for t in time_horizon}

    G.graph['energy_source_LCA'] = {t: dict() for t in time_horizon}

    for t in time_horizon:
        G = lca_battery_step_mp(G=G, time_step=t, clean_energy=clean_energy)

    return G


def lca_battery_step_mp(G: nx.DiGraph, time_step: str, clean_energy=False):
    """
    Conducts emissions life-cycle analysis for all charging facilities in G based on their size, consumption,
    and location.

    Parameters
    ----------
    G : nx.DiGraph
        The networkx graph to add the emissions information to.
    clean_energy : bool, optional
        True if considering clean energy, False o.w.

    Returns
    -------
    nx.DiGraph
        Updated version of G, which includes all the relevant life-cycle analysis calculations.

    """

    # load dataframe for emissions factors: index is state, column is year, value is in [g CO2/kWh]
    df_e = load_lca_battery_lookup_mp(year=time_step).div(1e6)  # in [ton CO2/kWh]

    for n in G:
        if clean_energy:
            e_factor = 0
        else:
            e_factor = df_e.loc[G.nodes[n]['state'], 'emissions']
        G.nodes[n]['energy_source_LCA'][time_step] = {'emissions_tonco2_kwh': e_factor,
                                                      'daily_emissions_tonco2':
                                                          (1000 * e_factor *
                                                           G.nodes[n]['avg'][time_step]['daily_supply_mwh']),
                                                      'annual_total_emissions_tonco2':
                                                          (365 * 1000 * e_factor *
                                                           G.nodes[n]['avg'][time_step]['daily_supply_mwh'])}

    df_dropin = load_lca_dropin_lookup()
    diesel_factor = df_dropin.loc['diesel', 'gCO2/gal'] / 1e6  # in [ton-CO2/gal]
    # comm_list = ['AG_FOOD', 'CHEM_PET', 'COAL', 'FOR_PROD', 'IM', 'MET_ORE', 'MO_VEH', 'NONMET_PRO', 'OTHER']
    # comm_er = load_comm_energy_ratios()['Weighted ratio'][comm_list].to_numpy()  # commodity energy ratios
    # comm_tonmi_dict = {c: sum([G.edges[u, v]['baseline_avg_ton'][c] * G.edges[u, v]['miles'] for u, v in G.edges])
    #                    for c in comm_list}
    # total_tonmi = sum(comm_tonmi_dict.values())
    # comm_diesel_factor = diesel_factor * sum([comm_er[i] * comm_tonmi_dict[comm_list[i]] / total_tonmi
    #                                           for i in range(len(comm_list))])

    comm_list = list({c for u, v in G.edges for c in G.edges[u, v]['battery_avg_kwh'][time_step].keys()})

    # battery_annual_energy_kwh = G.graph['MCNF_avg']['total_energy_mwh'] * 1000 * 365
    battery_annual_energy_kwh = G.graph['operations']['alt_tech_total_annual_kwh'][time_step]
    battery_annual_tonmi = G.graph['operations']['alt_tech_total_annual_tonmi'][time_step]
    support_diesel_annual_tonmi = G.graph['operations']['support_diesel_total_annual_tonmi'][time_step]
    baseline_annual_tonmi = {c: battery_annual_tonmi[c] + support_diesel_annual_tonmi[c] for c in comm_list}
    support_diesel_annual_gal = G.graph['operations']['support_diesel_total_annual_gal'][time_step]
    # battery_annual_energy_kwh = {c: 365 * sum(G.edges[u, v]['battery_avg_kwh'][c] for u, v in G.edges)
    #                              for c in comm_list}
    # annual_new_total_ton_mi = G.graph['operations']['annual_new_total_ton_mi']
    # support_diesel_annual_tonmi = G.graph['operations']['annual_support_diesel_total_ton_mi']
    # battery_actual_annual_tonmi = annual_new_total_ton_mi - support_diesel_annual_tonmi
    # use the percentage of ton-mi increase to calculate all in terms of baseline ton-miles
    # tonmi_deflation_factor = {c: 1 - G.graph['operations']['perc_tonmi_inc'][c] / 100 for c in comm_list}
    # battery_annual_tonmi = {c: 365 * sum([G.edges[u, v]['battery_avg_ton'][c] * G.edges[u, v]['miles']
    #                                       for u, v in G.edges]) * tonmi_deflation_factor[c] for c in comm_list}
    # support_diesel_annual_tonmi = {c: 365 * sum([G.edges[u, v]['support_diesel_avg_ton'][c] * G.edges[u, v]['miles']
    #                                              for u, v in G.edges]) * tonmi_deflation_factor[c] for c in comm_list}
    # support_diesel_annual_gal = {c: 365 * sum([G.edges[u, v]['support_diesel_avg_gal'][c] for u, v in G.edges])
    #                              for c in comm_list}
    annual_battery_total_emissions = {c: sum([G.nodes[n]['energy_source_LCA'][time_step]
                                              ['annual_total_emissions_tonco2']
                                              for n in G]) * battery_annual_energy_kwh[c] /
                                         battery_annual_energy_kwh['TOTAL'] for c in comm_list}

    battery_avg_emissions_tonco2_kwh = annual_battery_total_emissions['TOTAL'] / battery_annual_energy_kwh['TOTAL']

    G.graph['energy_source_LCA'][time_step] = dict(
        annual_battery_total_emissions=annual_battery_total_emissions,
        annual_support_diesel_total_emissions=dict(zip(
            comm_list,
            [diesel_factor * support_diesel_annual_gal[c] for c in comm_list])),
        battery_emissions_tonco2_kwh={c: battery_avg_emissions_tonco2_kwh for c in comm_list},
        battery_emissions_tonco2_tonmi=dict(zip(
            comm_list,
            [battery_avg_emissions_tonco2_kwh * battery_annual_energy_kwh[c] / battery_annual_tonmi[c]
             if battery_annual_tonmi[c] > 0 else 0 for c in comm_list])),
        support_diesel_emissions_tonco2_tonmi=dict(zip(
            comm_list,
            [diesel_factor * support_diesel_annual_gal[c] / support_diesel_annual_tonmi[c]
             if support_diesel_annual_tonmi[c] > 0 else 0 for c in comm_list])),
        avg_emissions_tonco2_tonmi=dict(zip(
            comm_list,
            [(annual_battery_total_emissions[c] + diesel_factor * support_diesel_annual_gal[c]) /
             baseline_annual_tonmi[c] if baseline_annual_tonmi[c] > 0 else 0 for c in comm_list])),
        annual_total_emissions_tonco2=dict(zip(
            comm_list,
            [annual_battery_total_emissions[c] + diesel_factor * support_diesel_annual_gal[c] for c in comm_list]))
    )

    return G


'''
HYDROGEN
'''


def lca_hydrogen(G: nx.DiGraph, h2_fuel_type: str = 'Natural Gas'):
    """
    Conducts emissions life-cycle analysis for all refueling facilities in G based on their size, and consumption.

    Parameters
    ----------
    G : nx.DiGraph
        The networkx graph to add the emissions information to.
    h2_fuel_type : str, optional
        Type of hydrogen fuel pathway under consideration. Allowed values are:
        - 'Natural Gas'
        - 'NG with CO2 Sequestration'
        - 'PEM Electrolysis - Solar'
        - 'PEM Electrolysis - Nuclear'

    Returns
    -------
    nx.DiGraph
        Updated version of G, which includes all the relevant life-cycle analysis calculations.

    """
    # return G with node attribute with emissions info for hydrogen
    year = '2034'
    e_factor = load_lca_hydrogen_lookup().loc[h2_fuel_type, year] / 1e6  # tonCO2/kgh2
    G.graph['scenario']['h2_fuel_type'] = h2_fuel_type

    for n in G:
        # e_factor = df_e.loc[G.nodes[n]['state'], 'emissions']
        G.nodes[n]['energy_source_LCA'] = {'emissions_tonco2_kgh2': e_factor,
                                           'daily_emissions_tonco2':
                                               (e_factor * G.nodes[n]['avg']['daily_supply_kgh2']),
                                           'annual_total_emissions_tonco2':
                                               (365 * e_factor * G.nodes[n]['avg']['daily_supply_kgh2'])}

    comm_list = list({c for u, v in G.edges for c in G.edges[u, v]['hydrogen_avg_kgh2'].keys()})

    df_dropin = load_lca_dropin_lookup()
    diesel_factor = df_dropin.loc['diesel', 'gCO2/gal'] / 1e6  # in [ton-CO2/gal]
    # comm_list = ['AG_FOOD', 'CHEM_PET', 'COAL', 'FOR_PROD', 'IM', 'MET_ORE', 'MO_VEH', 'NONMET_PRO', 'OTHER']
    # comm_er = load_comm_energy_ratios()['Weighted ratio'][comm_list].to_numpy()  # commodity energy ratios
    # comm_tonmi_dict = {c: sum([G.edges[u, v]['baseline_avg_ton'][c] * G.edges[u, v]['miles'] for u, v in G.edges])
    #                    for c in comm_list}
    # total_tonmi = sum(comm_tonmi_dict.values())
    # comm_diesel_factor = diesel_factor * sum([comm_er[i] * comm_tonmi_dict[comm_list[i]] / total_tonmi
    #                                           for i in range(len(comm_list))])

    # hydrogen_annual_energy_kgh2 = G.graph['MCNF_avg']['total_energy_kgh2'] * 365
    # annual_new_total_ton_mi = G.graph['operations']['annual_new_total_ton_mi']
    # support_diesel_annual_tonmi = G.graph['operations']['annual_support_diesel_total_ton_mi']
    # battery_actual_annual_tonmi = annual_new_total_ton_mi - support_diesel_annual_tonmi
    # use the percentage of ton-mi increase to calculate all in terms of baseline ton-miles
    # tonmi_deflation_factor = (100 - G.graph['operations']['perc_tonmi_inc']) / 100
    # hydrogen_annual_tonmi = sum([G.edges[u, v]['hydrogen_avg_ton']['TOTAL'] * G.edges[u, v]['miles'] * 365
    #                              for u, v in G.edges]) * tonmi_deflation_factor
    # support_diesel_annual_tonmi = sum([G.edges[u, v]['support_diesel_avg_ton']['TOTAL'] * G.edges[u, v]['miles'] * 365
    #                                    for u, v in G.edges]) * tonmi_deflation_factor
    # baseline_annual_tonmi = hydrogen_annual_tonmi + support_diesel_annual_tonmi
    # support_diesel_annual_gal = sum([G.edges[u, v]['support_diesel_avg_gal']['TOTAL'] * 365 for u, v in G.edges])
    # annual_hydrogen_total_emissions = sum([G.nodes[n]['energy_source_LCA']['annual_total_emissions_tonco2'] for n in G])
    # G.graph['energy_source_LCA'] = dict(
    #     annual_hydrogen_total_emissions=annual_hydrogen_total_emissions,
    #     annual_support_diesel_total_emissions=diesel_factor * support_diesel_annual_gal,
    #     hydrogen_emissions_tonco2_kgh2=annual_hydrogen_total_emissions / hydrogen_annual_energy_kgh2,
    #     hydrogen_emissions_tonco2_tonmi=annual_hydrogen_total_emissions / hydrogen_annual_tonmi,
    #     support_diesel_emissions_tonco2_tonmi=diesel_factor * support_diesel_annual_gal / support_diesel_annual_tonmi
    #     if support_diesel_annual_tonmi > 0 else 0,
    #     avg_emissions_tonco2_tonmi=((annual_hydrogen_total_emissions + diesel_factor * support_diesel_annual_gal) /
    #                                 baseline_annual_tonmi),
    #     annual_total_emissions_tonco2=annual_hydrogen_total_emissions + diesel_factor * support_diesel_annual_gal
    # )
    annual_hydrogen_total_emissions = {c: sum(e_factor * G.edges[u, v]['hydrogen_avg_kgh2'][c] * 365
                                              for u, v in G.edges) for c in comm_list}
    tonmi_deflation_factor = {c: 1 - G.graph['operations']['perc_tonmi_inc'][c] / 100 for c in comm_list}
    hydrogen_annual_tonmi = {c: sum([G.edges[u, v]['hydrogen_avg_ton'][c] * G.edges[u, v]['miles'] * 365
                                     for u, v in G.edges]) * tonmi_deflation_factor[c] for c in comm_list}
    support_diesel_annual_tonmi = {c: sum([G.edges[u, v]['support_diesel_avg_ton'][c] * G.edges[u, v]['miles'] * 365
                                           for u, v in G.edges]) * tonmi_deflation_factor[c] for c in comm_list}
    baseline_annual_tonmi = {c: hydrogen_annual_tonmi[c] + support_diesel_annual_tonmi[c] for c in comm_list}
    annual_support_diesel_total_emissions = {c: sum([diesel_factor * G.edges[u, v]['support_diesel_avg_gal'][c] * 365
                                                     for u, v in G.edges]) for c in comm_list}

    G.graph['energy_source_LCA'] = dict(
        annual_hydrogen_total_emissions=annual_hydrogen_total_emissions,
        annual_support_diesel_total_emissions=annual_support_diesel_total_emissions,
        hydrogen_emissions_tonco2_kgh2={c: e_factor for c in comm_list},
        hydrogen_emissions_tonco2_tonmi=dict(zip(
            comm_list,
            [annual_hydrogen_total_emissions[c] / hydrogen_annual_tonmi[c]
             if hydrogen_annual_tonmi[c] > 0 else 0 for c in comm_list])),
        support_diesel_emissions_tonco2_tonmi=dict(zip(
            comm_list,
            [annual_support_diesel_total_emissions[c] / support_diesel_annual_tonmi[c]
             if support_diesel_annual_tonmi[c] > 0 else 0 for c in comm_list])),
        avg_emissions_tonco2_tonmi=dict(zip(
            comm_list,
            [(annual_hydrogen_total_emissions[c] + annual_support_diesel_total_emissions[c]) / baseline_annual_tonmi[c]
             if baseline_annual_tonmi[c] > 0 else 0 for c in comm_list])),
        annual_total_emissions_tonco2=dict(zip(
            comm_list,
            [annual_hydrogen_total_emissions[c] + annual_support_diesel_total_emissions[c] for c in comm_list]))
    )

    return G


'''
DROP-IN
'''


def load_lca_dropin_lookup():
    """
    Load dataframe for emissions factors: index is fuel_type, column is value in [g CO2/gal]

    Returns
    -------
    pd.DataFrame
        The dataframe containing the emissions factors for drop-in fuels in [g CO2/gal]

    """
    return pd.read_csv(os.path.join(LCA_DIR, 'lca_dropin_fuel_g_gal.csv'), header=0, index_col='fuel_type')


def lca_dropin_mp(G: nx.DiGraph, time_horizon: list, fuel_type: str, deployment_perc: float,
                  scenario_fuel_type: str = None):

    for e in G.edges:
        G.edges[e][fuel_type + '_LCA'] = {t: dict() for t in time_horizon}

    G.graph[fuel_type + '_LCA'] = {t: dict() for t in time_horizon}

    for t in time_horizon:
        G = lca_dropin_step_mp(G=G, time_step=t, fuel_type=fuel_type, deployment_perc=deployment_perc,
                               scenario_fuel_type=scenario_fuel_type)

    return G


def lca_dropin_step_mp(G: nx.DiGraph, time_step: str, fuel_type: str, deployment_perc: float,
                       scenario_fuel_type: str = None):
    """
    Calculate the emissions information for a given fuel type and deployment percentage in the networkx graph.

    Parameters
    ----------
    G : nx.DiGraph
        The networkx graph to add the emissions information to.
    fuel_type : str
        The fuel type to calculate emissions for. Acceptable values:
        - 'diesel'
        - 'biodiesel'
        - 'e-fuel'
    deployment_perc : float
        The percentage of deployment for the fuel type.
    scenario_fuel_type : str, optional
        The fuel type to use for the scenario comparison, by default None and uses the fuel_type parameter.

    Returns
    -------
    nx.DiGraph
        The input networkx graph with added edge attribute with emissions information for the specified fuel type for
        the specified deployment percentage.
    """
    # for comparing alternative fuels against diesel
    if scenario_fuel_type is None:
        scenario_fuel_type = fuel_type

    # load dataframe for emissions factors: index is fuel_type, column is value in [g CO2/gal]
    df_e = load_lca_dropin_lookup()
    e_factor = df_e.loc[fuel_type, 'gCO2/gal'] / 1e6  # in [ton CO2/gal]
    e_diesel = df_e.loc['diesel', 'gCO2/gal'] / 1e6
    weighted_e_factor = e_factor * deployment_perc + e_diesel * (1 - deployment_perc)

    if fuel_type != scenario_fuel_type:
        fuel_name = fuel_type
    else:
        fuel_name = 'energy_source'

    # get the list of all commodities present in the graph
    comm_list = list(G.graph['operations']['baseline_total_annual_tonmi'][time_step].keys())
    # loop through each edge in the graph
    for u, v in G.edges:
        e = G.edges[u, v]
        # calculate the ton-miles for each commodity
        tonmi = dict(zip(comm_list, [e['baseline_avg_ton'][time_step][c] * e['miles'] for c in comm_list]))
        # add the LCA attributes to the edge
        G.edges[u, v][fuel_name + '_LCA'][time_step] = dict(
            # <fuel_type> emissions in ton CO2 per ton-mile for each commodity
            emissions_tonco2_tonmi=dict(zip(
                comm_list,
                [e_factor * e['baseline_avg_gal'][time_step][c] / tonmi[c] if tonmi[c] > 0 else 0 for c in comm_list])),
            # total (weighted combination of <fuel_type> and diesel)
            # emissions in ton CO2 per ton-mile for each commodity
            total_emissions_tonco2_tonmi=dict(zip(
                comm_list,
                [weighted_e_factor * e['baseline_avg_gal'][time_step][c] / tonmi[c] if tonmi[c] > 0 else 0
                 for c in comm_list])),
            # daily <fuel_type> emissions in ton CO2 for each commodity
            daily_fuel_emissions_tonco2=dict(zip(
                comm_list,
                [e_factor * G.edges[u, v]['baseline_avg_gal'][time_step][c] * deployment_perc for c in comm_list])),
            # annual <fuel_type> emissions in ton CO2 for each commodity
            annual_fuel_emissions_tonco2=dict(zip(
                comm_list,
                [365 * e_factor * G.edges[u, v]['baseline_avg_gal'][time_step][c] * deployment_perc
                 for c in comm_list])),
            # daily total (weighted combination of <fuel_type> and diesel) emissions in ton CO2 for each commodity
            daily_total_emissions_tonco2=dict(zip(
                comm_list,
                [weighted_e_factor * G.edges[u, v]['baseline_avg_gal'][time_step][c] for c in comm_list])),
            # annual total (weighted combination of <fuel_type> and diesel) emissions in ton CO2 for each commodity
            annual_total_emissions_tonco2=dict(zip(
                comm_list,
                [365 * weighted_e_factor * G.edges[u, v]['baseline_avg_gal'][time_step][c] for c in comm_list])))

    # calculate the total ton-miles for each commodity in the graph
    total_tonmi = dict(zip(comm_list,
                           [sum([G.edges[e]['baseline_avg_ton'][time_step][c] * G.edges[e]['miles'] for e in G.edges])
                            for c in comm_list]))
    # take the average of all the LCA attributes and store in graph
    G.graph[fuel_name + '_LCA'][time_step] = dict(
        # <fuel_type> emissions in ton CO2 per ton-mile for each commodity
        emissions_tonco2_tonmi=dict(zip(
            comm_list,
            [sum(G.edges[u, v][fuel_name + '_LCA'][time_step]['emissions_tonco2_tonmi'][c] *
                 G.edges[u, v]['baseline_avg_ton'][time_step][c] * G.edges[u, v]['miles']
                 for u, v in G.edges) / total_tonmi[c] if total_tonmi[c] > 0 else 0 for c in comm_list])),
        # total (weighted combination) emissions in ton CO2 per ton-mile for each commodity
        total_emissions_tonco2_tonmi=dict(zip(
            comm_list,
            [sum(G.edges[u, v][fuel_name + '_LCA'][time_step]['total_emissions_tonco2_tonmi'][c] *
                 G.edges[u, v]['baseline_avg_ton'][time_step][c] * G.edges[u, v]['miles']
                 for u, v in G.edges) / total_tonmi[c] if total_tonmi[c] > 0 else 0 for c in comm_list])),
        # annual <fuel_type> emissions in ton CO2 for each commodity
        annual_fuel_emissions_tonco2=dict(zip(
            comm_list,
            [365 * sum(G.edges[u, v][fuel_name + '_LCA'][time_step]['daily_fuel_emissions_tonco2'][c]
                       for u, v in G.edges) for c in comm_list])),
        # annual total (weighted combination) emissions in ton CO2 for each commodity
        annual_total_emissions_tonco2=dict(zip(
            comm_list,
            [365 * sum(G.edges[u, v][fuel_name + '_LCA'][time_step]['daily_total_emissions_tonco2'][c]
                       for u, v in G.edges) for c in comm_list])))

    return G


def lca_diesel_mp(G: nx.DiGraph, time_horizon: list):

    for e in G.edges:
        G.edges[e]['diesel_LCA'] = {t: dict() for t in time_horizon}

    G.graph['diesel_LCA'] = {t: dict() for t in time_horizon}

    for t in time_horizon:
        G = lca_diesel_step_mp(G=G, time_step=t)

    return G


def lca_diesel_step_mp(G: nx.DiGraph, time_step: str):
    """
    Calculate the emissions information for a given fuel type and deployment percentage in the networkx graph.

    Parameters
    ----------
    G : nx.DiGraph
        The networkx graph to add the emissions information to.
    fuel_type : str
        The fuel type to calculate emissions for. Acceptable values:
        - 'diesel'
        - 'biodiesel'
        - 'e-fuel'
    deployment_perc : float
        The percentage of deployment for the fuel type.
    scenario_fuel_type : str, optional
        The fuel type to use for the scenario comparison, by default None and uses the fuel_type parameter.

    Returns
    -------
    nx.DiGraph
        The input networkx graph with added edge attribute with emissions information for the specified fuel type for
        the specified deployment percentage.
    """
    # for comparing alternative fuels against diesel

    # load dataframe for emissions factors: index is fuel_type, column is value in [g CO2/gal]
    df_e = load_lca_dropin_lookup()
    e_diesel = df_e.loc['diesel', 'gCO2/gal'] / 1e6

    # get the list of all commodities present in the graph
    comm_list = list(G.graph['operations']['baseline_total_annual_tonmi'][time_step].keys())
    # loop through each edge in the graph
    for u, v in G.edges:
        e = G.edges[u, v]
        # calculate the ton-miles for each commodity
        tonmi = dict(zip(comm_list, [e['baseline_avg_ton'][time_step][c] * e['miles'] for c in comm_list]))
        # add the LCA attributes to the edge
        G.edges[u, v]['diesel_LCA'][time_step] = dict(
            # total diesel emissions in ton CO2 per ton-mile for each commodity
            total_emissions_tonco2_tonmi=dict(zip(
                comm_list,
                [e_diesel * e['baseline_avg_gal'][time_step][c] / tonmi[c] if tonmi[c] > 0 else 0
                 for c in comm_list])),
            # daily total diesel emissions in ton CO2 for each commodity
            daily_total_emissions_tonco2=dict(zip(
                comm_list,
                [e_diesel * G.edges[u, v]['baseline_avg_gal'][time_step][c] for c in comm_list])),
            # annual total diesel emissions in ton CO2 for each commodity
            annual_total_emissions_tonco2=dict(zip(
                comm_list,
                [365 * e_diesel * G.edges[u, v]['baseline_avg_gal'][time_step][c] for c in comm_list]))
        )

    # calculate the total ton-miles for each commodity in the graph
    total_tonmi = dict(zip(comm_list,
                           [sum([G.edges[e]['baseline_avg_ton'][time_step][c] * G.edges[e]['miles'] for e in G.edges])
                            for c in comm_list]))
    # take the average of all the LCA attributes and store in graph
    G.graph['diesel_LCA'][time_step] = dict(
        # total diesel emissions in ton CO2 per ton-mile for each commodity
        total_emissions_tonco2_tonmi=dict(zip(
            comm_list,
            [sum(G.edges[e]['diesel_LCA'][time_step]['total_emissions_tonco2_tonmi'][c] *
                 G.edges[e]['baseline_avg_ton'][time_step][c] * G.edges[e]['miles'] for e in G.edges) /
             total_tonmi[c] if total_tonmi[c] > 0 else 0 for c in comm_list])),
        # annual total diesel emissions in ton CO2 for each commodity
        annual_total_emissions_tonco2=dict(zip(
            comm_list,
            [365 * sum(G.edges[u, v]['diesel_LCA'][time_step]['daily_total_emissions_tonco2'][c]
                       for u, v in G.edges) for c in comm_list])))

    return G
