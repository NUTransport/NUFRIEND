import networkx as nx
import numpy as np

from helper import load_fuel_tech_eff_factor, load_conversion_factors, load_railroad_values, \
    load_tea_battery_lookup, load_tea_hydrogen_lookup, elec_rate_state, elec_rate_state_mp, load_diesel_prices_mp
from util import *

'''
BATTERY
'''


def tea_battery_mp(G: nx.DiGraph, time_horizon: list):
    for n in G:
        G.nodes[n]['energy_source_TEA'] = {t: dict() for t in time_horizon}

    G.graph['energy_source_TEA'] = {t: dict() for t in time_horizon}

    for t in time_horizon:
        G = tea_battery_all_facilities_mp(G=G, time_step=t)

    return G


# def tea_battery(peak_loc: float, avg_loc: float, avg_mwh: float, elec_rate: float,
#                 max_util: float = 0.88, station_type: str = None):
#     """
#     Calculates the breakdown of LCO into capital, O&M, and energy costs as well as the capital investment,
#     annual O&M + energy cost, and actual average utilization and number of chargers for a battery charging facility
#     of given throughput and size based on <peak_loc>, <max_util>, and <avg_loc>.
#
#     Parameters
#     ----------
#     peak_loc : float
#         The peak number of locomotives that need to be charged at the facility
#     avg_loc : float
#         The average number of locomotives that need to be charged at the facility
#     avg_mwh : float
#         The total average energy consumption of the locomotives at the facility in MWh
#     elec_rate : float
#         The cost of electricity in $/kWh
#     max_util : float, optional
#         The maximum utilization of the facility, by default 0.88
#     station_type : str, optional
#         The type of charging station, by default None
#
#     Returns
#     -------
#     dict
#         A dictionary containing the breakdown of LCO, capital investment, annual costs, actual utilization and
#         number of chargers
#     """
#     # perform interpolation of results for a facility of given <peak_loc>, <max_util>, and <avg_loc> and
#     # return the breakdown of LCO into: capital, O&M, and energy
#     # and the capital investment, annual O&M + energy cost, and actual avg utilization and number of chargers
#     # future versions will include params for charger type (power) locomotive energy storage, etc.
#
#     # for selected facilities that do not consume electricity from the grid, but instead from locomotive traffic,
#     # size them according to their energy demand from the facility_sizing module
#
#     df = load_tea_battery_lookup()
#     charge_time = df.loc[0, 'Total Charging Time per locomotive [hrs]']
#
#     if peak_loc == 0 and avg_loc == 0:
#         return dict(station_LCO=0, om_LCO=0, energy_LCO=0, total_LCO=0, annual_total_cost=0, daily_energy_kwh=0,
#                     annual_energy_kwh=0, station_total=0, annual_om_energy=0, actual_utilization=0, number_chargers=0,
#                     charge_time=charge_time)
#
#     locos_p_charger = list(df['Locomotive per charger'].unique())
#     # isolate station utilization and locomotive per charger to get relationship between util and throughput
#     df_util_loco_p_charger = df[['Station Utilization', 'Locomotive per charger']].groupby(
#         by=['Station Utilization', 'Locomotive per charger'], as_index=False).first()
#     # find maximum number of locomotives per charger for given maximum utilization (discrete throughput)
#     max_int_loco_p_charger = df_util_loco_p_charger[df_util_loco_p_charger['Station Utilization'] <= max_util].dropna()[
#         'Locomotive per charger'].max()
#     # compute number of chargers needed
#     number_of_charger = np.ceil(peak_loc / max_int_loco_p_charger)
#     # compute actual (average) number of locomotives per charger for station
#     actual_loco_p_charger = (avg_loc / number_of_charger)
#     # compute actual (average) utilization for station
#     actual_util = actual_loco_p_charger * charge_time / 24
#
#     # check if average number of locomotives exceeds values in lookup table and replace
#     max_loc = max(df['Number of Locomotive'])
#     avg_loc_multiplier = 1
#     if avg_loc >= max_loc:
#         avg_loc_multiplier = avg_loc / max_loc  # estimator of cost scaling when number of locomotives is very large
#         avg_loc = max_loc
#     elif avg_loc < 1:
#         avg_loc = 1
#     avg_loc = int(avg_loc)
#
#     value_cols = ['Total energy [MWh]', 'Capital cost [$/kWh]', 'O&M less energy [$/kWh]', 'Energy [$/kWh]',
#                   'Total charging cost [$/kWh]', 'Charging station capital investment',
#                   'Annual O&M cost (w/energy cost)']
#     df_lookup = df.groupby(by=['Number of Locomotive', 'Locomotive per charger']).first().loc[(avg_loc, slice(None))]
#     # TODO: verify this interpolation makes sense and is like the one for H2
#     if actual_loco_p_charger <= min(locos_p_charger):
#         # use min util
#         lookup_locos = min(locos_p_charger)
#         df_results = df_lookup.loc[lookup_locos]
#     elif actual_loco_p_charger >= max(locos_p_charger):
#         # use max util
#         lookup_locos = max(locos_p_charger)
#         df_results = df_lookup.loc[lookup_locos]
#     else:
#         # use upper and lower and interpolate
#         # find maximum number of locomotives per charger for given maximum utilization (discrete throughput)
#         upper_loco = int(np.ceil(actual_loco_p_charger))
#         lower_loco = int(np.floor(actual_loco_p_charger))
#         if upper_loco == lower_loco:
#             scale = 1
#         else:
#             scale = (actual_loco_p_charger - lower_loco) / (upper_loco - lower_loco)
#         df_upper = df_lookup.loc[upper_loco][value_cols]
#         df_lower = df_lookup.loc[lower_loco][value_cols]
#         df_results = df_lower + scale * (df_upper - df_lower)
#
#     total_LCO = df_results['Capital cost [$/kWh]'] + df_results['O&M less energy [$/kWh]'] + elec_rate
#     return dict(station_LCO=df_results['Capital cost [$/kWh]'], om_LCO=df_results['O&M less energy [$/kWh]'],
#                 energy_LCO=elec_rate, total_LCO=total_LCO,
#                 annual_total_cost=(total_LCO * avg_mwh * 1000 * 365),
#                 daily_energy_kwh=avg_mwh * 1000, annual_energy_kwh=avg_mwh * 1000 * 365,
#                 station_total=avg_loc_multiplier * df_results['Charging station capital investment'],
#                 annual_om_energy=avg_loc_multiplier * df_results['Annual O&M cost (w/energy cost)'],
#                 actual_utilization=actual_util, number_chargers=number_of_charger, charge_time=charge_time)


def tea_battery(avg_loc: float, avg_mwh: float, elec_rate: float, max_util: float = 0.88, station_type: str = None):
    """
    Calculates the breakdown of LCO into capital, O&M, and energy costs as well as the capital investment,
    annual O&M + energy cost, and actual average utilization and number of chargers for a battery charging facility
    of given throughput and size based on <peak_loc>, <max_util>, and <avg_loc>.

    Parameters
    ----------
    peak_loc : float
        The peak number of locomotives that need to be charged at the facility
    avg_loc : float
        The average number of locomotives that need to be charged at the facility
    avg_mwh : float
        The total average energy consumption of the locomotives at the facility in MWh
    elec_rate : float
        The cost of electricity in $/kWh
    max_util : float, optional
        The maximum utilization of the facility, by default 0.88
    station_type : str, optional
        The type of charging station, by default None

    Returns
    -------
    dict
        A dictionary containing the breakdown of LCO, capital investment, annual costs, actual utilization and
        number of chargers
    """
    # perform interpolation of results for a facility of given <peak_loc>, <max_util>, and <avg_loc> and
    # return the breakdown of LCO into: capital, O&M, and energy
    # and the capital investment, annual O&M + energy cost, and actual avg utilization and number of chargers
    # future versions will include params for charger type (power) locomotive energy storage, etc.

    # for selected facilities that do not consume electricity from the grid, but instead from locomotive traffic,
    # size them according to their energy demand from the facility_sizing module

    df = load_tea_battery_lookup()
    charge_time = df.loc[0, 'Total Charging Time per locomotive [hrs]']

    if avg_loc == 0:
        return dict(station_LCO=0, om_LCO=0, energy_LCO=0, total_LCO=0, annual_total_cost=0, daily_energy_kwh=0,
                    annual_energy_kwh=0, station_total=0, annual_om_energy=0, actual_utilization=0, number_chargers=0,
                    charge_time=charge_time)

    locos_p_charger = list(df['Locomotive per charger'].unique())
    # isolate station utilization and locomotive per charger to get relationship between util and throughput
    df_util_loco_p_charger = df[['Station Utilization', 'Locomotive per charger']].groupby(
        by=['Station Utilization', 'Locomotive per charger'], as_index=False).first()
    # find maximum number of locomotives per charger for given maximum utilization (discrete throughput)
    max_int_loco_p_charger = df_util_loco_p_charger[df_util_loco_p_charger['Station Utilization'] <= max_util].dropna()[
        'Locomotive per charger'].max()
    # compute number of chargers needed
    number_of_charger = np.ceil(avg_loc / max_int_loco_p_charger)
    # compute actual (average) number of locomotives per charger for station
    actual_loco_p_charger = (avg_loc / number_of_charger)
    # compute actual (average) utilization for station
    actual_util = actual_loco_p_charger * charge_time / 24

    # check if average number of locomotives exceeds values in lookup table and replace
    max_loc = max(df['Number of Locomotive'])
    avg_loc_multiplier = 1
    if avg_loc >= max_loc:
        avg_loc_multiplier = avg_loc / max_loc  # estimator of cost scaling when number of locomotives is very large
        avg_loc = max_loc
    elif avg_loc < 1:
        avg_loc = 1
    avg_loc = int(avg_loc)

    value_cols = ['Total energy [MWh]', 'Capital cost [$/kWh]', 'O&M less energy [$/kWh]', 'Energy [$/kWh]',
                  'Total charging cost [$/kWh]', 'Charging station capital investment',
                  'Annual O&M cost (w/energy cost)']
    df_lookup = df.groupby(by=['Number of Locomotive', 'Locomotive per charger']).first().loc[(avg_loc, slice(None))]
    # TODO: verify this interpolation makes sense and is like the one for H2
    if actual_loco_p_charger <= min(locos_p_charger):
        # use min util
        lookup_locos = min(locos_p_charger)
        df_results = df_lookup.loc[lookup_locos]
    elif actual_loco_p_charger >= max(locos_p_charger):
        # use max util
        lookup_locos = max(locos_p_charger)
        df_results = df_lookup.loc[lookup_locos]
    else:
        # use upper and lower and interpolate
        # find maximum number of locomotives per charger for given maximum utilization (discrete throughput)
        upper_loco = int(np.ceil(actual_loco_p_charger))
        lower_loco = int(np.floor(actual_loco_p_charger))
        if upper_loco == lower_loco:
            scale = 1
        else:
            scale = (actual_loco_p_charger - lower_loco) / (upper_loco - lower_loco)
        df_upper = df_lookup.loc[upper_loco][value_cols]
        df_lower = df_lookup.loc[lower_loco][value_cols]
        df_results = df_lower + scale * (df_upper - df_lower)

    total_LCO = df_results['Capital cost [$/kWh]'] + df_results['O&M less energy [$/kWh]'] + elec_rate
    return dict(station_LCO=df_results['Capital cost [$/kWh]'], om_LCO=df_results['O&M less energy [$/kWh]'],
                energy_LCO=elec_rate, total_LCO=total_LCO,
                annual_total_cost=(total_LCO * avg_mwh * 1000 * 365),
                daily_energy_kwh=avg_mwh * 1000, annual_energy_kwh=avg_mwh * 1000 * 365,
                station_total=avg_loc_multiplier * df_results['Charging station capital investment'],
                annual_om_energy=avg_loc_multiplier * df_results['Annual O&M cost (w/energy cost)'],
                actual_utilization=actual_util, number_chargers=number_of_charger, charge_time=charge_time)


def tea_battery_all_facilities_mp(G: nx.DiGraph, time_step: str, max_util: float = 0.88, station_type: str = None,
                                  clean_energy_cost: float = None, tender_cost_p_tonmi: float = None,
                                  diesel_cost_p_gal: float = None) -> nx.DiGraph:
    """
    Compute aggregate statistics for battery technology deployment in all facilities. Use the percentage of ton-mi increase
    to calculate all in terms of baseline ton-miles.

    Parameters
    ----------
    tender_cost_p_tonmi
    G : nx.DiGraph
        Graph containing all the facilities and edges.
    max_util : float, optional
        Maximum station utilization, by default 0.88
    station_type : str, optional
        Type of station, by default None
    clean_energy_cost : float, optional
        Cost of clean energy, by default None

    Returns
    -------
    None
    """
    # compute aggregate statistics for tech. deployment
    # use the percentage of ton-mi increase to calculate all in terms of baseline ton-miles

    if clean_energy_cost is None:
        clean_energy_cost = 0

    # cost of electricity for each node based on state rates in [$/MWh]
    cost_p_location = elec_rate_state_mp(G, year=time_step)

    # load fuel technology factors
    ds = G.graph['scenario']
    ft_ef = load_fuel_tech_eff_factor().loc[ds['fuel_type']]  # fuel tech efficiency factors
    # lookup dataframes for constants
    rr_v = load_railroad_values().loc[ds['railroad']]
    cf = load_conversion_factors()['Value']  # numerical constants for conversion across units
    eff_kwh_p_batt = ds['eff_kwh_p_batt']  # effective battery capacity
    # calculate conversion factor from ton-miles to kwh
    tonmi2kwh = (rr_v['Energy intensity (btu/ton-mi)'] * (1 / cf['btu/kwh']) *
                 (1 / rr_v['Energy correction factor']) * (1 / ft_ef['Efficiency factor']) * (1 / ft_ef['Loss']))
    # calculate number of batteries per locomotive based on range and effective battery energy capacity
    batt_p_loc = tonmi2kwh * rr_v['ton/loc'] * ds['range_mi'] * (1 / eff_kwh_p_batt)
    # store listed kwh per battery in graph data
    G.graph['scenario']['listed_kwh_p_batt'] = eff_kwh_p_batt * (1 / ft_ef['Effective capacity'])
    G.graph['scenario']['batt_p_loc'] = batt_p_loc

    comm_list = list({c for u, v in G.edges for c in G.edges[u, v]['battery_avg_ton'][time_step].keys()})
    # calculate total battery ton-miles for each commodity
    battery_tonmi = {c: sum([G.edges[u, v]['battery_avg_ton'][time_step][c] * G.edges[u, v]['miles']
                             for u, v in G.edges]) for c in comm_list}
    # replace any zero values with 1
    battery_tonmi.update({c: battery_tonmi[c] if battery_tonmi[c] > 0 else 1 for c in comm_list})

    # car_dol_hr = 0  # [$/hr] delay cost per car-hr
    # car_dol_hr_im = 0  # [$/hr] delay cost per car-hr for intermodal
    car_dol_hr = 8.42  # [$/hr] delay cost per car-hr
    car_dol_hr_im = 26.95  # [$/hr] delay cost per car-hr for intermodal
    # delay_tonmi for IM
    im_share_tonmi = battery_tonmi['IM'] / battery_tonmi['TOTAL'] if battery_tonmi['TOTAL'] != 0 else 0

    # for each node in G
    for n in G:
        # if there is a facilit located at n
        if G.nodes[n]['facility'][time_step] == 1:
            # if the facility is merely an energy transfer point (does not consume energy from the grid)
            if 'energy_transfer' in G.nodes[n]['avg'][time_step].keys() and \
                    G.nodes[n]['avg'][time_step]['energy_transfer']:
                # apply tea_battery function to compute the costs based on the peak and average number of locomotives
                G.nodes[n]['energy_source_TEA'][time_step] = tea_battery(
                    np.ceil(G.nodes[n]['avg'][time_step]['number_loc'] * batt_p_loc),
                    -G.nodes[n]['avg'][time_step]['daily_demand_mwh'], 0,
                    max_util=max_util, station_type=station_type)
            # if the facility does consume energy from the grid
            else:
                # apply tea_battery function to compute the costs based on the peak and average number of locomotives
                G.nodes[n]['energy_source_TEA'][time_step] = tea_battery(
                    np.ceil(G.nodes[n]['avg'][time_step]['number_loc'] * batt_p_loc),
                    G.nodes[n]['avg'][time_step]['daily_supply_mwh'], cost_p_location[n] / 1000,
                    max_util=max_util, station_type=station_type)
            # get the time required to charge per locomotive
            charge_time = G.nodes[n]['energy_source_TEA'][time_step]['charge_time']
            # get the average and peak queue times and lengths
            lq_avg, wq_avg = queue_model(G.nodes[n]['avg'][time_step]['number_loc'] * batt_p_loc / 24,
                                         1 / charge_time,
                                         G.nodes[n]['energy_source_TEA'][time_step]['number_chargers'])
            # update the TEA dictionary with the new values
            G.nodes[n]['energy_source_TEA'][time_step].update(dict(
                charge_time=charge_time,
                avg_queue_time_p_loc=wq_avg,
                avg_queue_length=lq_avg / batt_p_loc,
                avg_daily_delay_cost_p_car=(charge_time + wq_avg) * car_dol_hr,
                avg_daily_delay_cost_p_loc=((charge_time + wq_avg) * car_dol_hr *
                                            (rr_v['car/train'] / rr_v['loc/train'])),
                total_daily_delay_cost=((car_dol_hr + (car_dol_hr_im - car_dol_hr) * im_share_tonmi) *
                                        (charge_time + wq_avg) * (rr_v['car/train'] / rr_v['loc/train']) *
                                        G.nodes[n]['avg'][time_step]['number_loc'])
            ))
        # if there is no facility at node n
        else:
            G.nodes[n]['energy_source_TEA'][time_step] = tea_battery(0, 0, 0, max_util=max_util,
                                                                     station_type=station_type)
            G.nodes[n]['energy_source_TEA'][time_step].update(dict(
                charge_time=0,
                avg_queue_time_p_loc=0,
                avg_queue_length=0,
                avg_daily_delay_cost_p_car=0,
                avg_daily_delay_cost_p_loc=0,
                total_daily_delay_cost=0
            ))

    # get the maximum charge time per locomotive of all the station locations
    charge_time = max([G.nodes[n]['energy_source_TEA'][time_step]['charge_time'] for n in G])
    # calculate the battery ton-miles by commodity
    battery_tonmi = {c: sum([G.edges[u, v]['battery_avg_ton'][time_step][c] * G.edges[u, v]['miles']
                             for u, v in G.edges]) for c in comm_list}
    # calculate the support diesel ton-miles by commodity
    support_diesel_tonmi = {c: sum([G.edges[u, v]['support_diesel_avg_ton'][time_step][c] * G.edges[u, v]['miles']
                                    for u, v in G.edges]) for c in comm_list}
    # calculate the support diesel fuel consumption [gal] by commodity
    support_diesel_gal = {c: sum([G.edges[u, v]['support_diesel_avg_gal'][time_step][c] for u, v in G.edges])
                          for c in comm_list}
    # calculate the baseline (diesel) ton-miles by commodity
    baseline_total_tonmi = {c: battery_tonmi[c] + support_diesel_tonmi[c] for c in comm_list}
    # update to remove zero values (division issues)
    baseline_total_tonmi.update({c: baseline_total_tonmi[c] if baseline_total_tonmi[c] > 0 else 1 for c in comm_list})
    # calculate the average energy consumed [kWh] by commodity
    avg_battery_energy_kwh = {c: sum(G.edges[u, v]['battery_avg_kwh'][time_step][c] for u, v in G.edges)
                              for c in comm_list}

    if tender_cost_p_tonmi is None:
        tender_cost_p_tonmi = rr_v['battery $/ton-mile']
    # load battery $/ton-mi; cost of battery is with respect to nameplate capacity, not effective capacity
    battery_LCO_tonmi = batt_p_loc * tender_cost_p_tonmi * (1 / ft_ef['Effective capacity'])
    # convert to battery $/kWh by commodity
    battery_LCO_kwh = {c: (battery_LCO_tonmi * battery_tonmi[c] /
                           avg_battery_energy_kwh[c]) if avg_battery_energy_kwh[c] > 0 else 0 for c in comm_list}

    if diesel_cost_p_gal is None:
        # load dataframe for cost factors of diesel: index is fuel_type, column is value in [$/gal]
        diesel_factor = load_diesel_prices_mp().loc[int(time_step)].item()
    else:
        diesel_factor = diesel_cost_p_gal
    # calculate the total of average number of locomotives
    avg_tot_loc = sum([G.nodes[n]['avg'][time_step]['number_loc'] for n in G if G.nodes[n]['facility'][time_step]])
    if round(avg_tot_loc) == 0:
        avg_tot_loc = 0

    # compute and store average TEA calculations as graph attributes
    G.graph['energy_source_TEA'][time_step] = dict(
        # avg station_LCO per kWh should be the total cost of the station (from peak value) over the avg usage
        station_LCO_kwh=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA'][time_step]['station_LCO'] * 1000 *
                  G.nodes[n]['avg'][time_step]['daily_supply_mwh'] for n in G]) / avg_battery_energy_kwh['TOTAL']
             for c in comm_list])),
        # battery cost per kWh
        battery_LCO_kwh=battery_LCO_kwh,
        # O&M cost per kWh
        om_LCO_kwh=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA'][time_step]['om_LCO'] * 1000 *
                  G.nodes[n]['avg'][time_step]['daily_supply_mwh'] for n in G]) / avg_battery_energy_kwh['TOTAL']
             for c in comm_list])),
        # electricity cost per kWh
        energy_LCO_kwh=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA'][time_step]['energy_LCO'] * 1000 * G.nodes[n]['avg'][time_step][
                'daily_supply_mwh']
                  for n in G]) / avg_battery_energy_kwh['TOTAL'] for c in comm_list])),
        # estimated delay cost of charging and queuing per kWh
        delay_LCO_kwh=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA'][time_step]['total_daily_delay_cost'] for n in G]) /
             avg_battery_energy_kwh['TOTAL'] for c in comm_list])),
        # total cost per kWh
        total_LCO_kwh=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA'][time_step]['total_LCO'] * 1000 * G.nodes[n]['avg'][time_step][
                'daily_supply_mwh'] +
                  G.nodes[n]['energy_source_TEA'][time_step]['total_daily_delay_cost'] for n in G]) /
             avg_battery_energy_kwh['TOTAL'] + battery_LCO_kwh['TOTAL'] for c in comm_list])),
        # amortized station capital cost (annual)
        station_annual_cost=365 * sum([G.nodes[n]['energy_source_TEA'][time_step]['station_LCO'] * 1000 *
                                       G.nodes[n]['avg'][time_step]['daily_supply_mwh'] for n in G]),
        # amortized battery capital cost (annual)
        battery_annual_cost={c: 365 * battery_LCO_tonmi * battery_tonmi[c] for c in comm_list},
        # station capital cost
        station_total=(sum([G.nodes[n]['energy_source_TEA'][time_step]['station_total'] for n in G])),
        # average station utilization over entire network
        actual_utilization=(sum([G.nodes[n]['energy_source_TEA'][time_step]['actual_utilization'] *
                                 G.nodes[n]['energy_source_TEA'][time_step]['daily_energy_kwh'] for n in G]) /
                            avg_battery_energy_kwh['TOTAL']),
        # total number of chargers installed
        number_chargers=sum([G.nodes[n]['energy_source_TEA'][time_step]['number_chargers'] for n in G]),
        # average number of chargers per station
        charger_per_station=round((sum([G.nodes[n]['energy_source_TEA'][time_step]['number_chargers'] *
                                        G.nodes[n]['energy_source_TEA'][time_step]['daily_energy_kwh'] for n in G]) /
                                   avg_battery_energy_kwh['TOTAL']), 1),
        # daily energy consumption in kWh
        daily_energy_kwh=sum([G.nodes[n]['energy_source_TEA'][time_step]['daily_energy_kwh'] for n in G]),
        # annual energy consumption in kWh
        annual_energy_kwh=sum([G.nodes[n]['energy_source_TEA'][time_step]['annual_energy_kwh'] for n in G]),
        # charge time per locomotive [hrs]
        charge_time=charge_time,
        # average queue time per locomotive [hrs]
        avg_queue_time_p_loc=(sum([G.nodes[n]['energy_source_TEA'][time_step]['avg_queue_time_p_loc'] *
                                   G.nodes[n]['avg'][time_step]['number_loc'] for n in G if
                                   G.nodes[n]['facility'][time_step]]) /
                              avg_tot_loc),
        # average queue length [# locomotives]
        avg_queue_length=(sum([G.nodes[n]['energy_source_TEA'][time_step]['avg_queue_length'] *
                               G.nodes[n]['avg'][time_step]['number_loc'] for n in G if
                               G.nodes[n]['facility'][time_step]]) / avg_tot_loc),
        # average daily delay cost per car
        avg_daily_delay_cost_p_car=(sum([G.nodes[n]['energy_source_TEA'][time_step]['avg_daily_delay_cost_p_car'] *
                                         G.nodes[n]['avg'][time_step]['number_loc'] for n in G if
                                         G.nodes[n]['facility'][time_step]]) /
                                    avg_tot_loc),
        # total daily delay cost
        total_daily_delay_cost=sum([G.nodes[n]['energy_source_TEA'][time_step]['total_daily_delay_cost'] for n in G]),
        # total annual delay cost
        total_annual_delay_cost=365 * sum(
            [G.nodes[n]['energy_source_TEA'][time_step]['total_daily_delay_cost'] for n in G])
    )
    # update dictionary with levelized cost calculations computed in terms of [$/ton-mile]
    G.graph['energy_source_TEA'][time_step].update(dict(
        # avg station_LCO per tonmi should be the total cost of the station (from peak value) over the battery tonmi
        station_LCO_tonmi=dict(zip(
            comm_list,
            [G.graph['energy_source_TEA'][time_step]['station_LCO_kwh'][c] * avg_battery_energy_kwh[c] / battery_tonmi[
                c]
             for c in comm_list])),
        # battery levelized cost per ton-mi
        battery_LCO_tonmi={c: battery_LCO_tonmi for c in comm_list},
        # O&M levelized cost per ton-mi
        om_LCO_tonmi=dict(zip(
            comm_list,
            [G.graph['energy_source_TEA'][time_step]['om_LCO_kwh'][c] * avg_battery_energy_kwh[c] / battery_tonmi[c]
             for c in comm_list])),
        # energy/electricity levelized cost per ton-mi
        energy_LCO_tonmi=dict(zip(
            comm_list,
            [G.graph['energy_source_TEA'][time_step]['energy_LCO_kwh'][c] * avg_battery_energy_kwh[c] / battery_tonmi[c]
             for c in comm_list])),
        # delay levelized cost per ton-mi
        delay_LCO_tonmi=dict(zip(
            comm_list,
            [G.graph['energy_source_TEA'][time_step]['delay_LCO_kwh'][c] * avg_battery_energy_kwh[c] / battery_tonmi[c]
             for c in comm_list]))
    ))

    G.graph['energy_source_TEA'][time_step].update(dict(
        # total levelized cost per ton-mi (only for battery costs)
        total_LCO_tonmi={c: (G.graph['energy_source_TEA'][time_step]['station_LCO_tonmi'][c] +
                             G.graph['energy_source_TEA'][time_step]['battery_LCO_tonmi'][c] +
                             G.graph['energy_source_TEA'][time_step]['om_LCO_tonmi'][c] +
                             G.graph['energy_source_TEA'][time_step]['energy_LCO_tonmi'][c] +
                             G.graph['energy_source_TEA'][time_step]['delay_LCO_tonmi'][c]) for c in comm_list},
        # total levelized cost per ton-mi (for battery and support diesel operations costs)
        total_scenario_LCO_tonmi={c: ((G.graph['energy_source_TEA'][time_step]['station_LCO_tonmi'][c] +
                                       G.graph['energy_source_TEA'][time_step]['battery_LCO_tonmi'][c] +
                                       G.graph['energy_source_TEA'][time_step]['om_LCO_tonmi'][c] +
                                       G.graph['energy_source_TEA'][time_step]['energy_LCO_tonmi'][c] +
                                       G.graph['energy_source_TEA'][time_step]['delay_LCO_tonmi'][c]) * battery_tonmi[
                                          c] +
                                      diesel_factor * support_diesel_gal[c]) / baseline_total_tonmi[c]
                                  for c in comm_list},
        # total levelized cost (excluding delay costs) per ton-mi (only for battery costs)
        total_nodelay_LCO_tonmi={c: (G.graph['energy_source_TEA'][time_step]['station_LCO_tonmi'][c] +
                                     G.graph['energy_source_TEA'][time_step]['battery_LCO_tonmi'][c] +
                                     G.graph['energy_source_TEA'][time_step]['om_LCO_tonmi'][c] +
                                     G.graph['energy_source_TEA'][time_step]['energy_LCO_tonmi'][c]) for c in
                                 comm_list},
        # total levelized cost (excluding delay costs) per ton-mi (for battery and support diesel operations costs)
        total_scenario_nodelay_LCO_tonmi={c: ((G.graph['energy_source_TEA'][time_step]['station_LCO_tonmi'][c] +
                                               G.graph['energy_source_TEA'][time_step]['battery_LCO_tonmi'][c] +
                                               G.graph['energy_source_TEA'][time_step]['om_LCO_tonmi'][c] +
                                               G.graph['energy_source_TEA'][time_step]['energy_LCO_tonmi'][c]) *
                                              battery_tonmi[c] +
                                              diesel_factor * support_diesel_gal[c]) / baseline_total_tonmi[c]
                                          for c in comm_list},
        # annual cost related to battery operations
        annual_battery_total_cost={c: 365 * ((G.graph['energy_source_TEA'][time_step]['station_LCO_tonmi'][c] +
                                              G.graph['energy_source_TEA'][time_step]['battery_LCO_tonmi'][c] +
                                              G.graph['energy_source_TEA'][time_step]['om_LCO_tonmi'][c] +
                                              G.graph['energy_source_TEA'][time_step]['energy_LCO_tonmi'][c]) *
                                             battery_tonmi[c]) for c in comm_list},
        # annual cost associated with support diesel operations
        annual_support_diesel_total_cost={c: 365 * diesel_factor * support_diesel_gal[c] for c in comm_list},
        # annual total cost for complete scenario (includes respective batter and support diesel costs)
        annual_total_cost={c: 365 * ((G.graph['energy_source_TEA'][time_step]['station_LCO_tonmi'][c] +
                                      G.graph['energy_source_TEA'][time_step]['battery_LCO_tonmi'][c] +
                                      G.graph['energy_source_TEA'][time_step]['om_LCO_tonmi'][c] +
                                      G.graph['energy_source_TEA'][time_step]['energy_LCO_tonmi'][c] +
                                      G.graph['energy_source_TEA'][time_step]['delay_LCO_tonmi'][c]) *
                                     battery_tonmi[c] + diesel_factor * support_diesel_gal[c]) for c in comm_list}
    ))

    return G


'''
HYDROGEN
'''


# TODO: make for MP
def tea_hydrogen(peak_loc: float, avg_loc: float, avg_kgh2: float, max_util: float = 0.88,
                 loc2kgh2: float = 4000, station_type: str = 'Cryo-pump', clean_energy_dolkg: float = None):
    """
    Calculates the breakdown of LCO into capital, O&M, and energy costs as well as the capital investment,
    annual O&M + energy cost, and actual average utilization and number of chargers for a battery charging facility
    of given throughput and size based on <peak_loc>, <max_util>, and <avg_loc>.

    TODO: fix this/ trim down

    # perform interpolation of results for a facility of given <peak_loc>, <max_util>, and <avg_loc> and
    # return the breakdown of LCO into: capital, O&M, and energy
    # and the capital investment, annual O&M + energy cost, and actual avg utilization and number of chargers
    # future versions will include params for charger type (power) locomotive energy storage, etc.

    # for selected facilities that do not consume electricity from the grid, but instead from locomotive traffic,
    # size them according to their energy demand from the facility_sizing module

    Parameters
    ----------
    peak_loc : float
        The peak number of locomotives that need to be charged at the facility
    avg_loc : float
        The average number of locomotives that need to be charged at the facility
    avg_kgh2 : float
        The total average energy consumption of the locomotives at the facility in kg H2
    max_util : float, optional
        The maximum utilization of the facility, by default 0.88
    loc2kgh2 : float, optional
        The capacity of kg H2 per locomotive, by default 4000 kg H2 per locomotive
    station_type : str, optional
        The type of charging station, by default None
    clean_energy_dolkg : float, optional
        The premium (additional cost) for cleanly-sourced hydrogen in [$/kg H2]

    Returns
    -------
    dict
        A dictionary containing the breakdown of LCO, capital investment, annual costs, actual utilization and
        number of pumps

    """

    if clean_energy_dolkg is None:
        clean_energy_dolkg = 0

    if peak_loc == 0 and avg_loc == 0:
        return dict(station_LCO=0, terminal_LCO=0, liquefier_LCO=0, fuel_LCO=0, total_LCO=0, annual_total_cost=0,
                    daily_energy_kgh2=0, annual_energy_kgh2=0, station_total=0, annual_om_energy=0,
                    actual_utilization=0, number_pumps=0, pump_time=0)

    # load lookup table and filter out only <station_type> of interest
    df = load_tea_hydrogen_lookup().loc[(station_type, slice(None, None))]
    # and operation hours of interest
    # list of allowable hours per day
    hrs = np.array(df.index.get_level_values('Operation Hours').unique())
    # max allowable hours per day given the <max_util> param.
    op_hr = hrs[max(np.where(hrs <= max_util * 24)[0])] if np.where(hrs <= max_util * 24)[0].any() else min(hrs)
    df = df.loc[op_hr, slice(None)]

    pump_rate = 100  # [kgh2/min]
    pump_time = (loc2kgh2 / pump_rate) * (1 / 60)  # [kgh2/loc] / [kgh2/min] * [hr/60 min] = [hrs/loc]

    number_of_pumps = np.ceil(peak_loc * pump_time / op_hr)
    max_loc = max(df.index)
    min_loc = min(df.index)

    avg_number_of_pumps = np.ceil(avg_loc * pump_time / op_hr)

    if peak_loc < min_loc:
        df = df.loc[min_loc]
        tea_d = dict(
            station_LCO=df[' Liquid Refueling Station [$/kg] '],
            terminal_LCO=df[' Terminal [$/kg] '],
            liquefier_LCO=df[' Liquefier [$/kg] '],
            fuel_LCO=df[' Production [$/kg] '] + clean_energy_dolkg,
            total_LCO=df[' Total Levelized Cost of Refueling [$/kg] '],
            annual_total_cost=(df[' Total Levelized Cost of Refueling [$/kg] '] * avg_kgh2 * 365),
            daily_energy_kgh2=avg_kgh2,
            annual_energy_kgh2=avg_kgh2 * 365,
            station_total=df[' Total Capital Investment '],
            actual_utilization=avg_number_of_pumps / number_of_pumps,
            number_pumps=number_of_pumps,
            pump_time=pump_time)
    elif peak_loc <= max_loc:
        # interpolate
        # df.reset_index(inplace=True)
        # df.set_index('Number of Dispenser', inplace=True)
        df1 = df.loc[df.index[max(np.where(np.array(df.index) <= peak_loc)[0])]]
        df2 = df.loc[df.index[min(np.where(np.array(df.index) >= peak_loc)[0])]]
        # use these to do interpolation by using number of pumps, not loc, as the x
        tea_d = dict(
            station_LCO=df1[' Liquid Refueling Station [$/kg] '] + \
                        ((df2[' Liquid Refueling Station [$/kg] '] - df1[
                            ' Liquid Refueling Station [$/kg] ']) *
                         (number_of_pumps - df1['Number of Dispenser']) /
                         (df2['Number of Dispenser'] - df1['Number of Dispenser'])),
            terminal_LCO=df1[' Terminal [$/kg] '] + \
                         ((df2[' Terminal [$/kg] '] - df1[' Terminal [$/kg] ']) *
                          (number_of_pumps - df1['Number of Dispenser']) /
                          (df2['Number of Dispenser'] - df1['Number of Dispenser'])),
            liquefier_LCO=df1[' Liquefier [$/kg] '] + \
                          ((df2[' Liquefier [$/kg] '] - df1[' Liquefier [$/kg] ']) *
                           (number_of_pumps - df1['Number of Dispenser']) /
                           (df2['Number of Dispenser'] - df1['Number of Dispenser'])),
            fuel_LCO=df1[' Production [$/kg] '] + clean_energy_dolkg
        )

        tea_d.update(dict(
            total_LCO=tea_d['station_LCO'] + tea_d['terminal_LCO'] + tea_d['liquefier_LCO'] + tea_d['fuel_LCO'],
            annual_total_cost=((tea_d['station_LCO'] + tea_d['terminal_LCO'] + tea_d['liquefier_LCO'] +
                                tea_d['fuel_LCO']) * avg_kgh2 * 365),
            daily_energy_kgh2=avg_kgh2,
            annual_energy_kgh2=avg_kgh2 * 365,
            station_total=df1[' Total Capital Investment '] + \
                          ((df2[' Total Capital Investment '] - df1[' Total Capital Investment ']) *
                           (number_of_pumps - df1['Number of Dispenser']) /
                           (df2['Number of Dispenser'] - df1['Number of Dispenser'])),
            actual_utilization=avg_number_of_pumps / number_of_pumps,
            number_pumps=number_of_pumps,
            pump_time=pump_time))

    else:
        max_pump = np.ceil(max_loc * pump_time / op_hr)
        peak_pump_multiplier = 1
        if number_of_pumps > max_pump:
            peak_pump_multiplier = number_of_pumps / max_pump
        # return the values for 200 loc/day and scale the capital costs accordingly
        df = df.loc[max_loc]
        tea_d = dict(
            station_LCO=df[' Liquid Refueling Station [$/kg] '],
            terminal_LCO=df[' Terminal [$/kg] '],
            liquefier_LCO=df[' Liquefier [$/kg] '],
            fuel_LCO=df[' Production [$/kg] '] + clean_energy_dolkg,
            total_LCO=df[' Total Levelized Cost of Refueling [$/kg] '],
            annual_total_cost=(df[' Total Levelized Cost of Refueling [$/kg] '] * avg_kgh2 * 365),
            daily_energy_kgh2=avg_kgh2,
            annual_energy_kgh2=avg_kgh2 * 365,
            station_total=peak_pump_multiplier * df[' Total Capital Investment '],
            actual_utilization=avg_number_of_pumps / number_of_pumps,
            number_pumps=number_of_pumps,
            pump_time=pump_time)

    return tea_d


def tea_hydrogen_all_facilities(G: nx.DiGraph, max_util: float = 0.88, station_type: str = 'Cryo-pump',
                                clean_energy_cost: float = None, diesel_cost_p_gal: float = None):
    # lookup dataframes for constants
    rr_v = load_railroad_values().loc[G.graph['scenario']['railroad']]
    # calculate average # batteries per locomotive based on range and effective battery energy capacity
    # eff_kwh_p_batt = ds['kwh_p_batt'] * ft_ef['Effective capacity']

    comm_list = list({c for u, v in G.edges for c in G.edges[u, v]['hydrogen_avg_ton'].keys()})

    tonmi_deflation_factor = {c: 1 - G.graph['operations']['perc_tonmi_inc'][c] / 100 for c in comm_list}
    hydrogen_tonmi = {c: sum([G.edges[u, v]['hydrogen_avg_ton'][c] * G.edges[u, v]['miles']
                              for u, v in G.edges]) * tonmi_deflation_factor[c] for c in comm_list}
    hydrogen_tonmi.update({c: hydrogen_tonmi[c] if hydrogen_tonmi[c] > 0 else 1 for c in comm_list})

    car_dol_hr = 8.42  # [$/hr] delay cost per car-hr
    car_dol_hr_im = 26.95  # [$/hr] delay cost per car-hr for intermodal
    # car_dol_hr = 0  # [$/hr] delay cost per car-hr
    # car_dol_hr_im = 0   # [$/hr] delay cost per car-hr for intermodal
    # delay_tonmi for IM
    im_share_tonmi = hydrogen_tonmi['IM'] / hydrogen_tonmi['TOTAL'] if hydrogen_tonmi['TOTAL'] != 0 else 0

    for n in G:
        if G.nodes[n]['facility'] == 1:
            # must supply the number of batteries of 10 MWh effective capacity that need to be charged
            G.nodes[n]['energy_source_TEA'] = tea_hydrogen(int(G.nodes[n]['peak']['number_loc']),
                                                           int(G.nodes[n]['avg']['number_loc']),
                                                           G.nodes[n]['avg']['daily_supply_kgh2'],
                                                           max_util=max_util, station_type=station_type,
                                                           clean_energy_dolkg=clean_energy_cost)
            if G.nodes[n]['avg']['energy_transfer']:
                G.nodes[n]['energy_source_TEA'].update(dict(
                    pump_time=0,
                    avg_queue_time_p_loc=0,
                    avg_queue_length=0,
                    peak_queue_time_p_loc=0,
                    peak_queue_length=0,
                    avg_daily_delay_cost_p_car=0,
                    avg_daily_delay_cost_p_loc=0,
                    total_daily_delay_cost=0
                ))
            else:
                pump_time = G.nodes[n]['energy_source_TEA']['pump_time']
                lq_avg, wq_avg = queue_model(G.nodes[n]['avg']['number_loc'] / 24,
                                             1 / pump_time,
                                             G.nodes[n]['energy_source_TEA']['number_pumps'])
                lq_peak, wq_peak = queue_model(G.nodes[n]['peak']['number_loc'] / 24,
                                               1 / pump_time,
                                               G.nodes[n]['energy_source_TEA']['number_pumps'])
                G.nodes[n]['energy_source_TEA'].update(dict(
                    pump_time=pump_time,
                    avg_queue_time_p_loc=wq_avg,
                    avg_queue_length=lq_avg,
                    peak_queue_time_p_loc=wq_peak,
                    peak_queue_length=lq_peak,
                    avg_daily_delay_cost_p_car=(pump_time + wq_avg) * car_dol_hr,
                    avg_daily_delay_cost_p_loc=((pump_time + wq_avg) * car_dol_hr *
                                                (rr_v['car/train'] / rr_v['loc/train'])),
                    total_daily_delay_cost=((car_dol_hr + (car_dol_hr_im - car_dol_hr) * im_share_tonmi) *
                                            (pump_time + wq_avg) *
                                            (rr_v['car/train'] / rr_v['loc/train']) * G.nodes[n]['avg']['number_loc'])
                ))
        else:
            G.nodes[n]['energy_source_TEA'] = tea_hydrogen(0, 0, 0, max_util=max_util)
            G.nodes[n]['energy_source_TEA'].update(dict(
                pump_time=0,
                avg_queue_time_p_loc=0,
                avg_queue_length=0,
                peak_queue_time_p_loc=0,
                peak_queue_length=0,
                avg_daily_delay_cost_p_car=0,
                avg_daily_delay_cost_p_loc=0,
                total_daily_delay_cost=0
            ))

    pump_time = max([G.nodes[n]['energy_source_TEA']['pump_time'] for n in G])
    # compute aggregate statistics for tech. deployment
    # use the percentage of ton-mi increase to calculate all in terms of baseline ton-miles

    avg_tot_loc = sum([G.nodes[n]['avg']['number_loc'] for n in G if G.nodes[n]['facility']])
    if round(avg_tot_loc) == 0:
        avg_tot_loc = 0
    peak_tot_loc = sum([G.nodes[n]['peak']['number_loc'] for n in G if G.nodes[n]['facility']])
    if round(peak_tot_loc) == 0:
        peak_tot_loc = 0

    support_diesel_tonmi = {c: sum([G.edges[u, v]['support_diesel_avg_ton'][c] * G.edges[u, v]['miles']
                                    for u, v in G.edges]) * tonmi_deflation_factor[c] for c in comm_list}
    support_diesel_gal = {c: sum([G.edges[u, v]['support_diesel_avg_gal'][c] for u, v in G.edges]) for c in comm_list}
    baseline_total_tonmi = {c: hydrogen_tonmi[c] + support_diesel_tonmi[c] for c in comm_list}
    baseline_total_tonmi.update({c: baseline_total_tonmi[c] if baseline_total_tonmi[c] > 0 else 1 for c in comm_list})

    avg_hydrogen_energy_kgh2 = {c: sum(G.edges[u, v]['hydrogen_avg_kgh2'][c] for u, v in G.edges) for c in comm_list}
    peak_hydrogen_energy_kgh2 = {c: sum(G.edges[u, v]['hydrogen_peak_kgh2'][c] for u, v in G.edges) for c in comm_list}

    tender_LCO_tonmi = rr_v[station_type + ' tender $/ton-mile']
    # convert to battery $/kWh
    tender_LCO_kgh2 = {c: tender_LCO_tonmi * hydrogen_tonmi[c] / avg_hydrogen_energy_kgh2[c]
    if avg_hydrogen_energy_kgh2[c] > 0 else 0 for c in comm_list}

    if diesel_cost_p_gal is None:
        # load dataframe for cost factors of diesel: index is fuel_type, column is value in [$/gal]
        df_dropin = load_tea_dropin_lookup()
        diesel_factor = df_dropin.loc['diesel', '$/gal']
    else:
        diesel_factor = diesel_cost_p_gal

    G.graph['energy_source_TEA'] = dict(
        # avg station_LCO per kWh should be the total cost of the station (from peak value) over the avg usage
        station_LCO_kgh2=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA']['station_LCO'] * G.nodes[n]['peak']['daily_supply_kgh2']
                  for n in G]) / avg_hydrogen_energy_kgh2['TOTAL'] for c in comm_list])),
        terminal_LCO_kgh2=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA']['terminal_LCO'] * G.nodes[n]['avg']['daily_supply_kgh2']
                  for n in G]) / avg_hydrogen_energy_kgh2['TOTAL'] for c in comm_list])),
        liquefier_LCO_kgh2=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA']['liquefier_LCO'] * G.nodes[n]['avg']['daily_supply_kgh2']
                  for n in G]) / avg_hydrogen_energy_kgh2['TOTAL'] for c in comm_list])),
        fuel_LCO_kgh2=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA']['fuel_LCO'] * G.nodes[n]['avg']['daily_supply_kgh2']
                  for n in G]) / avg_hydrogen_energy_kgh2['TOTAL'] for c in comm_list])),
        tender_LCO_kgh2=tender_LCO_kgh2,
        delay_LCO_kgh2=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA']['total_daily_delay_cost'] for n in G]) /
             avg_hydrogen_energy_kgh2['TOTAL'] for c in comm_list])),
        total_LCO_kgh2=dict(zip(
            comm_list,
            [sum([G.nodes[n]['energy_source_TEA']['total_LCO'] * G.nodes[n]['avg']['daily_supply_kgh2'] +
                  G.nodes[n]['energy_source_TEA']['total_daily_delay_cost'] for n in G]) /
             avg_hydrogen_energy_kgh2['TOTAL'] + tender_LCO_kgh2[c] for c in comm_list])),
        station_annual_cost=365 * sum([G.nodes[n]['energy_source_TEA']['station_LCO'] *
                                       G.nodes[n]['peak']['daily_supply_kgh2'] for n in G]),
        station_total=(sum([G.nodes[n]['energy_source_TEA']['station_total'] for n in G])),
        actual_utilization=(sum([G.nodes[n]['energy_source_TEA']['actual_utilization'] *
                                 G.nodes[n]['energy_source_TEA']['daily_energy_kgh2'] for n in G]) /
                            avg_hydrogen_energy_kgh2['TOTAL']),
        number_pumps=sum([G.nodes[n]['energy_source_TEA']['number_pumps'] for n in G]),
        pump_per_station=round((sum([G.nodes[n]['energy_source_TEA']['number_pumps'] *
                                     G.nodes[n]['energy_source_TEA']['daily_energy_kgh2'] for n in G]) /
                                avg_hydrogen_energy_kgh2['TOTAL']), 1),
        daily_energy_kgh2=sum([G.nodes[n]['energy_source_TEA']['daily_energy_kgh2'] for n in G]),
        annual_energy_kgh2=sum([G.nodes[n]['energy_source_TEA']['annual_energy_kgh2'] for n in G]),
        pump_time=pump_time,
        avg_queue_time_p_loc=(sum([G.nodes[n]['energy_source_TEA']['avg_queue_time_p_loc'] *
                                   G.nodes[n]['avg']['number_loc'] for n in G if G.nodes[n]['facility']]) /
                              avg_tot_loc),
        avg_queue_length=(sum([G.nodes[n]['energy_source_TEA']['avg_queue_length'] *
                               G.nodes[n]['avg']['number_loc'] for n in G if G.nodes[n]['facility']]) / avg_tot_loc),
        peak_queue_time_p_loc=(sum([G.nodes[n]['energy_source_TEA']['peak_queue_time_p_loc'] *
                                    G.nodes[n]['peak']['number_loc'] for n in G if G.nodes[n]['facility']])
                               / peak_tot_loc),
        peak_queue_length=(sum([G.nodes[n]['energy_source_TEA']['peak_queue_length'] *
                                G.nodes[n]['peak']['number_loc'] for n in G if G.nodes[n]['facility']])
                           / peak_tot_loc),
        avg_daily_delay_cost_p_car=(sum([G.nodes[n]['energy_source_TEA']['avg_daily_delay_cost_p_car'] *
                                         G.nodes[n]['avg']['number_loc'] for n in G if G.nodes[n]['facility']]) /
                                    avg_tot_loc),
        total_daily_delay_cost=sum([G.nodes[n]['energy_source_TEA']['total_daily_delay_cost'] for n in G]),
        total_annual_delay_cost=365 * sum([G.nodes[n]['energy_source_TEA']['total_daily_delay_cost'] for n in G])
    )

    G.graph['energy_source_TEA'].update(dict(
        # avg station_LCO per tonmi should be the total cost of the station (from peak value) over the battery tonmi
        station_LCO_tonmi=dict(zip(
            comm_list,
            [G.graph['energy_source_TEA']['station_LCO_kgh2'][c] * peak_hydrogen_energy_kgh2[c] / hydrogen_tonmi[c]
             for c in comm_list])),
        terminal_LCO_tonmi=dict(zip(
            comm_list,
            [G.graph['energy_source_TEA']['terminal_LCO_kgh2'][c] * peak_hydrogen_energy_kgh2[c] / hydrogen_tonmi[c]
             for c in comm_list])),
        liquefier_LCO_tonmi=dict(zip(
            comm_list,
            [G.graph['energy_source_TEA']['liquefier_LCO_kgh2'][c] * peak_hydrogen_energy_kgh2[c] / hydrogen_tonmi[c]
             for c in comm_list])),
        fuel_LCO_tonmi=dict(zip(
            comm_list,
            [G.graph['energy_source_TEA']['fuel_LCO_kgh2'][c] * avg_hydrogen_energy_kgh2[c] / hydrogen_tonmi[c]
             for c in comm_list])),
        tender_LCO_tonmi={c: tender_LCO_tonmi for c in comm_list},
        delay_LCO_tonmi=dict(zip(
            comm_list,
            [G.graph['energy_source_TEA']['delay_LCO_kgh2'][c] * avg_hydrogen_energy_kgh2[c] / hydrogen_tonmi[c]
             for c in comm_list]))
    ))

    G.graph['energy_source_TEA'].update(dict(
        total_LCO_tonmi={c: (G.graph['energy_source_TEA']['station_LCO_tonmi'][c] +
                             G.graph['energy_source_TEA']['terminal_LCO_tonmi'][c] +
                             G.graph['energy_source_TEA']['liquefier_LCO_tonmi'][c] +
                             G.graph['energy_source_TEA']['fuel_LCO_tonmi'][c] +
                             G.graph['energy_source_TEA']['tender_LCO_tonmi'][c] +
                             G.graph['energy_source_TEA']['delay_LCO_tonmi'][c]) for c in comm_list},
        total_scenario_LCO_tonmi={c: ((G.graph['energy_source_TEA']['station_LCO_tonmi'][c] +
                                       G.graph['energy_source_TEA']['terminal_LCO_tonmi'][c] +
                                       G.graph['energy_source_TEA']['liquefier_LCO_tonmi'][c] +
                                       G.graph['energy_source_TEA']['fuel_LCO_tonmi'][c] +
                                       G.graph['energy_source_TEA']['tender_LCO_tonmi'][c] +
                                       G.graph['energy_source_TEA']['delay_LCO_tonmi'][c]) * hydrogen_tonmi[c] +
                                      diesel_factor * support_diesel_gal[c]) / baseline_total_tonmi[c]
                                  for c in comm_list},
        total_nodelay_LCO_tonmi={c: (G.graph['energy_source_TEA']['station_LCO_tonmi'][c] +
                                     G.graph['energy_source_TEA']['terminal_LCO_tonmi'][c] +
                                     G.graph['energy_source_TEA']['liquefier_LCO_tonmi'][c] +
                                     G.graph['energy_source_TEA']['fuel_LCO_tonmi'][c] +
                                     G.graph['energy_source_TEA']['tender_LCO_tonmi'][c]) for c in comm_list},
        total_scenario_nodelay_LCO_tonmi={c: ((G.graph['energy_source_TEA']['station_LCO_tonmi'][c] +
                                               G.graph['energy_source_TEA']['terminal_LCO_tonmi'][c] +
                                               G.graph['energy_source_TEA']['liquefier_LCO_tonmi'][c] +
                                               G.graph['energy_source_TEA']['fuel_LCO_tonmi'][c] +
                                               G.graph['energy_source_TEA']['tender_LCO_tonmi'][c]) *
                                              hydrogen_tonmi[c] + diesel_factor *
                                              support_diesel_gal[c]) / baseline_total_tonmi[c] for c in comm_list},
        annual_hydrogen_total_cost={c: 365 * (G.graph['energy_source_TEA']['station_LCO_tonmi'][c] +
                                              G.graph['energy_source_TEA']['terminal_LCO_tonmi'][c] +
                                              G.graph['energy_source_TEA']['liquefier_LCO_tonmi'][c] +
                                              G.graph['energy_source_TEA']['fuel_LCO_tonmi'][c] +
                                              G.graph['energy_source_TEA']['tender_LCO_tonmi'][c]) * hydrogen_tonmi[c]
                                    for c in comm_list},
        annual_support_diesel_total_cost={c: 365 * diesel_factor * support_diesel_gal[c] for c in comm_list},
        annual_total_cost={c: 365 * ((G.graph['energy_source_TEA']['station_LCO_tonmi'][c] +
                                      G.graph['energy_source_TEA']['terminal_LCO_tonmi'][c] +
                                      G.graph['energy_source_TEA']['liquefier_LCO_tonmi'][c] +
                                      G.graph['energy_source_TEA']['fuel_LCO_tonmi'][c] +
                                      G.graph['energy_source_TEA']['tender_LCO_tonmi'][c] +
                                      G.graph['energy_source_TEA']['delay_LCO_tonmi'][c]) * hydrogen_tonmi[c] +
                                     diesel_factor * support_diesel_gal[c]) for c in comm_list}
    ))

    return G


def queue_model(lam: float, mu: float, s: int):
    # queuing model results for an M/M/s queue with <lam> arrival rate, <mu> service rate, and <s> many servers
    # s = 171 is max python can handle, at this point, set s = 150,
    # p0 approaches some value for fixed lam and mu for large s, s = 150 is large enough
    warnings.simplefilter('ignore')  # comment out to investigate for any issues

    s = int(s)

    if lam == 0:
        lq = 0
        wq = 0
    else:
        rho = lam / (s * mu) if s * mu != 0 else np.inf

        if rho >= 1:
            lq = np.inf
            wq = np.inf
        else:
            try:
                assert (lam / mu) ** s != np.inf

                p0 = 1 / (sum([((lam / mu) ** n) / math.factorial(n) for n in range(s)]) +
                          ((lam / mu) ** s) / (math.factorial(s) * (1 - rho)))
                lq = (p0 * rho * (lam / mu) ** s) / (math.factorial(s) * (1 - rho) ** 2)
                wq = lq / lam
            except (OverflowError, AssertionError):
                lq = 0
                wq = 0

    return lq, wq


'''
DROP-IN
'''


def load_tea_dropin_lookup():
    # return dataframe for cost factors: index is fuel_type, column is railroad, value is in [$/ton-mile]
    return pd.read_csv(os.path.join(TEA_DIR, 'tea_dropin_fuel_dollar_gal.csv'),
                       header=0, index_col='fuel_type')


def tea_diesel_mp(G: nx.DiGraph, time_horizon: list):
    for e in G.edges:
        G.edges[e]['diesel_TEA'] = {t: dict() for t in time_horizon}

    G.graph['diesel_TEA'] = {t: dict() for t in time_horizon}

    for t in time_horizon:
        G = tea_diesel_step_mp(G=G, time_step=t)

    return G


def tea_diesel_step_mp(G: nx.DiGraph, time_step: str):
    # return G with edge attribute with emissions info for <fuel_type> at <deployment_perc> w/ diesel as the baseline
    # load dataframe for cost factors: index is fuel_type, column is value in [$/gal]

    c_diesel = load_diesel_prices_mp().loc[int(time_step)].item()
    fuel_name = 'diesel'

    # if diesel_cost_p_gal is None:
    #     df_cost = load_tea_dropin_lookup()
    #     c_factor = df_cost.loc[fuel_type, '$/gal']
    #     c_diesel = df_cost.loc['diesel', '$/gal']
    # else:
    #     c_diesel = diesel_cost_p_gal
    #     if fuel_type == 'diesel':
    #         c_factor = diesel_cost_p_gal
    #     else:
    #         df_cost = load_tea_dropin_lookup()
    #         c_factor = df_cost.loc[fuel_type, '$/gal']
    #
    # weighted_c_factor = c_factor * deployment_perc + c_diesel * (1 - deployment_perc)

    # if fuel_type != scenario_fuel_type:
    #     fuel_name = fuel_type
    # else:
    #     fuel_name = 'energy_source'

    comm_list = list({c for u, v in G.edges for c in G.edges[u, v]['baseline_avg_gal'][time_step].keys()})
    for u, v in G.edges:
        e = G.edges[u, v]
        e_baseline_avg_gal = e['baseline_avg_gal'][time_step]
        tonmi = dict(zip(comm_list, [e['baseline_avg_ton'][time_step][c] * e['miles'] for c in comm_list]))
        G.edges[u, v][fuel_name + '_TEA'][time_step] = dict(
            fuel_LCO_tonmi=dict(zip(
                comm_list,
                [c_diesel * e_baseline_avg_gal[c] / tonmi[c] if tonmi[c] > 0 else 0 for c in comm_list])),
            total_LCO_tonmi=dict(zip(
                comm_list,
                [c_diesel * e_baseline_avg_gal[c] / tonmi[c] if tonmi[c] > 0 else 0 for c in comm_list])),
            daily_fuel_cost=dict(zip(
                comm_list,
                [c_diesel * e_baseline_avg_gal[c] for c in comm_list])),
            annual_fuel_cost=dict(zip(
                comm_list,
                [365 * c_diesel * e_baseline_avg_gal[c] for c in comm_list])),
            daily_total_cost=dict(zip(
                comm_list,
                [c_diesel * e_baseline_avg_gal[c] for c in comm_list])),
            annual_total_cost=dict(zip(
                comm_list,
                [365 * c_diesel * e_baseline_avg_gal[c] for c in comm_list]))
        )

    total_gal = dict(zip(comm_list,
                         [sum([G.edges[u, v]['baseline_avg_gal'][time_step][c] for u, v in G.edges])
                          for c in comm_list]))
    total_tonmi = dict(zip(comm_list,
                           [sum([G.edges[u, v]['baseline_avg_ton'][time_step][c] * G.edges[u, v]['miles']
                                 for u, v in G.edges]) for c in comm_list]))
    G.graph[fuel_name + '_TEA'][time_step] = dict(
        fuel_LCO_tonmi=dict(zip(
            comm_list,
            [c_diesel * total_gal[c] / total_tonmi[c] if total_tonmi[c] > 0 else 0
             for c in comm_list])),
        total_LCO_tonmi=dict(zip(
            comm_list,
            [c_diesel * total_gal[c] / total_tonmi[c] if total_tonmi[c] > 0 else 0 for c in comm_list])),
        daily_fuel_cost=dict(zip(
            comm_list,
            [sum([G.edges[u, v][fuel_name + '_TEA'][time_step]['daily_fuel_cost'][c] for u, v in G.edges])
             for c in comm_list])),
        annual_fuel_cost=dict(zip(
            comm_list,
            [sum([G.edges[u, v][fuel_name + '_TEA'][time_step]['annual_fuel_cost'][c] for u, v in G.edges])
             for c in comm_list])),
        annual_total_cost=dict(zip(
            comm_list,
            [sum([G.edges[u, v][fuel_name + '_TEA'][time_step]['annual_total_cost'][c] for u, v in G.edges])
             for c in comm_list])),
        daily_total_cost=dict(zip(
            comm_list,
            [sum([G.edges[u, v][fuel_name + '_TEA'][time_step]['daily_total_cost'][c] for u, v in G.edges])
             for c in comm_list]))
    )
    return G


def tea_dropin_mp(G: nx.DiGraph, time_horizon: list, fuel_type: str, deployment_perc: float,
                  scenario_fuel_type: str = None, diesel_cost_p_gal: float = None):

    for e in G.edges:
        G.edges[e][fuel_type + '_TEA'] = {t: dict() for t in time_horizon}

    G.graph[fuel_type + '_TEA'] = {t: dict() for t in time_horizon}

    for t in time_horizon:
        G = tea_dropin_step_mp(G=G, time_step=t, fuel_type=fuel_type, deployment_perc=deployment_perc,
                               scenario_fuel_type=scenario_fuel_type)

    return G


def tea_dropin_step_mp(G: nx.DiGraph, time_step: str, fuel_type: str, deployment_perc: float,
                       scenario_fuel_type: str = None, diesel_cost_p_gal: float = None):
    # return G with edge attribute with emissions info for <fuel_type> at <deployment_perc> w/ diesel as the baseline
    # load dataframe for cost factors: index is fuel_type, column is value in [$/gal]

    if scenario_fuel_type is None:
        scenario_fuel_type = fuel_type

    if diesel_cost_p_gal is None:
        df_cost = load_tea_dropin_lookup()
        c_factor = df_cost.loc[fuel_type, '$/gal']
        c_diesel = df_cost.loc['diesel', '$/gal']
    else:
        c_diesel = diesel_cost_p_gal
        if fuel_type == 'diesel':
            c_factor = diesel_cost_p_gal
        else:
            df_cost = load_tea_dropin_lookup()
            c_factor = df_cost.loc[fuel_type, '$/gal']

    weighted_c_factor = c_factor * deployment_perc + c_diesel * (1 - deployment_perc)

    if fuel_type != scenario_fuel_type:
        fuel_name = fuel_type
    else:
        fuel_name = 'energy_source'

    comm_list = list({c for u, v in G.edges for c in G.edges[u, v]['baseline_avg_gal'].keys()})
    for u, v in G.edges:
        e = G.edges[u, v]
        tonmi = dict(zip(comm_list, [e['baseline_avg_ton'][c] * e['miles'] for c in comm_list]))
        G.edges[u, v][fuel_name + '_TEA'] = dict(
            fuel_LCO_tonmi=dict(zip(
                comm_list,
                [c_factor * e['baseline_avg_gal'][c] / tonmi[c] if tonmi[c] > 0 else 0 for c in comm_list])),
            total_LCO_tonmi=dict(zip(
                comm_list,
                [weighted_c_factor * e['baseline_avg_gal'][c] / tonmi[c] if tonmi[c] > 0 else 0 for c in comm_list])),
            daily_fuel_cost=dict(zip(
                comm_list,
                [c_factor * e['baseline_avg_gal'][c] * deployment_perc for c in comm_list])),
            annual_fuel_cost=dict(zip(
                comm_list,
                [365 * c_factor * e['baseline_avg_gal'][c] * deployment_perc for c in comm_list])),
            daily_total_cost=dict(zip(
                comm_list,
                [weighted_c_factor * e['baseline_avg_gal'][c] for c in comm_list])),
            annual_total_cost=dict(zip(
                comm_list,
                [365 * weighted_c_factor * e['baseline_avg_gal'][c] for c in comm_list]))
        )

    total_gal = dict(zip(comm_list,
                         [sum([G.edges[u, v]['baseline_avg_gal'][c] for u, v in G.edges]) for c in comm_list]))
    total_tonmi = dict(zip(comm_list,
                           [sum([G.edges[u, v]['baseline_avg_ton'][c] * G.edges[u, v]['miles'] for u, v in G.edges])
                            for c in comm_list]))
    G.graph[fuel_name + '_TEA'] = dict(
        fuel_LCO_tonmi=dict(zip(
            comm_list,
            [c_factor * total_gal[c] / total_tonmi[c] if total_tonmi[c] > 0 else 0
             for c in comm_list])),
        total_LCO_tonmi=dict(zip(
            comm_list,
            [weighted_c_factor * total_gal[c] / total_tonmi[c] if total_tonmi[c] > 0 else 0 for c in comm_list])),
        daily_fuel_cost=dict(zip(
            comm_list,
            [sum([G.edges[u, v][fuel_name + '_TEA']['daily_fuel_cost'][c] for u, v in G.edges]) for c in comm_list])),
        annual_fuel_cost=dict(zip(
            comm_list,
            [sum([G.edges[u, v][fuel_name + '_TEA']['annual_fuel_cost'][c] for u, v in G.edges]) for c in comm_list])),
        annual_total_cost=dict(zip(
            comm_list,
            [sum([G.edges[u, v][fuel_name + '_TEA']['annual_total_cost'][c] for u, v in G.edges]) for c in comm_list])),
        daily_total_cost=dict(zip(
            comm_list,
            [sum([G.edges[u, v][fuel_name + '_TEA']['daily_total_cost'][c] for u, v in G.edges]) for c in comm_list]))
    )
    return G
