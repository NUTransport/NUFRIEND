from util import *
# MODULES
from helper import mmddyyyy_to_datetime, datetime_to_mmddyyyy


'''
LOAD CCWS FILES
'''


def load_ccws_df(filename: str):
    pkl_path = os.path.join(FLOW_DIR, filename[:-4] + '.pkl')

    # if pickled dataframe does not exist
    if not os.path.exists(pkl_path):
        # load from csv and pickle dataframe
        df = load_ccws_df_csv(filename)
        df.to_pickle(pkl_path)
    else:
        # unpickle dataframe
        df = pd.read_pickle(pkl_path)

    return df


def load_ccws_df_csv(filename: str):
    return pd.read_csv(os.path.join(FLOW_DIR, filename), header=0)


def df_repkl(df: pd.DataFrame, filename: str):
    # repickle dataframe df

    pkl_path = os.path.join(FLOW_DIR, filename[:-4] + '.pkl')
    df.to_pickle(pkl_path)


# TODO: generalize these methods for any kind of input O-D flow data?
def RR_SPLC_comm_grouping(filename: str, time_window=None) -> pd.DataFrame:
    # produce summary file of ton-miles, car-miles, and container-miles grouped by RR, O-D (SPLC Codes), and commodity
    # consider only waybills that are handled by a single railroad, such that
    # originating RR == terminating RR and the first interchange RR is empty
    # NOTE: Seems like samples report many single carload data, is this an erroneous input?
    # NOTE: returns average daily traffic for each OD pair reporting with in <time_window>

    df = load_ccws_df(filename)  # load CCWS dataframe

    class1_filename = os.path.join(FLOW_DIR, filename[:6] + '_summary_Class1_SPLC.pkl')
    if os.path.exists(class1_filename):
        df_class1 = pd.read_pickle(class1_filename)
        return df_class1
    # keep waybills within <time_window> only (inclusive)
    if time_window is None or time_window == (None, None):
        year = filename[2:6]
        time_window = ('0101' + year, '1231' + year)
    # convert to datetime.date objects
    time_window = (mmddyyyy_to_datetime(time_window[0]), mmddyyyy_to_datetime(time_window[1]))
    df['Waybill Date (mmddccyy)'] = df['Waybill Date (mmddccyy)'].apply(lambda x: mmddyyyy_to_datetime(x))
    # filter out data before and after <time_window> (inclusive)
    df = df[(df['Waybill Date (mmddccyy)'] >= time_window[0]) & (df['Waybill Date (mmddccyy)'] <= time_window[1])]

    # keep waybills that are handled by a single RR only; '    ' is the empty string used to record no interchange RR
    df_s = df[(df['Origin Railroad Alpha'] == df['Termination Railroad Alpha']) &
              (df['First Interchange RR Alpha'] == '    ')].copy()
    # make new column with general railroad name and strip whitespace from it
    df_s['Railroad'] = df_s['Origin Railroad Alpha'].apply(lambda x: x.strip())
    # replace variations of Canadian RR names with a single consistent name representing US operations
    df_s['Railroad'].replace({'CNUS': 'CN', 'GTC': 'CN', 'CPRS': 'CP', 'CPUS': 'CP', 'SOO': 'CP'}, inplace=True)
    # convert origin and termination BEA area codes to strings
    df_s['Origin SPLC'] = df_s['Origin SPLC'].apply(lambda x: str(x) if len(str(x)) > 5 else '0' + str(x))
    df_s['Destination SPLC'] = df_s['Destination SPLC'].apply(lambda x: str(x) if len(str(x)) > 5 else '0' + str(x))
    # combine origin and termination BEA codes to one code representing the O-D pair
    df_s['Origin-Destination SPLC'] = "'" + df_s['Origin SPLC'] + df_s['Destination SPLC']
    # create new cols for total distance statistics
    df_s['Total Distance Mean'] = df_s['Total Distance']
    df_s['Total Distance Standard Deviation'] = df_s['Total Distance']
    df_s['Total Distance Max'] = df_s['Total Distance']
    df_s['Total Distance Min'] = df_s['Total Distance']
    # columns to keep in summary output, drop all others
    cols = {'Railroad', 'Origin-Destination SPLC', 'Commodity Group Name',
            'Expanded Number of Samples', 'Expanded Tons', 'Expanded Carloads', 'Expanded Trailer/Container Count',
            'Expanded Ton-Miles', 'Expanded Car-Miles', 'Expanded Container-Miles',
            'Total Distance Mean', 'Total Distance Standard Deviation', 'Total Distance Max', 'Total Distance Min'}
    df_s.drop(columns=set(df_s.columns).difference(cols), inplace=True)
    # group and sum ton-miles and car-miles by RR, OD, and commodity group
    df_g = df_s.groupby(by=['Railroad', 'Origin-Destination SPLC', 'Commodity Group Name'])
    # create aggregation function
    agg_fxn = {'Expanded Number of Samples': np.sum, 'Expanded Tons': np.sum, 'Expanded Carloads': np.sum,
               'Expanded Trailer/Container Count': np.sum, 'Expanded Ton-Miles': np.sum, 'Expanded Car-Miles': np.sum,
               'Expanded Container-Miles': np.sum, 'Total Distance Mean': np.mean,
               'Total Distance Standard Deviation': np.std, 'Total Distance Max': np.max, 'Total Distance Min': np.min}
    # apply aggregation function
    df_g_agg = df_g.agg(agg_fxn)

    num_days = (time_window[1] - time_window[0]).days + 1
    avg_cols = ['Expanded Number of Samples', 'Expanded Tons', 'Expanded Carloads',
                'Expanded Trailer/Container Count', 'Expanded Ton-Miles',
                'Expanded Car-Miles', 'Expanded Container-Miles']
    # return df_class1
    for a in avg_cols:
        df_g_agg[a] = df_g_agg[a].div(num_days)

    # write summary results to csv
    df_g_agg.to_csv(os.path.join(FLOW_DIR, filename[:6] + '_summary_SPLC.csv'), index=True)

    # produce a separate summary file for Class I's only
    class_1 = ['BNSF', 'CN', 'CP', 'CSXT', 'KCS', 'NS', 'UP', 'WCAN', 'EAST', 'USA1']
    # class_1 = ['BNSF', 'CN', 'CP', 'CSXT', 'KCS', 'NS', 'UP']
    df_class1 = df_g_agg.loc[class_1]
    df_class1.to_csv(os.path.join(FLOW_DIR, filename[:6] + '_summary_Class1_SPLC.csv'),
                     index=True)
    df_class1.to_pickle(class1_filename)

    return df_class1


def RR_SPLC_comm_date_grouping(filename: str, time_window_list: list = None):
    # produce summary file of ton-miles, car-miles, and container-miles grouped by RR, O-D (SPLC Codes), and commodity
    # consider only waybills that are handled by a single railroad, such that
    # originating RR == terminating RR and the first interchange RR is empty
    # NOTE: Seems like samples report many single carload data, is this an erroneous input?
    # produces daily average for each time window in <time_window_list>

    class1_sum_file = os.path.join(FLOW_DIR, filename[:6] + '_summary_Class1_date_SPLC.csv')
    class1_sum_pkl_file = class1_sum_file[:-4] + '.pkl'
    if os.path.exists(class1_sum_file):
        if time_window_list and os.path.exists(class1_sum_pkl_file):
            return pd.read_pickle(class1_sum_pkl_file)

        df_class1 = pd.read_csv(class1_sum_file, header=0)
        df_class1['Waybill Date (mmddccyy)'] = df_class1['Waybill Date (mmddccyy)'].apply(lambda x:
                                                                                          str(x) if len(str(x)) > 7
                                                                                          else '0' + str(x))
        df_class1 = df_class1.groupby(
            by=['Railroad', 'Origin-Destination SPLC', 'Commodity Group Name', 'Waybill Date (mmddccyy)'])
        # create aggregation function
        agg_fxn = {'Expanded Number of Samples': np.sum, 'Expanded Tons': np.sum, 'Expanded Carloads': np.sum,
                   'Expanded Trailer/Container Count': np.sum, 'Expanded Ton-Miles': np.sum,
                   'Expanded Car-Miles': np.sum,
                   'Expanded Container-Miles': np.sum, 'Total Distance Mean': np.mean,
                   'Total Distance Standard Deviation': np.std, 'Total Distance Max': np.max,
                   'Total Distance Min': np.min}
        # apply aggregation function
        df_class1 = df_class1.agg(agg_fxn)
    else:
        df = load_ccws_df(filename)  # load CCWS dataframe
        df['Waybill Date (mmddccyy)'] = df['Waybill Date (mmddccyy)'].apply(lambda x:
                                                                            str(x) if len(str(x)) > 7 else '0' + str(x))
        # keep waybills that are handled by a single RR only;'    ' is the empty string used to record no interchange RR
        df_s = df[(df['Origin Railroad Alpha'] == df['Termination Railroad Alpha']) &
                  (df['First Interchange RR Alpha'] == '    ')].copy()
        # make new column with general railroad name and strip whitespace from it
        df_s['Railroad'] = df_s['Origin Railroad Alpha'].apply(lambda x: x.strip())
        # replace variations of Canadian RR names with a single consistent name representing US operations
        df_s['Railroad'].replace({'CNUS': 'CN', 'CPRS': 'CP', 'CPUS': 'CP', 'SOO': 'CP'}, inplace=True)
        # convert origin and termination BEA area codes to strings
        df_s['Origin SPLC'] = df_s['Origin SPLC'].apply(lambda x: str(x) if len(str(x)) > 5 else '0' + str(x))
        df_s['Destination SPLC'] = df_s['Destination SPLC'].apply(lambda x: str(x) if len(str(x)) > 5 else '0' + str(x))
        # combine origin and termination BEA codes to one code representing the O-D pair
        df_s['Origin-Destination SPLC'] = "'" + df_s['Origin SPLC'] + df_s['Destination SPLC']
        # create new cols for total distance statistics
        df_s['Total Distance Mean'] = df_s['Total Distance']
        df_s['Total Distance Standard Deviation'] = df_s['Total Distance']
        df_s['Total Distance Max'] = df_s['Total Distance']
        df_s['Total Distance Min'] = df_s['Total Distance']
        # columns to keep in summary output, drop all others
        cols = {'Waybill Date (mmddccyy)', 'Railroad', 'Origin-Destination SPLC', 'Commodity Group Name',
                'Expanded Number of Samples', 'Expanded Tons', 'Expanded Carloads', 'Expanded Trailer/Container Count',
                'Expanded Ton-Miles', 'Expanded Car-Miles', 'Expanded Container-Miles',
                'Total Distance Mean', 'Total Distance Standard Deviation', 'Total Distance Max', 'Total Distance Min'}
        df_s.drop(columns=set(df_s.columns).difference(cols), inplace=True)
        # group and sum ton-miles and car-miles by RR, OD, and commodity group
        df_g = df_s.groupby(by=['Railroad', 'Origin-Destination SPLC', 'Commodity Group Name',
                                'Waybill Date (mmddccyy)'])
        # create aggregation function
        agg_fxn = {'Expanded Number of Samples': np.sum, 'Expanded Tons': np.sum, 'Expanded Carloads': np.sum,
                   'Expanded Trailer/Container Count': np.sum, 'Expanded Ton-Miles': np.sum,
                   'Expanded Car-Miles': np.sum, 'Expanded Container-Miles': np.sum,
                   'Total Distance Mean': np.mean, 'Total Distance Standard Deviation': np.std,
                   'Total Distance Max': np.max, 'Total Distance Min': np.min}
        # apply aggregation function
        df_g_agg = df_g.agg(agg_fxn)

        # write summary results to csv
        df_g_agg.to_csv(os.path.join(FLOW_DIR, filename[:6] + '_summary_date_SPLC.csv'), index=True)

        # produce a separate summary file for Class I's only
        class_1 = ['BNSF', 'CN', 'CP', 'CSXT', 'KCS', 'NS', 'UP', 'WCAN', 'EAST', 'USA1']
        # class_1 = ['BNSF', 'CN', 'CP', 'CSXT', 'KCS', 'NS', 'UP']
        df_class1 = df_g_agg.loc[(class_1, slice(None, None, None))]
        df_class1.to_csv(os.path.join(FLOW_DIR, filename[:6] + '_summary_Class1_date_SPLC.csv'),
                         index=True)

    if time_window_list:
        date_tw_dict = dict()
        tw_len_dict = dict()
        for s, e in time_window_list:
            dates = [datetime_to_mmddyyyy(dt) for dt in
                     pd.date_range(start=mmddyyyy_to_datetime(s), end=mmddyyyy_to_datetime(e))]
            date_tw_dict.update({dt: 'S' + s + 'E' + e for dt in dates})
            tw_len_dict['S' + s + 'E' + e] = len(dates)

        df_class1.reset_index(inplace=True)
        df_class1['Time Window (SmmddccyyEmmddccyy)'] = \
            df_class1['Waybill Date (mmddccyy)'].apply(lambda x:
                                                       date_tw_dict[x] if x in date_tw_dict.keys() else np.nan)
        df_class1.dropna(subset=['Time Window (SmmddccyyEmmddccyy)'], inplace=True)
        df_class1['tw_len'] = df_class1['Time Window (SmmddccyyEmmddccyy)'].apply(lambda x: tw_len_dict[x])
        avg_cols = ['Expanded Number of Samples', 'Expanded Tons', 'Expanded Carloads',
                    'Expanded Trailer/Container Count', 'Expanded Ton-Miles',
                    'Expanded Car-Miles', 'Expanded Container-Miles']
        # return df_class1
        for a in avg_cols:
            df_class1[a] = df_class1[a].div(df_class1['tw_len'])
        df_class1.drop(columns=['Waybill Date (mmddccyy)', 'tw_len'], inplace=True)
        df_class1 = df_class1.groupby(by=['Railroad', 'Origin-Destination SPLC', 'Commodity Group Name',
                                          'Time Window (SmmddccyyEmmddccyy)'])
        # create aggregation function
        agg_fxn = {'Expanded Number of Samples': np.sum, 'Expanded Tons': np.sum, 'Expanded Carloads': np.sum,
                   'Expanded Trailer/Container Count': np.sum, 'Expanded Ton-Miles': np.sum,
                   'Expanded Car-Miles': np.sum, 'Expanded Container-Miles': np.sum,
                   'Total Distance Mean': np.mean, 'Total Distance Standard Deviation': np.std,
                   'Total Distance Max': np.max, 'Total Distance Min': np.min}
        # apply aggregation function
        df_class1 = df_class1.agg(agg_fxn)

        df_class1.to_pickle(class1_sum_file[:-4] + '.pkl')

    return df_class1
