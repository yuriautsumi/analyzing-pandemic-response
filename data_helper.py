""" Data helper for datasets. """

import os
import glob

import re
import copy
import math
import pickle
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import epyestim
import epyestim.covid19 as covid19

import utils
import config 
import tsa_helper
import constants as C

from tqdm import tqdm
from textwrap import wrap
from sodapy import Socrata
from bs4 import BeautifulSoup
from datetime import datetime
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector


## Map from months (1-12) to corresponding point on cosine curve
months = np.array(list(range(1,13)))
to_cos_period_domain = lambda arr: ((arr-arr.min())/arr.max()) * np.pi*2 - np.pi # scales linearly spaced data to -3.14 to 3.14
cos_period_domain = to_cos_period_domain(months)
cos_transformed = np.cos(cos_period_domain)
MONTH_TO_COS = dict(zip(months, cos_transformed)) # maps month (1-12) to corresponding point on cos curve


def _round(x):
    try:
        return round(x)
    except:
        return x

def _set_column_astype(df, column_to_dtype):
    dtypes = df.dtypes
    for col, dtype in column_to_dtype.items():
        dtypes.loc[col] = dtype
    return df.astype(dtypes)

def _infer_dtypes(df):
    # Fix float64 to int64 dtype if necessary
    dtypes = df.dtypes; is_null = df.isna().all()
    for col in df.columns[(dtypes == 'float64') & (~is_null)]:
        # if we can cast as int64 and still retain same values, then set as int64
        try:
            if (df.loc[:, col].dropna().astype('Int64') == df.loc[:, col].dropna()).all(): dtypes.loc[col] = 'Int64' # pd.Int64Dtype(), works with NaNs
        except TypeError: # rises when Int64 is not the float64 equivalent
            continue

    return df.astype(dtypes)

def convert_int64_Int64_to_float64(df):
    # Convert all int64 to float64 (so nan's are normal)
    dtypes = df.dtypes
    column_to_dtype = {}
    for x,y in zip(dtypes.index, dtypes):
        if y == 'int64' or y == 'Int64' or y == pd.Int64Dtype(): column_to_dtype[x] = 'float64'
    # print(column_to_dtype)
    return _set_column_astype(df, column_to_dtype)

def convert_Int64_Float64_to_int64_float64(df):
    # Convert all pandas type to int64/float64 (so nan's are normal)
    dtypes = df.dtypes
    column_to_dtype = {}
    for x,y in zip(dtypes.index, dtypes):
        if y == 'Int64' or y == pd.Int64Dtype(): column_to_dtype[x] = 'int64'
        elif y == 'Float64' or y == pd.Float64Dtype(): column_to_dtype[x] = 'float64'
    # print(column_to_dtype)
    return _set_column_astype(df, column_to_dtype)

def get_OxCGRT_data_loader():
    def _process_vax(df, vax_column):
        def _encode_vax_ternary(x):
            try:
                if f'{x}' == 'Under 50%' or f'{x}' == 'Over 50%': return x
                elif float(x) <= 50: return 'Under 50%'
                elif float(x) > 50: return 'Over 50%'
            except:
                return np.nan
            
        def _encode_vax_missing(x):
            if f'{x}' == 'Under 50%' or f'{x}' == 'Over 50%': return np.nan
            return x

        # create second column encoding >50%, <50% or missing (0,1,2)
        df[f'{vax_column}_encoded'] = df.loc[:, vax_column].apply(lambda x: _encode_vax_ternary(x))

        # fill <50%, >50% as missing
        df.loc[:, vax_column] = df.loc[:, vax_column].apply(lambda x: _encode_vax_missing(x))

        # fix column type
        return _set_column_astype(df, {vax_column: 'float64', f'{vax_column}_encoded': 'object'})

    def _get_vax_df(region):
        vax_df = pd.read_csv(C.REGION_TO_VACCINE_DATA_FILE[region])

        if region == 'national':
            vax_df = vax_df.loc[vax_df.location == C.COUNTRY_NAME]

        temp = vax_df.date.apply(lambda x: pd.to_datetime(x))
        vax_df = _set_column_astype(vax_df, {'date': 'datetime64[ns]'})
        vax_df.loc[:, 'date'], vax_df.index = temp, temp
        vax_df.index.name = None

        return vax_df.loc[:, ['date', 'location', 'people_fully_vaccinated_per_hundred', 'people_vaccinated_per_hundred']]

    # %matplotlib inline # returns plots without having to write plt.show()
    # compact_national = pd.read_csv('data/covid-policy-dataset-main/data/OxCGRT_compact_national_v1.csv')
    compact_subnational = pd.read_csv('data/covid-policy-dataset-main/data/OxCGRT_compact_subnational_v1.csv', low_memory=False).infer_objects()
    compact_subnational = _process_vax(compact_subnational, 'PopulationVaccinated')
    # compact_subnational = _set_column_astype(compact_subnational, {'PopulationVaccinated': 'float64'}) # fix column type

    def load_OxCGRT_data(country):
        print('Loading OxCGRT pandemic response data...')
        country_subnational_df = compact_subnational.loc[compact_subnational.CountryCode == country]

        # Convert to datetime
        temp = country_subnational_df.Date.apply(lambda x: datetime.strptime(f'{x}', '%Y%m%d')).infer_objects()
        # dtypes = country_subnational_df.dtypes; dtypes.loc['Date'] = 'datetime64[ns]'
        # country_subnational_df = country_subnational_df.astype(dtypes)
        country_subnational_df = _set_column_astype(country_subnational_df, {'Date': 'datetime64[ns]'})
        country_subnational_df.loc[:, 'Date'] = temp # change type first, then set

        # # Make float 
        # country_subnational_df.loc[:, 'PopulationVaccinated'] = country_subnational_df['PopulationVaccinated'].apply(lambda x: float(x))

        # Make np.nan
        country_subnational_df.loc[country_subnational_df.RegionCode.isna(), 'RegionCode'] = np.nan

        # Select dataframes
        country_national_df = country_subnational_df.loc[country_subnational_df.RegionCode.isna()]
        country_subnational_df = country_subnational_df.loc[~country_subnational_df.index.isin(country_national_df.index)]
        
        # Add vax data 
        if C.ADD_VAX_DATA:
            # Percentage of people vaccinated per day per state, country 
            national_vax_df = _get_vax_df('national')
            subnational_vax_df = _get_vax_df('subnational')

            # Merge with data
            country_national_df = pd.merge(
                left=country_national_df,
                right=national_vax_df,
                left_on=['Date'],
                right_on=['date'],
                how='left'
            )
            country_subnational_df = pd.merge(
                left=country_subnational_df,
                right=subnational_vax_df,
                left_on=['Date', 'RegionName'],
                right_on=['date', 'location'],
                how='left'
            )

            # Assume missing = 0% (vaccine not available)

        # Make index datetime
        country_subnational_df.index = pd.to_datetime(country_subnational_df.Date)
        country_national_df.index = pd.to_datetime(country_national_df.Date)

        # Infer dtypes
        country_national_df = _infer_dtypes(country_national_df)
        country_subnational_df = _infer_dtypes(country_subnational_df)

        # Group by region
        country_subnational_by_region = country_subnational_df.groupby('RegionCode')
        assert country_subnational_by_region.size().all() # all same length, 1096
        print('Done!')

        return country_national_df, country_subnational_df, country_subnational_by_region
    
    return load_OxCGRT_data

def load_mobility_data():
    print('Loading Google Mobility data...')
    # mobility_csvs = glob.glob('Region_Mobility_Report_CSVs/*')
    # set([x.split('/')[-1].split('_')[0] for x in mobility_csvs]) # 2020, 2021, 2022
    mobility_csvs = [
        'data/Region_Mobility_Report_CSVs/2020_US_Region_Mobility_Report.csv',
        'data/Region_Mobility_Report_CSVs/2021_US_Region_Mobility_Report.csv',
        'data/Region_Mobility_Report_CSVs/2022_US_Region_Mobility_Report.csv'
    ]

    # Load and concatenate 
    dfs = [pd.read_csv(csv) for csv in mobility_csvs]
    df = pd.concat(dfs)

    # Separate by region level
    national_df = df.loc[df.sub_region_1.isna() & df.sub_region_2.isna()]
    subnational_df = df.loc[~df.sub_region_1.isna() & df.sub_region_2.isna()]
    subsubnational_df = df.loc[~df.sub_region_1.isna() & ~df.sub_region_2.isna()]

    # Group by region
    subnational_by_region = subnational_df.groupby('sub_region_1')
    assert subnational_by_region.size().all() # all same length, 974

    # Check that dates are equal
    all_dates = [x[1].reset_index(drop=True) for x in subnational_by_region.date] + [national_df.date.reset_index(drop=True)]
    assert all([all_dates[0].equals(x) for x in all_dates[1:]])

    # Make index datetime
    national_df = _set_column_astype(national_df, {'date': 'datetime64[ns]'})
    subnational_df = _set_column_astype(subnational_df, {'date': 'datetime64[ns]'})
    subsubnational_df = _set_column_astype(subsubnational_df, {'date': 'datetime64[ns]'})
    national_df.loc[:, 'date'], national_df.index = pd.to_datetime(national_df.date), pd.to_datetime(national_df.date)
    subnational_df.loc[:, 'date'], subnational_df.index = pd.to_datetime(subnational_df.date), pd.to_datetime(subnational_df.date)
    subsubnational_df.loc[:, 'date'], subsubnational_df.index = pd.to_datetime(subsubnational_df.date), pd.to_datetime(subsubnational_df.date)

    # Infer dtypes
    national_df = _infer_dtypes(national_df)
    subnational_df = _infer_dtypes(subnational_df)
    subsubnational_df = _infer_dtypes(subsubnational_df)

    # Group by region
    subnational_by_region = subnational_df.groupby('sub_region_1')
    subsubnational_by_region = subsubnational_df.groupby('sub_region_2')

    print('Done!')

    return {
        'national_df': national_df,
        'subnational_df': subnational_df,
        'subsubnational_df': subsubnational_df,
        'subsubnational_by_region': subsubnational_by_region, 
        'subnational_by_region': subnational_by_region, 
        'subsubnational_by_region': subsubnational_by_region
    }
    # return national_df, subnational_df, subsubnational_df, subsubnational_by_region, subnational_by_region, subsubnational_by_region

def load_demographic_data():
    def _load(table_to_file):
        selected_dfs = []
        for table, (index_sel, column_sel) in table_to_selection.items():
            df = pd.read_csv(table_to_file[table])

            # Process row index
            df.loc[:, 'Label (Grouping)'] = df.loc[:, 'Label (Grouping)'].apply(lambda s: s.strip('\xa0'))
            df.set_index('Label (Grouping)', inplace=True)

            # Process columns
            multiindex = all(list(map(lambda x: '!!' in x, df.columns)))
            if multiindex:
                n_subvalues = len(df.columns[0].split('!!'))-1
                df.columns = pd.MultiIndex.from_tuples(list(map(lambda c: tuple(c.split('!!')), df.columns)), names=['state'] + [f'subvalue{i+1}' for i in range(n_subvalues)])

            temp = df.loc[index_sel, column_sel]
            if type(column_sel)==list: temp.index = map(lambda x: f'{x} ({column_sel[1]})', temp.index)
            if type(temp.columns) == pd.MultiIndex: temp.columns = temp.columns.get_level_values('state')
            if len(df.index.unique()) < len(df.index): temp = temp.iloc[0::2]
            selected_dfs.append(temp)

        select_demographic_df = pd.concat(selected_dfs, axis=0)
        select_demographic_df = select_demographic_df.apply(lambda series: list(map(lambda x: float(x.replace(',', '').strip('%')), series)))
        X_demographic = select_demographic_df.T
        X_demographic.rename(columns={'Total:': 'Population'}, inplace=True)
        
        return X_demographic

    print('Loading demographic data from US Census tables...')
    # 1. Specify tables and columns 
    table_to_file_subnational = {
        'population': './data/US Census/2020/DECENNIALCD1182020.P8-2024-03-03T081542.csv', # population, Decennial P8
        'age_breakdown': './data/US Census/2020/ACSDP5Y2020.DP05-2024-03-03T081533.csv', # breakdown per age, ACS DP 5Y, DP05
        'poverty_employment': './data/US Census/2020/ACSST5Y2020.S1701-2024-03-03T081423.csv', # poverty breakdown per age, employment status, ACS ST 5Y, S1701
        'income': './data/US Census/2020/ACSST5Y2020.S1901-2024-03-03T081437.csv', # income breakdown, ACS ST 5Y, S1901
        'insurance': './data/US Census/2020/ACSST5Y2020.S2701-2024-03-03T081507.csv' # insurance / poverty, ACS ST 5Y, S2701
    }
    table_to_file_national = {
        'population': './data/US Census/2020/DECENNIALDHC2020.P8-2024-03-17T110215.csv', # population, Decennial P8
        'age_breakdown': './data/US Census/2020/ACSDP5Y2020.DP05-2024-03-17T110022.csv', # breakdown per age, ACS DP 5Y, DP05
        'poverty_employment': './data/US Census/2020/ACSST5Y2020.S1701-2024-03-17T110055.csv', # poverty breakdown per age, employment status, ACS ST 5Y, S1701
        'income': './data/US Census/2020/ACSST5Y2020.S1901-2024-03-17T110107.csv', # income breakdown, ACS ST 5Y, S1901
        'insurance': './data/US Census/2020/ACSST5Y2020.S2701-2024-03-17T110129.csv' # insurance / poverty, ACS ST 5Y, S2701
    }
    All = slice(None)
    table_to_selection = {
        'population': (['Total:'], (All)),
        'age_breakdown': (['65 years and over', 'White', 'Black or African American', 'Hispanic or Latino (of any race)'], (All, 'Percent')),
        'poverty_employment': (['Population for whom poverty status is determined'], (All, "Percent below poverty level", 'Estimate')),
        'income': (['Less than $10,000', '$10,000 to $14,999',
            '$15,000 to $24,999', '$25,000 to $34,999', '$35,000 to $49,999',
            'Median income (dollars)'], (All, "Households", 'Estimate')),
        'insurance': (['Civilian noninstitutionalized population'], (All, "Percent Uninsured", 'Estimate'))
    }

    # 2. Select specified data (see above) and create N_features x (N_states) dataframe of demographic data
    X_demographic_subnational = _load(table_to_file_subnational)
    X_demographic_national = _load(table_to_file_national)

    # Infer dtypes
    X_demographic_subnational = _infer_dtypes(X_demographic_subnational)
    X_demographic_national = _infer_dtypes(X_demographic_national)

    print('Done!')
    return X_demographic_subnational, X_demographic_national
    
def load_hospital_data(data_type, add_admission=True):
    print('Loading hospital data...')
    data_type_to_id_name = {
        'capacity-by-state-raw': ('g62h-syeh', 'date', 'COVID-19-Reported-Patient-Impact-and-Hospital-Capacity-by-State-RAW'),
        'capacity-by-facility-raw': ('uqq2-txqb', 'collection_week', 'COVID-19-Reported-Patient-Impact-and-Hospital-Capacity-by-Facility-RAW'),
        'capacity-by-facility': ('anag-cw7u', 'collection_week', 'COVID-19-Reported-Patient-Impact-and-Hospital-Capacity-by-Facility'),
        'coverage-meta': ('ewep-8fwa', 'update_date', 'COVID-19-Hospital-Data-Coverage-Report'),
        'coverage': (None, None, '20240301_Hospital_Data_Coverage_Report_og'),
    }
    data_id, time_col, data_name = data_type_to_id_name[data_type]
    
    # Load hospital data:
    # Check if file exists
    hospital_dir = './data/hospital'
    os.makedirs(hospital_dir, exist_ok=True)
    pkl_file = os.path.join(hospital_dir, f'{data_name}.pkl') if data_id is None else os.path.join(hospital_dir, f'{data_name}-{data_id}.pkl') # processed
    if os.path.isfile(pkl_file): 
        print('File was found. Unpickling...')
        with open(pkl_file, 'rb') as f:
            hospital_df = pickle.load(f)
        print('Done!')
    else:
        print('File was not found. \nRequesting data from Healthdata.gov...')
        # Unauthenticated client only works with public data sets. Note 'None'
        # in place of application token, and no username or password:
        client = Socrata("healthdata.gov", None)

        # Returned as JSON from API / converted to Python list of
        # dictionaries by sodapy.
        results = client.get(f"{data_id}")#, limit=2000)

        # Convert to pandas DataFrame
        hospital_df = pd.DataFrame.from_records(results)

        # Format date
        if time_col is not None:
            hospital_df.sort_values(by=time_col, inplace=True)
            hospital_df[time_col] = hospital_df[time_col].apply(lambda x: pd.to_datetime(datetime.fromisoformat(x.replace('.000', ''))))
            hospital_df.index = hospital_df[time_col]

        # Save to file
        print('Saving to csv and pickle...')
        csv_file = os.path.join(hospital_dir, f'{data_name}-{data_id}.csv')
        hospital_df.to_csv(csv_file)
        with open(pkl_file, 'wb') as f:
            pickle.dump(hospital_df, f)
        print('Done!')

    # Load admission data
    if add_admission:
        # Check if file exists
        hospital_admission_dir = 'data/hospital-admission-by-state'
        pkl_file = os.path.join(hospital_admission_dir, f'hospital_admission.pkl') # processed
        if os.path.isfile(pkl_file): 
            print('File was found. Unpickling...')
            with open(pkl_file, 'rb') as f:
                hospital_admission_df = pickle.load(f)
            print('Done!')
        else:
            print('File was not found. \nProcessing data...')
            # Hospital admission per week per state (from ~Aug '20)
            files = glob.glob(os.path.join(hospital_admission_dir, f'*.csv'))

            # Load data for each state
            state_to_data = {}
            for state_csv in files: 
                with open(state_csv, 'r') as f:
                    title = f.readline()
                    state = title.split(' - ')[-1]
                    f.readline()
                    x = f.readlines()
                    state_to_data[state] = np.array([l.strip('\n').split(',') for l in x])

            # Process data to df for each state
            all_state_dfs = []
            for state, state_data in state_to_data.items(): 
                state_df = pd.DataFrame(
                    state_data[1:,:],
                    columns=state_data[0]
                )
                state_df.Date = state_df.Date.apply(lambda x: datetime.strptime(x, '%b %d %Y'))
                state_df.index = state_df.Date
                state_df.loc[:, 'Weekly COVID-19 Hospital Admissions'] = state_df['Weekly COVID-19 Hospital Admissions'].replace('N/A', np.nan).astype('float64')
                all_state_dfs.append(state_df)

            # Upsample to get daily data
            for i, state_df in enumerate(all_state_dfs):
                state = state_df.Geography.iloc[0]; state_df = state_df.loc[:, ['Weekly COVID-19 Hospital Admissions']]
                state_df = state_df.resample('D').ffill()
                state_df['Geography'] = state
                state_df['Date'] = state_df.index
                all_state_dfs[i] = state_df
                # all_admission_df.append(state_admission_df)
            
            hospital_admission_df = pd.concat(all_state_dfs)
    else:
        hospital_admission_df = None
    
    # Infer dtypes
    hospital_df = _infer_dtypes(hospital_df)

    return hospital_df, hospital_admission_df

def _load_population_density(data_dir):
    population_density_csv = os.path.join(data_dir, 'US_Population_Density_simple_weighted.csv')
    population_density_pkl = os.path.join(data_dir, 'US_Population_Density_simple_weighted.pkl')

    if os.path.isfile(population_density_pkl):
        with open(population_density_pkl, 'rb') as f:
            population_density_df = pickle.load(f)
    else:
        url = 'https://wernerantweiler.ca/blog.php?item=2020-04-12'
        page = requests.get(url)
        assert page.status_code == 200

        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table')

        columns = ['State', 'Code', 'Population [million]', 'Land Area [1000 km2]', 'Population Density - simple [per km2]', 'Population Density  - weighted [per km2]', 'Difference [%]']

        # Obtain every row with tag <td>
        rows = []
        for i in table.find_all('td'):
            rows.append(i.text.replace('\xa0', ''))

        population_density_df = pd.DataFrame(
            [rows[i*7:(i+1)*7] for i in range(len(rows)//7)],
            columns=columns
        )
        population_density_df[columns[2:-1]] = population_density_df[columns[2:-1]].astype('float64')
        population_density_df[columns[-1]] = population_density_df[columns[-1]].map(lambda x: float(x.replace('%', '')))

        population_density_df.to_csv(population_density_csv)
        with open(population_density_pkl, 'wb') as f:
            pickle.dump(population_density_df, f)

    population_density_df.index = population_density_df.State

    # Infer dtypes
    population_density_df = _infer_dtypes(population_density_df)

    return population_density_df[['Population Density  - weighted [per km2]']]

def _load_political_leaning(data_dir):
    political_leaning_csv = os.path.join(data_dir, 'US_Political_Leaning_by_State_2020_Election.csv')
    political_leaning_pkl = os.path.join(data_dir, 'US_Political_Leaning_by_State_2020_Election.pkl')

    if os.path.isfile(political_leaning_pkl):
        with open(political_leaning_pkl, 'rb') as f:
            political_leaning_df = pickle.load(f)
    else:
        url = "https://en.wikipedia.org/wiki/2020_United_States_presidential_election#cite_note-FEC-2"
        page = requests.get(url)
        assert page.status_code == 200

        # Create a BeautifulSoup object to parse the HTML content
        soup = BeautifulSoup(page.text, 'html.parser')

        # Find the table with the caption "Results by state"
        table = soup.find(id='Results_by_state').find_next('div')

        # Find all the rows in the table
        rows = table.find_all("tr")

        # Iterate over the rows and extract the data
        all_rows = []
        for row in rows:
            # Find all the cells in the row
            cells = row.find_all("td")
            
            # Extract the data from each cell
            data = [cell.text.strip() for cell in cells]
            if len(data)>0: all_rows.append(data)

        # Create dataframe, with US %ages
        political_leaning_df = pd.DataFrame(
            [[xi for xi in x] for x in all_rows],
            # [[x[0], float(x[2].strip('%')), float(x[5].strip('%'))] for x in all_rows],
            # columns=['State', 'Democratic', 'Republican']
        )
        democratic_votes = political_leaning_df.iloc[:, 1].apply(lambda x: int(x.replace(',', '')))
        republican_votes = political_leaning_df.iloc[:, 4].apply(lambda x: int(x.replace(',', '')))
        total_votes = political_leaning_df.iloc[:, -1].apply(lambda x: int(x.replace(',', '')))
        democratic_pct = round(democratic_votes.sum() / total_votes.sum() * 100, 2)
        republican_pct = round(republican_votes.sum() / total_votes.sum() * 100, 2)

        political_leaning_df = political_leaning_df.loc[:, [0,2,5]]
        political_leaning_df.columns = ['State', 'Democratic', 'Republican']
        political_leaning_df.loc[:, 'Democratic'] = political_leaning_df.Democratic.apply(lambda x: float(x.strip('%')))
        political_leaning_df.loc[:, 'Republican'] = political_leaning_df.Republican.apply(lambda x: float(x.strip('%')))
        political_leaning_df.loc[-1] = ['United States', democratic_pct, republican_pct]

        # political_leaning_df = pd.DataFrame(
        #     [[x[0], float(x[2].strip('%')), float(x[5].strip('%'))] for x in all_rows],
        #     columns=['State', 'Democratic', 'Republican']
        # )
        political_leaning_df.State = political_leaning_df.State.apply(lambda x: C.STATE_ABBREVIATION_TO_NAME.get(x, np.nan))
        political_leaning_df = political_leaning_df.loc[~political_leaning_df.State.isna()].reset_index(drop=True)

        political_leaning_df.to_csv(political_leaning_csv)
        with open(political_leaning_pkl, 'wb') as f:
            pickle.dump(political_leaning_df, f)
    
    return political_leaning_df

def load_misc_data():
    misc_data_dir = './data/Misc/'
    os.makedirs(misc_data_dir, exist_ok=True)

    population_density_df = _load_population_density(misc_data_dir)
    political_leaning_df = _load_political_leaning(misc_data_dir)

    misc_df = pd.merge(
        left=political_leaning_df,
        right=population_density_df,
        left_on='State',
        right_on='State'
    )
    misc_df.index = misc_df.State

    return misc_df

def filter_data(data_to_df):
    for data_key, df in data_to_df.items():
        if type(df) == pd.DataFrame:
            # Remove any column containing value in config.columns_to_remove
            filtered_columns = list(filter(lambda col: all([c not in col for c in config.columns_to_remove]), df.columns))
            data_to_df[data_key] = df[filtered_columns]
        else:
            for k,v in df.items():
                if type(v) != pd.DataFrame: continue
                filtered_columns = list(filter(lambda col: all([c not in col for c in config.columns_to_remove]), v.columns))
                data_to_df[data_key][k] = v[filtered_columns]
    return data_to_df

"""
    for key, key_columns_to_remove in config.columns_to_remove.items():
        data = data_to_df[key]
        if type(data) == dict:
            data_list = [data['national_df'], data['subnational_df']]
        else:
            data_list = [data]

        for i, data_i in enumerate(data_list):
            try:
                data_i.drop(columns=key_columns_to_remove, inplace=True)
            except KeyError:
                print('Did not remove all columns. Please check the column names.')
            data_list[i] = data_i
        
        if len(data_list)==1: data_to_df[key] = data_list[0]
        else: data_to_df[key]['national_df'], data_to_df[key]['subnational_df'] = data_list
    return data_to_df
"""
"""
    for k,v in data_to_column_map.items():
        for k2,v2 in v.items():
            v2_dict = None
            if len(v2)==0: continue
            elif type(v2) == dict: continue
            if len(v2)>1: 
                if type(v2[1]) == dict:
                    v2,v2_dict = v2
            v2_filtered = list(filter(lambda col: all([c not in col for c in config.columns_to_remove]), v2))
            if v2_dict is None: data_to_column_map[k][k2] = v2_filtered
            else: data_to_column_map[k][k2] = (v2_filtered, v2_dict)
    return data_to_column_map
"""

def merge_data(data_to_df):
    # Combine dataframes: OxCGRT, US Census, Mobility, Hospital
    mobility_data_dict = data_to_df['mobility_data_dict']
    hospital_coverage_df = data_to_df['hospital_coverage_df']
    hospital_admission_df = data_to_df['hospital_admission_df']
    X_demographic_subnational = data_to_df['X_demographic_subnational']
    X_demographic_national = data_to_df['X_demographic_national']
    national_OxCGRT_df = data_to_df['national_OxCGRT_df']
    subnational_OxCGRT_df = data_to_df['subnational_OxCGRT_df']
    misc_df = data_to_df['misc_df']

    ## Part 1: Combine subnational data

    # Select only regions in OxCGRT, US Census, and Mobility
    region_names = np.intersect1d(
        np.intersect1d(X_demographic_subnational.index, subnational_OxCGRT_df.RegionName.unique()),
        mobility_data_dict['subnational_df'].sub_region_1.unique()
    )
    subnational_OxCGRT_df = subnational_OxCGRT_df.loc[subnational_OxCGRT_df.RegionName.isin(region_names)]
    X_demographic_subnational = X_demographic_subnational.loc[region_names]
    subnational_mobility_df = mobility_data_dict['subnational_df']
    subnational_mobility_df = subnational_mobility_df.loc[subnational_mobility_df.sub_region_1.isin(region_names)]

    # Merge on subnational region and date -> subnational_df
    temp = pd.merge(
        left=subnational_OxCGRT_df.reset_index(drop=True).add_prefix('OxCGRT_'),
        right=subnational_mobility_df.reset_index(drop=True).add_prefix('Mobility_'),
        how='left',
        left_on=['OxCGRT_RegionName', 'OxCGRT_Date'],
        right_on=['Mobility_sub_region_1', 'Mobility_date']
    )
    subnational_df = pd.merge(
        left=temp,
        right=X_demographic_subnational.add_prefix('Demographic_'),
        how='left',
        left_on='OxCGRT_RegionName',
        right_index=True
    )
    subnational_df = pd.merge(
        left=subnational_df,
        right=misc_df.add_prefix('Misc_'),
        how='left',
        left_on='OxCGRT_RegionName',
        right_on='Misc_State'
    )

    # Process and add hospital data
    regionID_to_regionName = \
        dict(zip(
            subnational_OxCGRT_df.RegionCode.apply(lambda x: x.split('_')[1]),
            subnational_OxCGRT_df.RegionName,
        ))
    # Get coverage data only (Bed count, # hospitals per 100_000)
    hospital_bed_count_by_state = hospital_coverage_df.groupby('State').sum(numeric_only=True)['Certified Bed Count']
    hospital_num_hospitals_by_state = hospital_coverage_df.groupby('State').count()['CCN']
    hospital_coverage_df = pd.merge(
        left=pd.DataFrame(hospital_bed_count_by_state),
        right=pd.DataFrame(hospital_num_hospitals_by_state),
        left_index=True,
        right_index=True
    ).add_prefix('Hospital_')

    hospital_coverage_df.index = map(lambda x: regionID_to_regionName.get(x, np.nan), hospital_coverage_df.index)
    hospital_coverage_df = hospital_coverage_df.loc[~hospital_coverage_df.index.isna()] # Removes US territories
    hospital_coverage_df.sort_index(inplace=True)

    hospital_coverage_df['Hospital_N_CCN_per_100_000'] = hospital_coverage_df['Hospital_CCN'] / X_demographic_subnational['Population'].sort_index() * 1e5
    hospital_coverage_df['Hospital_Certified_Bed_Count_per_100_000'] = hospital_coverage_df['Hospital_Certified Bed Count'] / X_demographic_subnational['Population'].sort_index() * 1e5

    subnational_df = pd.merge(
        left=subnational_df,
        right=hospital_coverage_df,
        left_on='OxCGRT_RegionName',
        right_index=True
    )

    # Get admission data (admits that week, # admits that week per 100_000, ratio between admission & bed count per 100_000)
    subnational_df = pd.merge(
        left=subnational_df,
        right=hospital_admission_df.add_prefix('Hospital_'),
        left_on=['OxCGRT_RegionName', 'OxCGRT_Date'],
        right_on=['Hospital_Geography', 'Hospital_Date']
    )

    subnational_df['Hospital_Weekly COVID-19 Hospital Admissions_per_100_000'] = subnational_df['Hospital_Weekly COVID-19 Hospital Admissions'] / subnational_df['Demographic_Population'] * 1e5
    subnational_df['Hospital_Weekly COVID-19 Hospital Admissions_per_100_000_to_Certified_Bed_Count_per_100_000'] = \
        (subnational_df['Hospital_Weekly COVID-19 Hospital Admissions_per_100_000'] / subnational_df['Hospital_Certified_Bed_Count_per_100_000']) * 100
    

    ## Part 2: Combine national data
    # Note: Adding mobility, demographics data. No hospital data (not available).
    #       Demographics data only for comparison. 
    #       We won't actually use this as training data bc we would need other country data as well. 

    # Merge on date -> national_df
    national_mobility_df = mobility_data_dict['national_df']
    temp = pd.merge(
        left=national_OxCGRT_df.add_prefix('OxCGRT_'),
        right=national_mobility_df.add_prefix('Mobility_'),
        how='left',
        left_index=True,
        right_index=True
    )
    national_df = temp.copy()
    temp2 = X_demographic_national.add_prefix('Demographic_')
    national_df[temp2.columns] = np.tile(temp2.values, (len(national_df),1)) # add demographic data
    national_df[[
        f'Misc_{x}' for x in misc_df.columns
    ]] = misc_df.loc[C.COUNTRY_NAME].values
    # national_df[f'Misc_{misc_df.loc[C.COUNTRY_NAME].index[0]}'] = misc_df.loc[C.COUNTRY_NAME, 'Population Density  - weighted [per km2]'].item() # add weighted population density

    # Add Hospital coverage data (Bed count, # hospitals per 100_000)
    national_df['Hospital_N_CCN_per_100_000'] = hospital_coverage_df.sum(axis=0)['Hospital_CCN'] / X_demographic_national['Population'].item() * 1e5
    national_df['Hospital_Certified_Bed_Count_per_100_000'] = hospital_coverage_df.sum(axis=0)['Hospital_Certified Bed Count'] / X_demographic_national['Population'].item() * 1e5

    # Add admission data (admits that week, # admits that week per 100_000, ratio between admission & bed count per 100_000)
    national_df = pd.merge(
        left=national_df,
        right=hospital_admission_df.loc[hospital_admission_df.Geography == C.COUNTRY_NAME].add_prefix('Hospital_'),
        left_on='OxCGRT_Date',
        right_index=True
    )

    national_df['Hospital_Weekly COVID-19 Hospital Admissions_per_100_000'] = national_df['Hospital_Weekly COVID-19 Hospital Admissions'] / national_df['Demographic_Population'] * 1e5
    national_df['Hospital_Weekly COVID-19 Hospital Admissions_per_100_000_to_Certified_Bed_Count_per_100_000'] = \
        (national_df['Hospital_Weekly COVID-19 Hospital Admissions_per_100_000'] / national_df['Hospital_Certified_Bed_Count_per_100_000']) * 100

    ## Part 3: Post-processing, get map of columns
    # Drop unneeded columns
    to_drop = subnational_df.columns[subnational_df.isna().all()]
    subnational_filtered_columns = list(filter(lambda x: x not in to_drop, subnational_df.columns))
    national_filtered_columns = list(filter(lambda x: x not in to_drop, national_df.columns))

    subnational_df = subnational_df.loc[:, subnational_filtered_columns]
    national_df = national_df.loc[:, national_filtered_columns]

    return subnational_df.reset_index(drop=True), national_df.reset_index(drop=True)

def get_OxCGRT_columns(columns):
    def _process_policy_columns(policy_columns):
        policy_identifiers = [x.split('_')[1] for x in policy_columns]
        policy_columns_, policy_id_to_flag = [], {}
        prev_pid = ''
        for pid, pcol in zip(policy_identifiers, policy_columns):
            if pid == prev_pid: 
                policy_id_to_flag[pid] = True
            else: 
                policy_columns_.append(pcol)
                if prev_pid not in policy_id_to_flag: policy_id_to_flag[prev_pid] = False

                prev_pid = pid

        del policy_id_to_flag['']
        if pid not in policy_id_to_flag: policy_id_to_flag[pid] = False

        return policy_columns_, policy_id_to_flag

    # Known columns
    location_columns = ['OxCGRT_CountryName', 'OxCGRT_CountryCode', 'OxCGRT_RegionName', 'OxCGRT_RegionCode']
    if 'CityName' in columns: location_columns += ['OxCGRT_CityName', 'OxCGRT_CityCode']
    policy_jurisdiction_columns = ['OxCGRT_Jurisdiction']
    date_columns = ['OxCGRT_Date']
    outcome_columns = ['OxCGRT_ConfirmedCases', 'OxCGRT_ConfirmedDeaths']
    if C.ADD_VAX_DATA: vax_status_columns = ['OxCGRT_people_vaccinated_per_hundred', 'OxCGRT_people_fully_vaccinated_per_hundred']
    else: vax_status_columns = ['OxCGRT_MajorityVaccinated', 'OxCGRT_PopulationVaccinated', 'OxCGRT_PopulationVaccinated_encoded']

    # Extract policy columns by category (same for both tables)
    c_pattern = re.compile(r'OxCGRT_C[0-9][A-Z]*')  # Compile the regular expression pattern
    e_pattern = re.compile(r'OxCGRT_E[0-9][A-Z]*') 
    h_pattern = re.compile(r'OxCGRT_H[0-9][A-Z]*') 
    v_pattern = re.compile(r'OxCGRT_V[0-9][A-Z]*') 
    index_pattern = re.compile(r'Index*')

    policy_C_columns = list(filter(lambda x: (c_pattern.match(x) is not None), columns))
    policy_E_columns = list(filter(lambda x: (e_pattern.match(x) is not None), columns))
    policy_H_columns = list(filter(lambda x: (h_pattern.match(x) is not None), columns))
    policy_V_columns = list(filter(lambda x: (v_pattern.match(x) is not None), columns))
    policy_index_columns = list(filter(lambda x: (index_pattern.search(x) is not None), columns))

    policy_C_columns_, policy_C_id_to_flag = _process_policy_columns(policy_C_columns)
    policy_E_columns_, policy_E_id_to_flag = _process_policy_columns(policy_E_columns)
    policy_H_columns_, policy_H_id_to_flag = _process_policy_columns(policy_H_columns)
    policy_V_columns_, policy_V_id_to_flag = _process_policy_columns(policy_V_columns)

    # Get policy flag columns 
    policy_id_dict = policy_C_id_to_flag# columns_map['policy_C_columns'][1]
    policy_id_dict.update(policy_E_id_to_flag)#columns_map['policy_H_columns'][1])
    policy_id_dict.update(policy_H_id_to_flag)#columns_map['policy_E_columns'][1])
    policy_id_dict.update(policy_V_id_to_flag)#columns_map['policy_V_columns'][1])

    OxCGRT_policy_support_ids = ['E1', 'H7']
    OxCGRT_policy_regional_ids = list(filter(lambda x: ((x[0] not in OxCGRT_policy_support_ids) and (x[1])), policy_id_dict.items()))
    # OxCGRT_policy_regional_ids = list(map(lambda x: x[0], OxCGRT_policy_regional_ids))
    OxCGRT_policy_support_flag_columns = list(map(lambda x: f'OxCGRT_{x}_Flag', OxCGRT_policy_support_ids))
    OxCGRT_policy_regional_flag_columns = list(map(lambda x: f'OxCGRT_{x[0]}_Flag', OxCGRT_policy_regional_ids))

    return {
        'location': location_columns,
        'jurisdiction': policy_jurisdiction_columns,
        'date': date_columns,
        'outcome_columns': outcome_columns,
        'vax_status_columns': vax_status_columns,
        'policy_C_columns': (policy_C_columns_, policy_C_id_to_flag),
        'policy_E_columns': (policy_E_columns_, policy_E_id_to_flag),
        'policy_H_columns': (policy_H_columns_, policy_H_id_to_flag),
        'policy_V_columns': (policy_V_columns_, policy_V_id_to_flag),
        'policy_support_target_flags': OxCGRT_policy_support_flag_columns,
        'policy_regional_level_flags': OxCGRT_policy_regional_flag_columns,
        'policy_index_columns': policy_index_columns,        
    }

def get_mobility_columns(columns):
    pct_mobility_p = re.compile('_percent_change_from_baseline')
    pct_mobility_columns = list(filter(lambda x: (pct_mobility_p.search(x) is not None), columns))
    location_columns = list(filter(lambda x: x not in pct_mobility_columns+['Mobility_date'], columns))
    return {
        'location': location_columns,
        'date': ['Mobility_date'],
        'mobility': pct_mobility_columns,
    }

def get_hospital_columns(columns):
    per_100_000_p = re.compile('_per_100_000')
    per_100_000_columns = list(filter(lambda x: (per_100_000_p.search(x) is not None), columns))
    other_columns = list(filter(lambda x: x not in per_100_000_columns, columns))
    return {
        'per_100_000_columns': per_100_000_columns,
        'other': other_columns,
    }

def get_demographic_columns(columns):
    count_columns = ['Demographic_Population', 'Demographic_Median income (dollars)']
    pct_columns = list(filter(lambda x: x not in count_columns, columns))
    return {
        'count_columns': count_columns,
        'pct_columns': pct_columns,
    }

def get_misc_columns(columns):
    return {
        'weighted_population_density': ['Misc_Population Density  - weighted [per km2]'],
        'political': ['Misc_Democratic', 'Misc_Republican']
    }

def get_columns(merged_df):
    # Get columns map
    OxCGRT_p = re.compile('OxCGRT_*')
    Hospital_p = re.compile('Hospital_*')
    Mobility_p = re.compile('Mobility_*')
    Demographic_p = re.compile('Demographic_*')
    Misc_p = re.compile('Misc_*')

    # Select columns
    OxCGRT_columns = list(filter(lambda x: OxCGRT_p.match(x), merged_df.columns))
    Hospital_columns = list(filter(lambda x: Hospital_p.match(x), merged_df.columns))
    Mobility_columns = list(filter(lambda x: Mobility_p.match(x), merged_df.columns))
    Demographic_columns = list(filter(lambda x: Demographic_p.match(x), merged_df.columns))
    Misc_columns = list(filter(lambda x: Misc_p.match(x), merged_df.columns))

    # Get map of columns
    OxCGRT_columns_map = get_OxCGRT_columns(OxCGRT_columns)
    mobility_columns_map = get_mobility_columns(Mobility_columns)
    hospital_columns_map = get_hospital_columns(Hospital_columns)
    demographic_columns_map = get_demographic_columns(Demographic_columns)
    misc_columns_map = get_misc_columns(Misc_columns)

    ## Columns of interest:
    # OxCGRT: policy (OxCGRT_policy_columns), policy_support_target_flags, vax_status_columns, policy_index_columns
    # Hospital: per_100_000_columns
    # Mobility: mobility
    # Demographic: all 
    # Misc: weighted_population_density
    OxCGRT_policy_columns = OxCGRT_columns_map['policy_C_columns'][0] + \
                            OxCGRT_columns_map['policy_E_columns'][0] + \
                            OxCGRT_columns_map['policy_H_columns'][0] + \
                            OxCGRT_columns_map['policy_V_columns'][0]

    data_to_column_map = {
        'OxCGRT': OxCGRT_columns_map,
        'mobility': mobility_columns_map,
        'hospital': hospital_columns_map,
        'demographic': demographic_columns_map,
        'misc': misc_columns_map,
    }
    data_to_columns = {
        'OxCGRT': OxCGRT_columns,
        'mobility': Mobility_columns,
        'hospital': Hospital_columns,
        'demographic': Demographic_columns,
        'misc': Misc_columns,
    }

    return data_to_columns, data_to_column_map, OxCGRT_policy_columns

def encode_non_numerical_columns(df):
    # encode non numerical columns
    label_encoders = {}
    for col, dtype in df.dtypes.items():
        if dtype == 'object':
            le = LabelEncoder()
            le.fit(df[col])
            df.loc[:, col] = le.transform(df[col]) # modify the `data` df
            label_encoders[col] = le

    return df, label_encoders


def load_all_data():
    mobility_data_dict = load_mobility_data()
    hospital_coverage_df, hospital_admission_df = load_hospital_data('coverage')
    X_demographic_subnational, X_demographic_national = load_demographic_data()
    load_OxCGRT_data = get_OxCGRT_data_loader()
    national_OxCGRT_df, subnational_OxCGRT_df, subnational_OxCGRT_by_region = load_OxCGRT_data(C.COUNTRY_CODE)
    misc_df = load_misc_data()

    # Filter columns, Merge data 
    data_to_df = {
        'mobility_data_dict': mobility_data_dict,
        'hospital_coverage_df': hospital_coverage_df,
        'hospital_admission_df': hospital_admission_df,
        'X_demographic_subnational': X_demographic_subnational,
        'X_demographic_national': X_demographic_national,
        'national_OxCGRT_df': national_OxCGRT_df,
        'subnational_OxCGRT_df': subnational_OxCGRT_df,
        'misc_df': misc_df,
    }
    data_to_df_unfiltered = copy.deepcopy(data_to_df)
    data_to_df = filter_data(data_to_df)
    subnational_df_unfiltered, national_df_unfiltered = merge_data(data_to_df_unfiltered)
    subnational_df, national_df = merge_data(data_to_df)

    # Get columns
    ## Columns of interest:
    # OxCGRT: policy (OxCGRT_policy_columns), policy_support_target_flags, vax_status_columns, policy_index_columns
    # Hospital: per_100_000_columns
    # Mobility: mobility
    # Demographic: all 
    # Misc: weighted_population_density
    data_to_columns_unfiltered, data_to_column_map_unfiltered, OxCGRT_policy_columns_unfiltered = get_columns(subnational_df_unfiltered)
    data_to_columns, data_to_column_map, OxCGRT_policy_columns = get_columns(subnational_df)

    # Update dtypes
    subnational_df = subnational_df.astype(subnational_df.infer_objects().dtypes)
    national_df = national_df.astype(national_df.infer_objects().dtypes)

    # Filter columns according to config file
    # Note: updating this column map will ensure we select from this subset.
    data_to_column_map = utils.filter_data_to_column_map(data_to_column_map)

    return ((subnational_df, national_df, data_to_df,
             data_to_columns, data_to_column_map, OxCGRT_policy_columns),
            (subnational_df_unfiltered, national_df_unfiltered, data_to_df_unfiltered,
             data_to_columns_unfiltered, data_to_column_map_unfiltered, OxCGRT_policy_columns_unfiltered))



def encode_policy(df, data_to_column_map, policy_ids, update_df=False):
    # 1) Indicators for when change occurs, 2) indicators for policies in place (confounders)
    all_policy_dfs, all_policy_intervention_columns_map = [], {}
    for policy_id in policy_ids:
        print(f'----- Policy {policy_id} -----')
        policy_columns = data_to_column_map['OxCGRT'][f'policy_{policy_id}_columns'][0]
        policy_df = df[policy_columns].copy()
        
        # Encode columns (if needed)
        dtype_tuple = [(x,y) for x,y in df[policy_columns].dtypes.items()]
        columns_to_encode = list(filter(lambda x:x[1]=='object', dtype_tuple))
        columns_to_encode = [x[0] for x in columns_to_encode]
        int_columns = []
        for col in columns_to_encode:
            categories = policy_df[[col]].iloc[:, 0].value_counts().index
            p = re.compile('yrs')
            if all([p.search(x) is not None for x in categories]): # if working with "age" categories, order encoding from highest lower bound to lowest lower bound
                categories = [np.nan] + sorted(categories, reverse=True)
                categories_map = dict(zip(categories, range(len(categories))))
                encoded_temp = df[col].apply(lambda x: categories_map[x]).values[:, None].astype('int64')
                int_columns.append(col)
            else:
                enc = OrdinalEncoder()
                encoded_temp = enc.fit_transform(df[[col]]).astype('Int64')
                # encoded_temp = pd.DataFrame(enc.fit_transform(subnational_df[columns_to_encode]))
            
            policy_df.loc[:, col] = encoded_temp

        dtypes = policy_df.dtypes
        dtypes.loc[policy_df.max() > 10] = 'Float64' # Set continuous vars as float64
        dtypes.loc[int_columns] = 'Int64'
        policy_df = policy_df.astype(dtypes)

        # Remove columns that provide no information
        to_remove = set()
        for col in policy_columns:
            n_unique = len(policy_df[col].dropna().unique())
            if n_unique <= 1: to_remove.add(col)
        policy_columns = list(filter(lambda x: x not in to_remove, policy_columns))
        policy_df = policy_df[policy_columns]

        if update_df: df[policy_columns] = policy_df # update df
        all_policy_dfs.append(policy_df)
        all_policy_intervention_columns_map[f'policy_{policy_id}_columns_intervention'] = [f'{x}_intervention' for x in policy_columns]
    
    return all_policy_dfs, all_policy_intervention_columns_map, df

# Process policy columns 
def process_policy(df, data_to_column_map, policy_ids):
    # 1) Indicators for when change occurs, 2) indicators for policies in place (confounders)
    # all_policy_dfs, all_policy_intervention_columns_map = [], {}
    # for policy_id in policy_ids:
    #     print(f'----- Policy {policy_id} -----')
    #     policy_columns = data_to_column_map['OxCGRT'][f'policy_{policy_id}_columns'][0]
    #     policy_df = df[policy_columns]
        
    #     # Encode columns (if needed)
    #     dtype_tuple = [(x,y) for x,y in df[policy_columns].dtypes.items()]
    #     columns_to_encode = list(filter(lambda x:x[1]=='object', dtype_tuple))
    #     columns_to_encode = [x[0] for x in columns_to_encode]
    #     int_columns = []
    #     for col in columns_to_encode:
    #         categories = policy_df[[col]].iloc[:, 0].value_counts().index
    #         p = re.compile('yrs')
    #         if all([p.search(x) is not None for x in categories]): # if working with "age" categories, order encoding from highest lower bound to lowest lower bound
    #             categories = [np.nan] + sorted(categories, reverse=True)
    #             categories_map = dict(zip(categories, range(len(categories))))
    #             encoded_temp = df[col].apply(lambda x: categories_map[x]).values[:, None].astype('int64')
    #             int_columns.append(col)
    #         else:
    #             enc = OrdinalEncoder()
    #             encoded_temp = enc.fit_transform(df[[col]]).astype('Int64')
    #             # encoded_temp = pd.DataFrame(enc.fit_transform(subnational_df[columns_to_encode]))
            
    #         policy_df.loc[:, [col]] = encoded_temp

    #     dtypes = policy_df.dtypes
    #     dtypes.loc[policy_df.max() > 10] = 'Float64' # Set continuous vars as float64
    #     dtypes.loc[int_columns] = 'Int64'
    #     policy_df = policy_df.astype(dtypes)

    #     # Remove columns that provide no information
    #     to_remove = set()
    #     for col in policy_columns:
    #         n_unique = len(policy_df[col].dropna().unique())
    #         if n_unique <= 1: to_remove.add(col)
    #     policy_columns = list(filter(lambda x: x not in to_remove, policy_columns))
    #     policy_df = policy_df[policy_columns]

    #     all_policy_dfs.append(policy_df)
    #     all_policy_intervention_columns_map[f'policy_{policy_id}_columns_intervention'] = [f'{x}_intervention' for x in policy_columns]
    all_policy_dfs, all_policy_intervention_columns_map, _ = encode_policy(df, data_to_column_map, policy_ids, update_df=False)

    all_policy_df = pd.concat(all_policy_dfs, axis=1)
    all_policy_intervention_df = all_policy_df.diff(1).fillna(0).add_suffix('_intervention').astype('Int64') # intervention indicators 
    
    # Replace "change" in intervention with actual intervened value
    for col in all_policy_intervention_df.columns:
        # all_policy_intervention_df[col]
        # (all_policy_intervention_df[col]!=0).sum()
        policy_col_series = all_policy_intervention_df[col]
        mask = (policy_col_series != 0)*1 # 1 if change was not 0, 0 otherwise
        all_policy_intervention_df.loc[:, col] = (mask * all_policy_df[col.replace('_intervention', '')]) # mask original data

    dtypes = df.dtypes
    dtypes.loc[all_policy_df.columns] = all_policy_df.dtypes
    df.loc[:, all_policy_df.columns] = 0 # temporarily, while we cast dtype
    df2 = df.astype(dtypes)
    df2.loc[:, all_policy_df.columns] = all_policy_df
    df2.loc[:, all_policy_intervention_df.columns] = all_policy_intervention_df

    return df2, all_policy_intervention_columns_map #all_policy_intervention_df.columns

def add_temporal_features(df, date_column):
    first_date = df.iloc[0][date_column]
    df['Temporal_month'] = df.loc[:, date_column].apply(lambda x: x.month)
    df['Temporal_day_of_week'] = df.loc[:, date_column].apply(lambda x: x.day_of_week)
    df['Temporal_month_transformed'] = df.loc[:, 'Temporal_month'].apply(lambda x: MONTH_TO_COS[x])
    df.loc[:, f'{date_column}_index'] = df.loc[:, date_column].apply(lambda x: (x-first_date).days)
    return {'date_index': [f'{date_column}_index']}, {'temporal': ['Temporal_month', 'Temporal_month_transformed', 'Temporal_day_of_week']}

def process_wave(df, date_column):
    def _get_wave_fn(start, end):
        # start, end = covid_wave_map[wave]
        fn = lambda date: date>=pd.Timestamp(start) and date<pd.Timestamp(end)
        return fn

    covid_wave_map = {
        'Wave_1': _get_wave_fn('2020-03-01', '2020-06-01'),
        'Wave_2': _get_wave_fn('2020-06-01', '2020-09-15'),
        'Wave_3': _get_wave_fn('2020-09-15', '2021-03-15'),
        'Wave_4': _get_wave_fn('2021-03-15', '2021-07-01'),
        'Wave_5': _get_wave_fn('2021-07-01', '2022-01-01'),
        'Wave_6': _get_wave_fn('2022-01-01', '2022-02-15'),
        # 'Wave_6': _get_wave_fn('2022-01-01', '2022-03-15'),
    }

    for wave, wave_fn in covid_wave_map.items():
        df.loc[:, wave] = \
            df.loc[:, date_column].apply(wave_fn)*1
        
    wave_columns = list(covid_wave_map.keys())
    return df.loc[df.loc[:, wave_columns].sum(axis=1)!=0], wave_columns # select only data within waves

def process_r0_outcomes(df, case_rate_column, groupby_column, date_column, population_column):
    assert df.groupby(groupby_column, dropna=False).size().all(), "Not all same length. Please preprocess."
    # T = df.groupby(groupby_column, dropna=False).size().iloc[0]
    cases_df = df.loc[:, [case_rate_column, groupby_column]]
    cases_df.index = df.loc[:, date_column]
    cases_df.loc[:, case_rate_column] = [_round(x*y) for x,y in zip(cases_df.loc[:, case_rate_column], df.loc[:, population_column])]

    def _compute_r0(region_code, region_cases_df):
        series = region_cases_df.loc[:, case_rate_column]
        series.ffill(inplace=True)
        series.fillna(0, inplace=True)
        series.loc[series < 0] = np.nan
        series.ffill(inplace=True)

        result = covid19.r_covid(
            series, gt_distribution=covid19.generate_standard_si_distribution(), delay_distribution=covid19.generate_standard_infection_to_reporting_distribution(), \
            a_prior=3, b_prior=1, smoothing_window=21, r_window_size=3, r_interval_dates=None, n_samples=100, quantiles=(0.025, 0.5, 0.975), auto_cutoff=True
        )
        result.loc[:, groupby_column] = region_code
        return result
    
    # Compute R0 values per region
    # t1=time.time()
    parallel = Parallel(n_jobs=2, return_as="generator")
    output_generator = parallel(delayed(_compute_r0)(region_code, region_cases_df) for region_code, region_cases_df in cases_df.groupby(groupby_column, dropna=False))
    result1 = (list(output_generator))
    # print(f'Total time: {(time.time()-t1)/60} min.')

    # Combine results 
    result_df = pd.concat(result1, axis=0)
    return pd.merge(
        left=df,
        right=result_df.reset_index().add_prefix('R0_'),
        left_on=[date_column, groupby_column],
        right_on=['R0_index', f'R0_{groupby_column}'],
        how='left'
    )




def post_process_data(subnational_df, national_df, data_to_column_map, OxCGRT_policy_columns):
    outcome_columns = data_to_column_map['OxCGRT']['outcome_columns']
    population_column = data_to_column_map['demographic']['count_columns'][0]
    date_column = data_to_column_map['OxCGRT']['date'][0]
    groupby_column = data_to_column_map['OxCGRT']['location'][-1]
    policy_index_columns = data_to_column_map['OxCGRT']['policy_index_columns']
    policy_raw_columns = list(filter(lambda x: 'Flag' not in x, OxCGRT_policy_columns))
    policy_ids = ['C', 'H', 'E', 'V']
    # policy_raw_columns = np.concatenate([data_to_column_map['OxCGRT'][f'policy_{pid}_columns'][0] for pid in policy_ids]).tolist()

    # 1. Apply differencing to outcomes
    # Apply differencing, adjust outcomes (Y) by population 
    print(f'Apply differencing, adjust outcomes by population')
    d = 7
    _, diff_outcome_columns = \
        tsa_helper.difference_outcomes(subnational_df, outcome_columns, d, groupby_col=groupby_column, scale_col=population_column, pct_change=False, add_to_df=True)
    _, _ = \
        tsa_helper.difference_outcomes(national_df, outcome_columns, d, groupby_col=groupby_column, scale_col=population_column, pct_change=False, add_to_df=True)
    _, diff2_outcome_columns = \
        tsa_helper.difference_outcomes(subnational_df, diff_outcome_columns, d, groupby_col=groupby_column, scale_col=None, pct_change=False, add_to_df=True) 
    _, _ = \
        tsa_helper.difference_outcomes(national_df, diff_outcome_columns, d, groupby_col=groupby_column, scale_col=None, pct_change=False, add_to_df=True) 
    data_to_column_map['OxCGRT'] = data_to_column_map['OxCGRT'] | {'outcome_columns_diff': diff_outcome_columns, 'outcome_columns_diff2': diff2_outcome_columns}


    # 2. Extract interventions from policy columns
    # a. Individual Containment/Health/Economic/Vax policies  (discrete)
    # subnational_df, all_policy_intervention_columns_map = \
    #     data_helper.process_policy(subnational_df, data_to_column_map, policy_ids)
    # national_df, _ = \
    #     data_helper.process_policy(national_df, data_to_column_map, policy_ids)
    # data_to_column_map['OxCGRT'] = data_to_column_map['OxCGRT'] | all_policy_intervention_columns_map
    # Encode columns first
    _, _, subnational_df = encode_policy(subnational_df, data_to_column_map, policy_ids, update_df=True)
    _, _, national_df = encode_policy(national_df, data_to_column_map, policy_ids, update_df=True)

    d = 7
    # outcome_dtypes = [subnational_df.dtypes.loc[col] for col in policy_raw_columns] 
    _, diff_policy_raw_columns = \
        tsa_helper.difference_outcomes(subnational_df, policy_raw_columns, d, groupby_col=groupby_column, scale_col=None, pct_change=False, add_to_df=True)
    _, _ = \
        tsa_helper.difference_outcomes(national_df, policy_raw_columns, d, groupby_col=groupby_column, scale_col=None, pct_change=False, add_to_df=True)

    # preserve dtype (Int64)
    subnational_df = _set_column_astype(subnational_df, dict(zip(diff_policy_raw_columns, ['Int64']*len(policy_raw_columns))))
    national_df = _set_column_astype(national_df, dict(zip(diff_policy_raw_columns, ['Int64']*len(policy_raw_columns))))

    # rename by adding "_intervention" suffix
    policy_raw_to_intervention_name_map = dict(zip(diff_policy_raw_columns, [f'{col}_intervention' for col in diff_policy_raw_columns]))
    subnational_df.rename(columns=policy_raw_to_intervention_name_map, inplace=True)
    national_df.rename(columns=policy_raw_to_intervention_name_map, inplace=True)

    data_to_column_map['OxCGRT'] = data_to_column_map['OxCGRT'] | {'policy_raw_columns_intervention': policy_raw_to_intervention_name_map}

    # b. Policy Indices (stringency, response, containment+health, economic)
    d = 7
    _, diff_policy_index_columns = \
        tsa_helper.difference_outcomes(subnational_df, policy_index_columns, d, groupby_col=groupby_column, scale_col=None, pct_change=False, add_to_df=True)
    _, _ = \
        tsa_helper.difference_outcomes(national_df, policy_index_columns, d, groupby_col=groupby_column, scale_col=None, pct_change=False, add_to_df=True)

    # rename by adding "_intervention" suffix
    policy_index_to_intervention_name_map = dict(zip(diff_policy_index_columns, [f'{col}_intervention' for col in diff_policy_index_columns]))
    subnational_df.rename(columns=policy_index_to_intervention_name_map, inplace=True)
    national_df.rename(columns=policy_index_to_intervention_name_map, inplace=True)

    data_to_column_map['OxCGRT'] = data_to_column_map['OxCGRT'] | {'policy_index_columns_intervention': policy_index_to_intervention_name_map}


    # 3. Add wave information and select data within identified waves
    subnational_df, wave_columns = process_wave(subnational_df, date_column)
    national_df, _ = process_wave(national_df, date_column)
    data_to_column_map['Wave'] = {'wave': wave_columns}
    """
    # 4. Compute R0 outcomes
    case_rate_column = data_to_column_map['OxCGRT']['outcome_columns_diff'][0]
    r0_outcome_columns = ['R0_R_mean', 'R0_R_var']

    subnational_df = data_helper.process_r0_outcomes(subnational_df, case_rate_column, groupby_column, date_column, population_column)
    national_df = data_helper.process_r0_outcomes(national_df, case_rate_column, groupby_column, date_column, population_column)
    """
    case_rate_column = data_to_column_map['OxCGRT']['outcome_columns_diff'][0]
    r0_outcome_columns=[]


    # 5. Add true outcome columns (0 days ahead, 7 days ahead, 14 days ahead)
    outcome_ts_ahead = [0, 7, 14]
    all_outcome_columns = diff_outcome_columns + diff2_outcome_columns + r0_outcome_columns + data_to_column_map['mobility']['mobility']

    outcome_ts_ahead_columns_map = \
    tsa_helper.add_forecast_outcomes(subnational_df, groupby_column, all_outcome_columns, outcome_ts_ahead)
    _ = tsa_helper.add_forecast_outcomes(national_df, groupby_column, all_outcome_columns, outcome_ts_ahead)
    data_to_column_map['OxCGRT'] = data_to_column_map['OxCGRT'] | {'outcome_columns_forecast': outcome_ts_ahead_columns_map}


    # 6. Featurize temporal information: 0 1 2..., month, day of week
    date_index_column_map, temporal_columns_map = add_temporal_features(subnational_df, date_column)
    _, _ = add_temporal_features(national_df, date_column)

    data_to_column_map['OxCGRT'] = data_to_column_map['OxCGRT'] | date_index_column_map
    data_to_column_map['Temporal'] = temporal_columns_map

    # 7. Filter to get weekly data only (day 0 for least variance)
    subnational_df = subnational_df.loc[subnational_df['Temporal_day_of_week'] == 0]
    national_df = national_df.loc[national_df['Temporal_day_of_week'] == 0]

    return subnational_df, national_df, data_to_column_map, outcome_ts_ahead_columns_map,\
           diff_policy_index_columns, diff_outcome_columns, date_column, outcome_columns



def process_columns(subnational_df, data_to_column_map, outcome_ts_ahead_columns_map, diff_outcome_columns, OxCGRT_policy_columns):
    # Consolidate columns
    # OxCGRT_policy_intervention_columns = np.concatenate([list(x) for x in all_policy_intervention_columns_map.values()]).tolist()
    OxCGRT_policy_raw_intervention_columns = list(data_to_column_map['OxCGRT']['policy_raw_columns_intervention'].values())
    OxCGRT_policy_support_flag_columns = list(filter(lambda col: col in config.flags_to_include, data_to_column_map['OxCGRT']['policy_support_target_flags']))
    OxCGRT_policy_index_intervention_columns = list(data_to_column_map['OxCGRT']['policy_index_columns_intervention'].values())

    # Filter: remove Flag columns, separate by static v dynamic
    all_data_columns_map = {
        'OxCGRT_policy_columns': list(filter(lambda x: 'Flag' not in x, OxCGRT_policy_columns)), # H7, V4 only (discrete)
        'OxCGRT_policy_index_columns': data_to_column_map['OxCGRT']['policy_index_columns'], # continuous
        'OxCGRT_policy_intervention_columns_discrete': OxCGRT_policy_raw_intervention_columns,
        'OxCGRT_policy_intervention_columns_continuous': OxCGRT_policy_index_intervention_columns,
        # 'OxCGRT_policy_support_target_flags': OxCGRT_policy_support_flag_columns,
        'OxCGRT_outcome_columns': diff_outcome_columns,
        'OxCGRT_outcome_ts_ahead_columns': np.concatenate(list(outcome_ts_ahead_columns_map.values())).tolist(),
        'OxCGRT_vax_status_columns': data_to_column_map['OxCGRT']['vax_status_columns'], # % vaccinated
        'Hospital_static_per_100_000_columns': list(filter(lambda col: 'Weekly' not in col, data_to_column_map['hospital']['per_100_000_columns'])),
        'Hospital_dynamic_per_100_000_columns': list(filter(lambda col: 'Weekly' in col, data_to_column_map['hospital']['per_100_000_columns'])),
        'Mobility_mobility': data_to_column_map['mobility']['mobility'],
        'Demographic_pct_columns': data_to_column_map['demographic']['pct_columns'],
        'Demographic_Median_income': ['Demographic_Median income (dollars)'],
        'Misc_weighted_population_density': data_to_column_map['misc']['weighted_population_density'],
        'Misc_political': data_to_column_map['misc']['political'],
        'Temporal_temporal': data_to_column_map['Temporal']['temporal'],
        'Wave_wave': data_to_column_map['Wave']['wave'],
    }

    # Remove variables as needed
    data_to_remove_columns_map = {
        'Demographic_pct_columns': ['Demographic_Civilian noninstitutionalized population'],
        'Temporal_temporal': ['Temporal_month']
    }
    [[all_data_columns_map[data_type].remove(col) for col in columns] for data_type,columns in data_to_remove_columns_map.items()]
    # del all_data_columns_map['OxCGRT_policy_index_columns']
    all_data_columns = np.concatenate([x for x in all_data_columns_map.values()])

    # Variables for plotting
    N_columns = len(all_data_columns)
    get_dims = lambda x: (math.ceil(N_columns/x), x)

    # Get map from Region Code to Region Name (and vice versa)
    rcode_to_rname = dict(zip(subnational_df.OxCGRT_RegionCode, subnational_df.OxCGRT_RegionName))
    rname_to_rcode = dict(zip(subnational_df.OxCGRT_RegionName, subnational_df.OxCGRT_RegionCode))

    return all_data_columns_map, rcode_to_rname, rname_to_rcode, N_columns, get_dims



# Define functions
def add_lagged_features(df, columns_to_lag, lag):
    lagged_columns = np.concatenate(list(np.array([f'{x}_{l}' for l in range(1,lag+1)]) for x in columns_to_lag))

    # Add lagged data
    shifted_data = pd.concat(
        [df[columns_to_lag + ['OxCGRT_RegionCode', 'OxCGRT_Date']] \
         .groupby('OxCGRT_RegionCode').shift(l).add_suffix(f'_{l}') \
            for l in range(lag+1)],
        axis=1
    ) # _1 and _2 columns are lagged
    shifted_data.loc[:, 'OxCGRT_RegionCode'] = df[['OxCGRT_RegionCode']]
    temp = pd.merge(
        left=df,
        right=shifted_data[lagged_columns.tolist() + ['OxCGRT_Date_0', 'OxCGRT_RegionCode']],
        left_on=['OxCGRT_Date', 'OxCGRT_RegionCode'],
        right_on=['OxCGRT_Date_0', 'OxCGRT_RegionCode']
    )

    return temp, lagged_columns 

def make_data_transformer(add_features=False):
    # Define transformers for continuous (float), continuous (int), and categorical variables (imputed)
    pipe1 = [
        PowerTransformer(standardize=True), # log normal transform (automatically turns into unit gaussian)
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    ]
    if not add_features: pipe1 = pipe1[:1] # remove polynomial features
    continuous_transformer = make_pipeline(*pipe1)    # Impute missing values, then transform into unit Gaussian

    pipe2 = [
        MinMaxScaler(), # min-max 0-1 transform 
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    ]
    if not add_features: pipe2 = pipe2[:1] # remove polynomial features
    continuous_transformer2 = make_pipeline(*pipe2)    # Impute missing values, then apply min-max scaling

    pipe3 = [
        # OrdinalEncoder(), # Encode categorical variables
        OneHotEncoder(drop='if_binary'), # Encode categorical variables
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    ]
    if not add_features: pipe3 = pipe3[:1] # remove polynomial features
    categorical_transformer = make_pipeline(*pipe3)

    # continuous_transformer = FeatureUnion(
    #     transformer_list=[
    #         ('lognormal', lognormal_transformer),   # impute, transform 
    #         ('indicators', MissingIndicator()),     # add missingness indicator
    #     ])
    # categorical_transformer = FeatureUnion(
    #     transformer_list=[
    #         ('ordinal_encoder', ordinal_encoder),   # impute, encode 
    #         ('indicators', MissingIndicator()),     # add missingness indicator
    #     ])

    # Assemble pipeline
    pipe4 = [
        (continuous_transformer,
            make_column_selector(dtype_include=[np.float64, pd.Float64Dtype()])), #, np.int64])),
        (continuous_transformer2,
            make_column_selector(dtype_include=[np.int64, pd.Int64Dtype()])), #, np.int64])),
        (categorical_transformer,
            make_column_selector(dtype_include=[object]))        
    ]

    return make_column_transformer(*pipe4)

def process_column(col):
    if 'pipeline-3' in col: 
        col2 = col.replace('pipeline-3__', '')
        return '_'.join(col2.split('_')[:-1])
    return col.replace('pipeline-1__', '').replace('pipeline-2__', '')
