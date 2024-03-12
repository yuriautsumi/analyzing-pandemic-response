""" Data helper for OxGRT data. """

import os
import glob

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from textwrap import wrap
from datetime import datetime
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

def get_data_loader():
    # %matplotlib inline # returns plots without having to write plt.show()
    # compact_national = pd.read_csv('data/covid-policy-dataset-main/data/OxCGRT_compact_national_v1.csv')
    compact_subnational = pd.read_csv('data/covid-policy-dataset-main/data/OxCGRT_compact_subnational_v1.csv')

    def data_loader(country):
        country_subnational_df = compact_subnational.loc[compact_subnational.CountryCode == country]

        # Convert to datetime
        country_subnational_df.loc[:, 'Date'] = country_subnational_df.Date.apply(lambda x: datetime.strptime(f'{x}', '%Y%m%d'))

        # Make float 
        country_subnational_df.loc[:, 'PopulationVaccinated'] = country_subnational_df['PopulationVaccinated'].apply(lambda x: float(x))

        # Make np.nan
        country_subnational_df.loc[country_subnational_df.RegionCode.isna(), 'RegionCode'] = np.nan

        # Group by region
        country_subnational_by_region = country_subnational_df.groupby('RegionCode')
        assert country_subnational_by_region.size().all() # all same length, 1096
        country_national_df = country_subnational_df.loc[country_subnational_df.RegionCode.isna()]
        country_subnational_df = country_subnational_df.loc[~country_subnational_df.index.isin(country_national_df.index)]
        
        return country_national_df, country_subnational_df, country_subnational_by_region
    
    return data_loader

def get_columns(df):
    def _process_policy_columns(policy_columns):
        policy_identifiers = [x.split('_')[0] for x in policy_columns]
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
    location_columns = ['CountryName', 'CountryCode', 'RegionName', 'RegionCode']
    if 'CityName' in df.columns: location_columns += ['CityName', 'CityCode']
    policy_jurisdiction_columns = ['Jurisdiction']
    date_columns = ['Date']
    outcome_columns = ['ConfirmedCases', 'ConfirmedDeaths']
    vax_status_columns = ['MajorityVaccinated', 'PopulationVaccinated']

    # Extract policy columns by category (same for both tables)
    c_pattern = re.compile(r'C[0-9][A-Z]*')  # Compile the regular expression pattern
    e_pattern = re.compile(r'E[0-9][A-Z]*') 
    h_pattern = re.compile(r'H[0-9][A-Z]*') 
    v_pattern = re.compile(r'V[0-9][A-Z]*') 
    index_pattern = re.compile(r'Index')

    policy_C_columns = list(filter(lambda x: (c_pattern.match(x) is not None), df.columns))
    policy_E_columns = list(filter(lambda x: (e_pattern.match(x) is not None), df.columns))
    policy_H_columns = list(filter(lambda x: (h_pattern.match(x) is not None), df.columns))
    policy_V_columns = list(filter(lambda x: (v_pattern.match(x) is not None), df.columns))
    policy_index_columns = list(filter(lambda x: (index_pattern.search(x) is not None), df.columns))

    policy_C_columns_, policy_C_id_to_flag = _process_policy_columns(policy_C_columns)
    policy_E_columns_, policy_E_id_to_flag = _process_policy_columns(policy_E_columns)
    policy_H_columns_, policy_H_id_to_flag = _process_policy_columns(policy_H_columns)
    policy_V_columns_, policy_V_id_to_flag = _process_policy_columns(policy_V_columns)

    output = {
        'location': location_columns,
        'jurisdiction': policy_jurisdiction_columns,
        'date': date_columns,
        'outcome_columns': outcome_columns,
        'vax_status_columns': vax_status_columns,
        'policy_C_columns': (policy_C_columns_, policy_C_id_to_flag),
        'policy_E_columns': (policy_E_columns_, policy_E_id_to_flag),
        'policy_H_columns': (policy_H_columns_, policy_H_id_to_flag),
        'policy_V_columns': (policy_V_columns_, policy_V_id_to_flag),
        'policy_index_columns': policy_index_columns,        
    }
    return output

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

