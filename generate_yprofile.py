import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from ydata_profiling import ProfileReport

from data_helper import get_data_loader, get_columns, encode_non_numerical_columns

COUNTRY_CODE = 'USA'

## Part 1. Load data
# Load data
OxCGRT_data_loader = get_data_loader()
national_df, subnational_df, subnational_by_region = OxCGRT_data_loader(COUNTRY_CODE)

# Get map of columns
columns_map = get_columns(subnational_df)
subnational_data_columns = np.setdiff1d(subnational_df.columns, columns_map['location'] + columns_map['date']) # Used to select main columns with data

# Encode non numerical columns 
temp, label_encoders = encode_non_numerical_columns(subnational_df[subnational_data_columns])
subnational_df.loc[:, subnational_data_columns] = temp


## Part 2. Generate yprofile reports
# Create directories
national_dir = f'national/{COUNTRY_CODE}'
subnational_dir = f'subnational/{COUNTRY_CODE}'
national_yprofile_dir = os.path.join(national_dir, 'yprofile')
subnational_yprofile_dir = os.path.join(subnational_dir, 'yprofile')

os.makedirs(national_yprofile_dir, exist_ok=True)
os.makedirs(subnational_yprofile_dir, exist_ok=True)

# Setting what variables are time series
type_schema = { col: 'timeseries' for col in subnational_data_columns }

# Generate yprofile for national data
print('Generating yprofile for national data...')
national_ffill = national_df.ffill()#.set_index(columns_map['date'])

#Enable tsmode to True to automatically identify time-series variables
#Provide the column name that provides the chronological order of your time-series
profile = ProfileReport(national_df, tsmode=True, type_schema=None, sortby=columns_map['date'], title=f"{COUNTRY_CODE} Time-Series EDA (RAW)")
profile.to_file(os.path.join(national_yprofile_dir, f"{COUNTRY_CODE}_Report_RAW.html"))

profile2 = ProfileReport(national_ffill, tsmode=True, type_schema=None, sortby=columns_map['date'], title=f"{COUNTRY_CODE} Time-Series EDA (FFILL)")
profile2.to_file(os.path.join(national_yprofile_dir, f"{COUNTRY_CODE}_Report_FFILL.html"))

# Generate yprofile for subnational data
print('Generating yprofile for subnational data...')
def generate_yprofile(region_code):
    # Filtering time-series to profile a single site
    site = subnational_by_region.get_group(region_code)#.set_index(columns_map['date'])
    site_ffill = site.ffill()

    #Enable tsmode to True to automatically identify time-series variables
    #Provide the column name that provides the chronological order of your time-series
    profile = ProfileReport(site, tsmode=True, type_schema=None, sortby=columns_map['date'], title=f"{region_code} Time-Series EDA (RAW)")
    profile.to_file(os.path.join(subnational_yprofile_dir, f"{region_code}_Report_RAW.html"))

    profile2 = ProfileReport(site_ffill, tsmode=True, type_schema=None, sortby=columns_map['date'], title=f"{region_code} Time-Series EDA (FFILL)")
    profile2.to_file(os.path.join(subnational_yprofile_dir, f"{region_code}_Report_FFILL.html"))

for region_code in tqdm(subnational_by_region.groups.keys()):
    generate_yprofile(region_code)

# Parallel(n_jobs=3)(delayed(generate_yprofile)(region_code) for region_code in tqdm(subnational_by_region.groups.keys()))
