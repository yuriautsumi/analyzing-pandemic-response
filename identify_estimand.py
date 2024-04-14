import os
import pickle
import argparse
# import pandas as pd
import constants as C


# Get estimand type
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--estimand', type=str, required=True)  # ate, nie, nde
ap.add_argument('-a', '--treatment', type=str, required=True) # ch, ev
ap.add_argument('-d', '--dtype', type=str, required=True)     # disc, cont
ap.add_argument('-t', '--test', type=str, required=True)      # test, not

arg = ap.parse_args()
estimand_type = arg.estimand
treatment_type = arg.treatment
dtype = arg.dtype
is_test = (arg.test == 'test')
# estimand_type = 'ate'
# treatment_type = 'ch' # 'ev'
# is_test = True
print(f'Identifying estimand: {estimand_type} for intervention on A{dtype}_{treatment_type}')
if is_test: print('### This is a test ###')

# Create directories
national_dir = f'national/{C.COUNTRY_CODE}'
subnational_dir = f'subnational/{C.COUNTRY_CODE}'
national_artifacts_dir = os.path.join(national_dir, 'artifacts')
subnational_artifacts_dir = os.path.join(subnational_dir, 'artifacts')

os.makedirs(national_artifacts_dir, exist_ok=True)
os.makedirs(subnational_artifacts_dir, exist_ok=True)

# Load objects
with open(os.path.join(national_artifacts_dir, 'subnational_processed_final_var_only_df.pkl'), 'rb') as f:
    subnational_processed_final_var_only_df = pickle.load(f)

treatment_prefix = f'A{dtype}_{treatment_type}'
model_file = os.path.join(national_artifacts_dir, f'causal_model_{treatment_prefix}.pkl')
if is_test: model_file = os.path.join(national_artifacts_dir, f'causal_model_{treatment_prefix}_simple.pkl')
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Identify estimand and pickle
if estimand_type == 'ate':
    # Average treatment effect (ate)
    identified_estimand_ate = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand_ate)

    pickle_file = os.path.join(national_artifacts_dir, f'identified_estimand_ate_{treatment_prefix}.pkl') 
    if is_test: pickle_file = pickle_file.replace('.pkl', '_simple.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(identified_estimand_ate, f)
elif estimand_type == 'nde':
    # Natural direct effect (nde)
    identified_estimand_nde = model.identify_effect(estimand_type="nonparametric-nde", 
                                                    proceed_when_unidentifiable=True)
    print(identified_estimand_nde)

    pickle_file = os.path.join(national_artifacts_dir, f'identified_estimand_nde_{treatment_prefix}.pkl')
    if is_test: pickle_file = pickle_file.replace('.pkl', '_simple.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(identified_estimand_nde, f)
elif estimand_type == 'nie':
    # Natural indirect effect (nie)
    identified_estimand_nie = model.identify_effect(estimand_type="nonparametric-nie", 
                                                    proceed_when_unidentifiable=True)
    print(identified_estimand_nie)

    pickle_file = os.path.join(national_artifacts_dir, f'identified_estimand_nie_{treatment_prefix}.pkl')
    if is_test: pickle_file = pickle_file.replace('.pkl', '_simple.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(identified_estimand_nie, f)
