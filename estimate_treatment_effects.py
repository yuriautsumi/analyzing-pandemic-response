import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import utils
import causal_analysis_helper

import config
import config_treatment_effect as config_te

import constants as C

# Get arguments
ap = argparse.ArgumentParser()
# ap.add_argument('-e', '--estimand', type=str, required=True)  # ate, nie, nde
ap.add_argument('-a', '--treatment', type=str, required=True) # ch, ev
ap.add_argument('-d', '--dtype', type=str, required=True)     # disc, cont
ap.add_argument('-y', '--outcome', type=str, required=True)     # {cases,deaths}-{daily,rate}, mobility-{grocery,retail,transit,work}
ap.add_argument('-t', '--target', type=str, required=True)     # ate, att, atc
ap.add_argument('-w', '--waves', type=str, required=True)     # by-wave, not
ap.add_argument('-v', '--verbose', type=str, required=True)    # verbose, not
# ap.add_argument('-t', '--test', type=str, required=True)      # test, not

arg = ap.parse_args()
treat_type = arg.treatment
dtype = arg.dtype
outcome_id = arg.outcome
target_units = arg.target
waves = (arg.waves == 'by-wave')
verbose = (arg.verbose == 'verbose')

# test_significance, confidence_intervals = False, False
test_significance, confidence_intervals = True, True

run_id = f'{treat_type}_{dtype}_{outcome_id}_{target_units}'
if waves: run_id += '_byWave'

############################

# Initialize directory
national_dir = f'national/{C.COUNTRY_CODE}'
national_artifacts_dir = os.path.join(national_dir, 'artifacts')
os.makedirs(national_artifacts_dir, exist_ok=True)

############################
# Load data, artifacts, models, estimands

subnational_processed_final_var_only_df, node_artifacts, treatment_type_to_artifacts = utils.load_artifacts(national_artifacts_dir)
(column_names_to_node_name, column_to_node_map, node_to_column_map,
 outcome_ts_ahead_columns_map, treatment_type_to_artifacts, node_prefix_to_count) = node_artifacts

############################
# Initialize variables

df = subnational_processed_final_var_only_df

additional_estimator_params = {'test_significance': test_significance, 'confidence_intervals': confidence_intervals, 'target_units': target_units}
wave_indices = range(2,7) if waves else range(0,1)
wave_columns = [f'Wave_{i}' for i in range(2,7)] if waves else [None]
outcome_category = config_te.outcome_id_to_category[outcome_id]
outcome_columns = outcome_ts_ahead_columns_map[outcome_category]
# outcome_columns = utils.get_columns('outcomes', all_data_columns_map, 1)[:-9][::-1] # TODO: fix, run by outcome type

# Get model and estimands for this treatment
artifact_dict = treatment_type_to_artifacts[treat_type][dtype]
model, estimand_map = artifact_dict['model'], artifact_dict['estimands']

# Get treatment values for this treatment
_, _, A0, A = causal_analysis_helper._get_node_values(f'treatments_{dtype}', column_to_node_map, variable_i=None, suffix=treat_type)
num_A = node_prefix_to_count[A]
node_As, treatment_columns, treatment_indices = causal_analysis_helper._get_node_column_names(A, num_A, node_to_column_map)

print(f'Estimate {target_units} of {node_As}, {treatment_columns} on {outcome_columns}.')

############################
# Estimate treatment effects

results = [] # [tuple indices, effect1, effect2, ...]
for wave_ix, wave_i in zip(wave_indices, wave_columns):
    Lwave_i = column_to_node_map.get(wave_i)
    wave_data = df if not waves else df.loc[df[Lwave_i]==1]
    
    # For each outcome (0 days, 7 days, 14 days),
    # compute estimate for specified treatment (CH/EV, discrete/continuous)
    for outcome_i in tqdm(outcome_columns):
        Yi, outcome_ix, Y0, _ = causal_analysis_helper._get_node_values('outcomes', column_to_node_map, variable_i=outcome_i, suffix=None)
        ts_ahead = int(outcome_i.strip('_ts_ahead').split('_')[-1])

        # Swap with outcome of interest. Store updated model to dict. 
        treatment_type_to_artifacts = causal_analysis_helper._update_values(
            Y0, Yi, artifacts=treatment_type_to_artifacts, data=wave_data, verbose=verbose)

        # For each treatment
        for treatment_ix, treatment_i, Ai in (zip(treatment_indices, treatment_columns[1:], node_As[1:])):
            # Swap with treatment of interest. 
            causal_analysis_helper._update_values(A0, Ai, model=model, verbose=verbose)

            # Get scaled treatment values 
            s0,s1,u0,u1 = config.treatment_dtype_to_control_treated_values[dtype]

            # Get indices 
            partial_indices = (wave_ix, Lwave_i, wave_i) +  (outcome_ix, Yi, outcome_i, ts_ahead) + (model,) + (treat_type, dtype, treatment_ix, Ai, treatment_i)

            # Estimate ATE/NDE/NIE
            for estimand_name,estimator_name,estimator_params in tqdm(zip(
                config_te.estimand_names, config_te.estimator_names, config_te.estimand_estimator_params
            )):
                if estimand_name not in estimand_map: continue
                if ('propensity' in estimator_params['method_name']) and (dtype == 'cont'): continue # skip propensity methods for continuous treatment

                # Set estimator parameters 
                identified_estimand = estimand_map[estimand_name]
                estimator_params['control_value'], estimator_params['treatment_value'] = s0, s1
                estimator_params |= additional_estimator_params

                if estimand_name in {'identified_estimand_nde', 'identified_estimand_nie'}:
                    partial_results = causal_analysis_helper._estimate_mediated_effect(model, identified_estimand, estimand_name, 
                                                                                       estimator_name, estimator_params, treat_type,
                                                                                       node_prefix_to_count, node_to_column_map, verbose=verbose)
                else:
                    partial_results = causal_analysis_helper._estimate_ate(model, identified_estimand, estimand_name, 
                                                                           estimator_name, estimator_params, verbose=verbose)

                results += [[ partial_indices+a, b ] for a,b in partial_results]

            # Reset swapped treatment.
            causal_analysis_helper._update_values(A0, Ai, model=model, verbose=verbose)
            treatment_type_to_artifacts[treat_type][dtype]['model'] = model
        
        # Reset swapped outcome. Update in model dict. 
        treatment_type_to_artifacts = causal_analysis_helper._update_values(
            Y0, Yi, artifacts=treatment_type_to_artifacts, data=wave_data, verbose=verbose)
        
############################
# Save results
results_file = os.path.join(national_artifacts_dir, f'treatment_effect_results_{run_id}.pkl')
utils._save_results(results, path=results_file)
