import os
import glob
import copy
import dowhy
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import config_treatment_effect as config_te

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

def _node_has_dummy(node):
    node_base = node.split('_')[0] # remove _ch, _ev
    node_category = config.node_prefix_to_category[node_base]
    return config.node_category_to_dummy_boolean[node_category]

def _get_all_edges_from_x_to_z(x, z, num_x, num_z):#, with_dummy=True, remove_1ix=False):
    def _node(n, ix, max_ix):
        # if remove_1ix:
        #     # if max_ix == 1 and n not in dummy_prefixes: return n
        #     if max_ix == 1 and not _node_has_dummy(n): return n
        #     elif max_ix == 1 and not with_dummy: return n
        return f'{n}{ix}'
    
    x_range = range(0, num_x+1) if _node_has_dummy(x) else range(1, num_x+1)
    z_range = range(0, num_z+1) if _node_has_dummy(z) else range(1, num_z+1)
    # if with_dummy:
    #     # x_range = range(0, num_x+1) if x in dummy_prefixes else range(1, num_x+1)
    #     # z_range = range(0, num_z+1) if z in dummy_prefixes else range(1, num_z+1)
    #     x_range = range(0, num_x+1) if _node_has_dummy(x) else range(1, num_x+1)
    #     z_range = range(0, num_z+1) if _node_has_dummy(z) else range(1, num_z+1)
    # else:
    #     x_range = range(1, num_x+1)
    #     z_range = range(1, num_z+1)
    
    # edges = ';\n'.join([';'.join([f'{x}{i}->{z}{j}' for j in z_range]) for i in x_range]) + ';'
    edges = ';\n'.join([';'.join([f'{_node(x,i,num_x)}->{_node(z,j,num_z)}' for j in z_range]) for i in x_range]) + ';'
    return edges

def get_graph(all_pairs, node_to_count_map):#, with_dummy=True, remove_1ix=False): 
    graph = ''
    for (node_x, node_z) in all_pairs:
        count_x, count_z = node_to_count_map[node_x], node_to_count_map[node_z]
        # if count_x==0 or count_z==0: continue
        graph += _get_all_edges_from_x_to_z(node_x, node_z, count_x, count_z)#, with_dummy, remove_1ix)
    return graph

# Get causal graph, given treatment category
def get_causal_model(df, node_prefix_to_count, treat_type, dtype, simplified=False):
    # Filter prefixes and node counts 
    node_prefixes, node_counts = list(zip(*node_prefix_to_count.items()))
    node_prefixes_to_remove = config.treatment_type_to_node_prefixes_to_remove[treat_type][dtype]
    node_prefixes, node_counts = list(zip(*filter(lambda x: x[0] not in node_prefixes_to_remove, zip(node_prefixes, node_counts))))

    # Filter edges
    filtered_edges = list(filter(lambda e: (e[0] in node_prefixes) and (e[1] in node_prefixes), config.edges))

    # Set treatment (A) and outcome (Y) to dummy only
    node_to_count = dict(zip(node_prefixes, node_counts))
    treatment_prefix = config.node_category_to_prefix[f'treatments_{dtype}'] + f'_{treat_type}'
    mediator_prefix1, mediator_prefix2 = config.node_category_to_prefix['mediators_mobility'], config.node_category_to_prefix['mediators_vax']
    outcome_prefix = config.node_category_to_prefix['outcomes']
    if simplified: 
        for k,v in node_to_count.items(): node_to_count[k]=1
    node_to_count[treatment_prefix] = 0 # Set to 0 (dummy only)
    if mediator_prefix1 in node_to_count: node_to_count[mediator_prefix1] = 0 # Set to 0 (dummy only)
    if mediator_prefix2 in node_to_count: node_to_count[mediator_prefix2] = 0 # Set to 0 (dummy only)
    node_to_count[outcome_prefix] = 0 # Set to 0 (dummy only)

    # Build graph
    # args = dict(with_dummy=True, remove_1ix=False) if simplified else dict(with_dummy=True, remove_1ix=False)
    graph = get_graph(filtered_edges, node_to_count)#, **args)
    causal_graph = "digraph {" + graph + "}"

    treatment, outcome = f'{treatment_prefix}0', f'{outcome_prefix}0'
    # treatment, outcome = treatment[:-1], outcome[:-1]
    model = dowhy.CausalModel(data=df,
                            graph=causal_graph.replace("\n", " "),
                            treatment=treatment, #f'{treatment_prefix}0',
                            outcome=outcome) #f'{outcome_prefix}0')
    return model 


def _get_updated_model_data(model, column, column0):
    # backupcopy = model._data.copy()
    tempcopy = copy.deepcopy(model._data)
    # X0, Xi = tempcopy[column0].values, tempcopy[column].values
    # tempcopy.loc[:, column0] = Xi # set Xi values to the intervention variable column
    # tempcopy.loc[:, column]  = X0 # set null values for ith intervention column
    # tempcopy.loc[:, column0], tempcopy.loc[:, column] = tempcopy[column].values, tempcopy[column0].values
    # model._data = tempcopy
    return tempcopy.rename(columns={column0: column, column: column0})

# def _get_reset_model_data(model, column, column0):
#     # end: swap back
#     tempcopy = model._data.copy()
#     # Xi, X0 = tempcopy[column0].values, tempcopy[column].values # get swapped data
#     # tempcopy.loc[:, column0] = X0 
#     # tempcopy.loc[:, column]  = Xi 
#     tempcopy.loc[:, column0], tempcopy.loc[:, column] = tempcopy[column].values, tempcopy[column0].values
#     # model._data = tempcopy
#     return tempcopy
    


def test_conditional_independence(estimands_map, model):
    identified_estimand_ate = estimands_map['identified_estimand_ate']
    identified_estimand_nde = estimands_map['identified_estimand_nde']

    Y = identified_estimand_ate.outcome_variable[0]
    A = identified_estimand_ate.treatment_variable[0]
    Z = identified_estimand_ate.get_backdoor_variables()
    independence_constraints_ate=[(A,Y, tuple(Z))]

    Y = identified_estimand_nde.outcome_variable[0]
    A = identified_estimand_nde.treatment_variable[0]
    M = identified_estimand_nde.get_mediator_variables()[0]
    Z1 = identified_estimand_nde.mediation_first_stage_confounders['backdoor']
    Z2 = identified_estimand_nde.mediation_second_stage_confounders['backdoor']
    independence_constraints_nde=[(A,M, tuple(Z1)), (M,Y, tuple(Z2))]

    ci_refuter_ate = model.refute_graph(
        independence_test = {
            'test_for_continuous': 'partial_correlation', 
            'test_for_discrete' : 'conditional_mutual_information'},
        independence_constraints=independence_constraints_ate
    )

    ci_refuter_nde = model.refute_graph(
        independence_test = {
            'test_for_continuous': 'partial_correlation', 
            'test_for_discrete' : 'conditional_mutual_information'},
        independence_constraints=independence_constraints_nde
    )

    return ci_refuter_ate, ci_refuter_nde

def print_ci_result(ci_refuer, refuter_id):
    print(f'{refuter_id}: passed? {ci_refuer.refutation_result}, ' + \
          f'{ci_refuer.number_of_constraints_satisfied} out of {ci_refuer.number_of_constraints_model} satisfied')



def _get_node_values(node_category, column_to_node_map, variable_i=None, suffix=None):
    node_i = None if variable_i is None else column_to_node_map[variable_i]
    node_prefix = config.node_category_to_prefix[node_category]
    if suffix is not None: node_prefix += f'_{suffix}'
    node_0 = f'{node_prefix}0'
    node_ix = None if variable_i is None else int(node_i.replace(node_prefix, ''))
    return node_i, node_ix, node_0, node_prefix

def _get_node_column_names(node_prefix, node_N, node_to_column_map):
    nodes = [f'{node_prefix}{i}' for i in range(node_N+1)]
    columns = [node_to_column_map[Ni] for Ni in nodes]
    index_range = range(1, node_N+1)
    return nodes, columns, index_range

def _update_values(column1, column2, model=None, artifacts=None, data=None, verbose=False):
    if model is not None:
        if data is not None: model._data = data
        model._data = _get_updated_model_data(model, column1, column2)
        if verbose: print(f'Mean of {column1}, {column2}: \n' + str(model._data[[column1, column2]].mean()))
        return 

    # Otherwise, swap values for all models in artifacts dictionary. Store updated model to dict. 
    for treat_type, treat_artifacts in artifacts.items():
        for dtype, dtype_artifacts in treat_artifacts.items():
            model_type_z = dtype_artifacts['model']
            if data is not None: model_type_z._data = data
            model_type_z._data = _get_updated_model_data(model_type_z, column1, column2)
            artifacts[treat_type][dtype]['model'] = model_type_z
            if verbose: print(f'Mean of {column1}, {column2}: \n' + str(model_type_z._data[[column1, column2]].mean()))

    return artifacts

def _estimate_mediated_effect(model, identified_estimand, estimand_name, 
                              estimator_name, estimator_params, treat_type, 
                              node_prefix_to_count, node_to_column_map, verbose=False):
    M = config.treatment_type_to_mediator[treat_type]; M0 = f'{M}0' # get mediator corresponding to this treatment type
    num_M = node_prefix_to_count[M]
    node_Ms, mediator_columns, mediator_indices = _get_node_column_names(M, num_M, node_to_column_map)

    partial_results = []
    for mediator_ix, mediator_i, Mi in zip(mediator_indices, mediator_columns[1:], node_Ms[1:]):
        if verbose: print(f'Mediator: {mediator_i}')
        
        # Swap with mediator of interest. 
        _update_values(M0, Mi, model=model, verbose=verbose)

        try:
            causal_effect = model.estimate_effect(identified_estimand, **estimator_params)
        except ValueError:
            causal_effect = None
        if verbose: print(causal_effect)
        # causal_effects.append(causal_effect)
        partial_results.append([
            (estimand_name, estimator_name, identified_estimand, \
             mediator_ix, Mi, mediator_i), [causal_effect]
        ])

        # Reset swapped mediator. 
        _update_values(M0, Mi, model=model, verbose=verbose)
    
    return partial_results

def _estimate_ate(model, identified_estimand, estimand_name, estimator_name, estimator_params, verbose=False):
    try:
        causal_effect = model.estimate_effect(identified_estimand, **estimator_params)
    except ValueError:
        causal_effect = None
    if verbose: print(causal_effect)

    partial_results = [[(estimand_name, estimator_name, identified_estimand, \
                         None, None, None), [causal_effect]]]

    return partial_results


def load_results(national_artifacts_dir, target_units='att', waves=False):
    results_dfs = []
    wave_suffix = '_byWave' if waves else ''
    files = glob.glob(os.path.join(national_artifacts_dir, f'treatment_effect_results_*_{target_units}{wave_suffix}.pkl'))
    for results_file in files:
        with open(results_file, 'rb') as f:
            results_df = pickle.load(f)
        target_units = results_file.split('_')[-1].strip('.pkl')
        results_df['target_units']=target_units
        results_dfs.append(copy.deepcopy(results_df))

    results_df = pd.concat(results_dfs)
    results_df = results_df.loc[results_df.estimator_name.isin(config_te.estimator_names)]
    return results_df

def refute_estimates(model, identified_estimand, estimator, verbose=False):
    identified_estimand.set_identifier_method("backdoor")

    # Remove Random Subset of Data
    refute_random_subset = model.refute_estimate(
                                identified_estimand,
                                estimator,
                                method_name="data_subset_refuter",
                                subset_fraction=0.9
                            )

    if verbose: print(refute_random_subset)
     
    refute_bootstrap = model.refute_estimate(
                                identified_estimand,
                                estimator,
                                method_name="bootstrap_refuter",
                            )

    if verbose: print(refute_bootstrap)

    # Dummy confounder, how sensitive is estimate to potential unobserved confounding
    res_random = model.refute_estimate(    #A
        identified_estimand,    #A
        estimator,    #A
        method_name="random_common_cause",    #A
        num_simulations=100,    #A
        n_jobs=2    #A
    )    #A
    if verbose: print(res_random)

    # Placebo treatment (will ATE become ~0)
    #A This refuter replaces the treatment variable with a dummy (placebo) variable.
    res_placebo = model.refute_estimate(
        identified_estimand,    #A
        estimator,    #A
        method_name="placebo_treatment_refuter",    #A
        placebo_type="permute",    #A
        num_simulations=100    #A,
    )
    if verbose: print(res_placebo)

    # Unobserved common cause, check if we've adjusted for enough confounding 
    #A Setting up a refuter that adds an unobserved common cause
    try:
        res_unobserved = model.refute_estimate(    #A
            identified_estimand,    #A
            estimator,    #A
            method_name="add_unobserved_common_cause"    #A
        )    #A
    except:
        res_unobserved = None
    if verbose: print(res_unobserved)

    return [refute_random_subset, refute_bootstrap, res_random, res_placebo, res_unobserved]

# for given outcome, for each treatment: plot 0,7,14 day ahead estimates (each estimator colored differently).
def plot_results(results_df, target_units, outcome_units, wave_ix=None):
    results_df_ = results_df if wave_ix is None else results_df.loc[results_df.wave_ix==wave_ix]

    # Get relevant columns
    outcome_base_columns = np.unique(['_'.join(col.strip('_ts_ahead').split('_')[:-1]) for col in results_df_['outcome_i'].unique()])
    ts_ahead = results_df_['ts_ahead'].unique()
    treatments = results_df_['treatment_i'].unique()
    treatments = ['OxCGRT_ContainmentHealthIndex_Average_diff(7)_intervention',
                # 'OxCGRT_EconomicSupportIndex_diff(7)_intervention',
                'OxCGRT_H7_Vaccination.policy_diff(7)_intervention',
                'OxCGRT_V2B_Vaccine.age.eligibility.availability.age.floor..general.population.summary._diff(7)_intervention',]
                # 'OxCGRT_V4_Mandatory.Vaccination..summary._diff(7)_intervention']

    # Get y values to plot treatment effects against
    max_y = 1
    max_y_per_outcome = max_y/len(treatments)
    delta = max_y_per_outcome/(len(ts_ahead)+1)
    y_base = np.arange(0, max_y_per_outcome+delta, delta)[1:-1]

    for outcome_base in outcome_base_columns:
        outcomes = [f'{outcome_base}_{ts}_ts_ahead' for ts in ts_ahead]
        _,ax=plt.subplots(figsize=(7, len(treatments)*3))
        plt.axvline(0, linestyle=':', c='black')
        yticks, yticklabels = [], []
        for i,treatment in enumerate(treatments):
            y_ = y_base+i*max_y_per_outcome
            yticks.extend(y_)
            results_Y_A = results_df_.loc[results_df_.outcome_i.isin(outcomes) & (results_df_.treatment_i == treatment)]

            # Collect estimates, corresponding y values, colors
            te_estimates, y, colors = [], [], []

            for ts_ix,ts in enumerate(ts_ahead):
                ates = results_Y_A.loc[ (results_Y_A.ts_ahead == ts) & (results_Y_A.estimand_name == 'identified_estimand_ate') ][['estimator_name', 'estimand_estimate']]
                nie = results_Y_A.loc[ (results_Y_A.ts_ahead == ts) & (results_Y_A.estimand_name == 'identified_estimand_nie') ]['estimand_estimate'].sum()
                nde = results_Y_A.loc[ (results_Y_A.ts_ahead == ts) & (results_Y_A.estimand_name == 'identified_estimand_nde') ]['estimand_estimate'].mean()
                
                te_estimates.extend([ nde, nie ] + [ ate for ate in ates['estimand_estimate'] ])
                y.extend([ y_[ts_ix] for _ in range(2 + ates.shape[0]) ])
                colors.extend(['orange', 'yellow'] + [ config_te.estimator_to_color[est] for est in ates['estimator_name'] ])

            # Plot, set y axis ticks and labels
            ax.scatter(te_estimates, y, c=colors)

            temp = treatment.split('_')[1:-2]
            if len(temp)==1: treatment_short = temp[-1]
            if 'Index' in treatment: treatment_short = temp[0]
            else: 
                temp[1] = temp[1].replace('.', ' ').replace('summary', '').replace('general population', '').strip(' ')
                treatment_short = f'{temp[1]} ({temp[0]})'

            ticklabels = [f'{d} days' for d in ts_ahead]
            ticklabels[len(ts_ahead)//2] = f'{treatment_short} - ' + ticklabels[len(ts_ahead)//2]
            yticklabels.extend(ticklabels)

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel(f'{target_units.upper()} ({outcome_units})')

        plt.legend()
        plt.show()

