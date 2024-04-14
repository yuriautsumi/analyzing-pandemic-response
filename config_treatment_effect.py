# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LassoCV
# from sklearn.ensemble import GradientBoostingRegressor
# import dowhy.causal_estimators.linear_regression_estimator
import dowhy

# Map outcome id to outcome 
outcome_categories = \
['OxCGRT_ConfirmedCases_diff(7)', 'OxCGRT_ConfirmedDeaths_diff(7)', 
 'OxCGRT_ConfirmedCases_diff(7)_diff(7)', 'OxCGRT_ConfirmedDeaths_diff(7)_diff(7)', 
 'Mobility_retail_and_recreation_percent_change_from_baseline', 'Mobility_grocery_and_pharmacy_percent_change_from_baseline', 
 'Mobility_transit_stations_percent_change_from_baseline', 'Mobility_workplaces_percent_change_from_baseline']

outcome_ids = \
['cases-daily', 'deaths-daily', 'cases-rate', 'deaths-rate',\
 'mobility-retail', 'mobility-grocery', 'mobility-transit', 'mobility-work']

outcome_id_to_category = dict(zip(outcome_ids, outcome_categories))

# Define parameters for estimators
ate_lr_params = {
    'method_name': "backdoor.linear_regression",
    'control_value': 0,
    'treatment_value': 1, # effect of changing treatment from t=0 to 1
    'method_params': {'need_conditional_estimates': False },
    'confidence_intervals': False,
    'test_significance': False,
}

ate_ps_strat_params = {
    'method_name': "backdoor.propensity_score_stratification",
    'control_value': 0,
    'treatment_value': 1, # effect of changing treatment from t=0 to 1
    'method_params': {'need_conditional_estimates': False },
    'confidence_intervals': False,
    'test_significance': False,
}

ate_ps_match_params = {
    'method_name': "backdoor.propensity_score_weighting",
    'control_value': 0,
    'treatment_value': 1, # effect of changing treatment from t=0 to 1
    'method_params': {"weighting_scheme":"ips_stabilized_weight", 'need_conditional_estimates': False },    #B
    'confidence_intervals': False,
    'test_significance': False,
}

ate_ps_wt_params = {
    'method_name': "backdoor.propensity_score_matching",
    'control_value': 0,
    'treatment_value': 1, # effect of changing treatment from t=0 to 1
    'method_params': {'need_conditional_estimates': False },
    'confidence_intervals': False,
    'test_significance': False,
}

mediation_params = {
    'method_name': "mediation.two_stage_regression",
    'control_value': 0,
    'treatment_value': 1, # effect of changing treatment from t=0 to 1
    'confidence_intervals': False,
    'test_significance': False,
    'method_params': {
        'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
        'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
        'need_conditional_estimates': False ,
    }
}

# estimand_names = ['identified_estimand_ate', 'identified_estimand_ate', 'identified_estimand_ate', \
#                   'identified_estimand_ate', 'identified_estimand_nde', 'identified_estimand_nie']
# estimator_names = ['backdoor_linear_regression', 'backdoor_propensity_score_stratification', 'backdoor_propensity_score_matching', \
#                    'backdoor_propensity_score_weighting', \
#                    'mediation_two_stage_regression', 'mediation_two_stage_regression']
# estimand_estimator_params = [ate_lr_params, ate_ps_strat_params, ate_ps_match_params, ate_ps_wt_params, \
#                              mediation_params, mediation_params]

estimand_names = ['identified_estimand_ate', 'identified_estimand_ate', \
                  'identified_estimand_nde', 'identified_estimand_nie']
estimator_names = ['backdoor_linear_regression', 'backdoor_propensity_score_matching', \
                   'mediation_two_stage_regression', 'mediation_two_stage_regression']
estimand_estimator_params = [ate_lr_params, ate_ps_match_params, \
                             mediation_params, mediation_params]

# estimand_names = ['identified_estimand_ate', 'identified_estimand_nde', 'identified_estimand_nie']
# estimator_names = ['backdoor_linear_regression', 'mediation_two_stage_regression', 'mediation_two_stage_regression']
# estimand_estimator_params = [ate_lr_params, mediation_params, mediation_params]


estimator_to_color = dict(
    backdoor_linear_regression='red',
    backdoor_propensity_score_stratification='purple',
    backdoor_propensity_score_matching='green',
    backdoor_propensity_score_weighting='blue'
)
