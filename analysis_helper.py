import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def get_correlation(policy_data):
    """ Measures correlation between continuous vars """
    return policy_data.corr()

def _compute_chi2_correlation(pair_dataframe):
    """ Input: dataframe with 2 (categorical) columns to compare """
    contingency_table = pd.crosstab(
        pair_dataframe.iloc[:, 0],
        pair_dataframe.iloc[:, 1]
    )
    
    res = chi2_contingency(contingency_table)
    return res.statistic, res.pvalue

def compute_chi2_correlation(policy_data):
    policy_columns = policy_data.columns
    N = len(policy_columns)
    chi2_statistics, chi2_pvalues = np.ones((N, N))*np.nan, np.ones((N, N))*np.nan
    for i in range(N-1):
        for j in range(i+1, N):
            col1, col2 = policy_columns[i], policy_columns[j]
            contingency_data = policy_data[[col1, col2]].astype(int)
            chi2_statistic, chi2_pvalue = _compute_chi2_correlation(contingency_data)
            chi2_statistics[i, j] = chi2_statistic
            chi2_pvalues[i, j] = chi2_pvalue
    
    chi2_stats_df = pd.DataFrame(chi2_statistics, columns=policy_columns, index=policy_columns)
    chi2_pval_df  = pd.DataFrame(chi2_pvalues, columns=policy_columns, index=policy_columns)

    return chi2_stats_df, chi2_pval_df
    