""" Helper functions for Plotting. """
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from textwrap import wrap
from datetime import datetime

from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL, MSTL

import data_helper


PLOTLY_COLORS = px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly 
# print(f"PLOTLY_COLORS: {len(PLOTLY_COLORS)}")
# pd.options.plotting.backend = "plotly"

def plot_boxplot(df, columns, n_rows, n_cols, title=None, save_path=None, align_axes=False):
    # n_rows, n_cols = 5, 10 # 5 rows, 10 cols = 50 total vars 
    # plt.figure(figsize=(4*n_rows, 2*n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2*n_rows, 5*n_cols))
    if align_axes: min_, max_ = df.min().min(), df.max().max()
    ax_ = ax.ravel()
    for i in range(len(columns)):
        sns.set_style('whitegrid')
        sns.boxplot(df[columns[i]].values,color='green',orient='v', ax=ax_[i])
        ax_[i].set_title('\n'.join(wrap(columns[i], 40)), fontsize=8)
        if align_axes: ax_[i].set_ylim(min_, max_)
        # plt.title(l[i], wrap=True)
        plt.tight_layout()
    if title is not None: fig.suptitle(title, y=1.05, fontsize=16)
    if save_path is not None: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_dist(df, columns, n_rows, n_cols, title=None, save_path=None, align_axes=False):
    # n_rows, n_cols = 5, 10 # 5 rows, 10 cols = 50 total vars 
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3*n_rows, 8*n_cols))
    if align_axes: min_, max_ = df.min().min(), df.max().max()    
    ax_ = ax.ravel()
    for i in range(len(columns)):
        g = sns.histplot(ax=ax_[i], data=df, x=columns[i], kde=True)
        # g.set(xticklabels=[])
        g.set(xlabel=None)
        g.set(ylabel=None)
        ax_[i].set_title('\n'.join(wrap(columns[i], 20)), fontsize=10)
        if align_axes: ax_[i].set_ylim(min_, max_)
        plt.tight_layout()
    if title is not None: fig.suptitle(title, y=1.05, fontsize=16)
    if save_path is not None: plt.savefig(save_path, dpi=300)
    plt.close()


def get_vline_labels(df, date_col, d):
    # Get monthly labels 
    dates = [pd.to_datetime(x) for x in df[date_col].iloc[d:].values]
    month_starts = list(filter(lambda x: x.day == 1, dates))
    year_starts = list(filter(lambda x: x.month == 1 and x.day == 1, dates))
    return month_starts, year_starts


def plot_lines(df, x_col, y_columns, vlines=None, fig=None, fig_params=None, same_axis=True, title=None, legend=True, dimensions=None):
    """ Plot run-sequence plot. 
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data to be plotted.
        x_col (str): The column name in the dataframe to be used as the x-axis values.
        y_columns (list): A list of column names in the dataframe to be used as the y-axis values.
        vlines (dict, optional): A dictionary where the keys are line colors and the values are lists of x-values 
            where the vertical lines should be drawn. Defaults to None.
        same_axis (bool, optional): Determines whether the y-axis should be shared between the lines or if each line 
            should have its own y-axis. By default, it is set to True.
        title (str, optional): The title of the plot. Defaults to None.
    """
    def _plot(params, plot_type='line'):
        if plot_type == 'line':
            if is_plotly: fig.add_trace(go.Scatter(**params), **fig_params)
            else: fig.plot(*params)
        else: # vline
            if is_plotly: fig.add_vline(**params)
            else: fig.axvline(**params)

    if fig is None: fig = go.Figure(); is_plotly = True
    else: is_plotly = type(fig)!=matplotlib.axes._axes.Axes
    if fig_params is None: fig_params = {}

    # Add traces for both lines to the same plot
    # print(f'NUM y_columns: {len(y_columns)}')
    for i, y_col in enumerate(y_columns):
        if is_plotly: params = {'x': df[x_col], 'y': df[y_col], 'name': y_col, 'yaxis': f'y{i+1}', 'marker': {'color': PLOTLY_COLORS[i]}}
        else: 
            params = (df[x_col], df[y_col]) #{'x': df[x_col], 'y': df[y_col]}
            if not same_axis and i==1: fig = fig.twinx()
        _plot(params, plot_type='line')
        # fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], name=y_col, yaxis=f'y{i+1}'), **fig_params)

    # Add vertical lines 
    if vlines is not None: 
        for c, (xs, xnames) in vlines.items():
            for x,name in zip(xs, xnames):
                if is_plotly: params = {'x': x, 'line_color': c, 'line_dash': 'dot', 'opacity': 0.5}
                else: params = {'x': x, 'color': c, 'linestyle': '.', 'alpha': 0.5}
                # if not same_axis: fig, fig1 = fig0, fig
                _plot(params, plot_type='vline')
                # fig0, fig = fig, fig1; _plot(params, plot_type='vline')
                # fig.add_vline(x=x, line_color=c, line_dash='dot', opacity=0.5) #line_width=3, line_dash="dash", line_color="green")

    # Update the layout with separate y-axis labels
    if title is not None: 
        if is_plotly: fig.update_layout(title=title)
        else: fig.set_title(title)
    if not same_axis and is_plotly: # assumes only 2 column
        fig.update_layout(
            yaxis=dict(title=y_columns[0], side='left'),
            yaxis2=dict(title=y_columns[1], side='right', overlaying='y'),
        )
    fig.update(layout_showlegend=legend)
    if dimensions is not None: fig.update_layout(dict(autosize=True,width=dimensions[0],height=dimensions[1],))
    if fig is None: fig.show()
    else: return fig

def difference(df, col, d, groupby_col=None, scale_col=None, pct_change=False, add_to_df=False):
    """
    Calculates the difference of a column in a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the column.
    col (str): The name of the column to calculate the difference for.
    d (int): The number of periods to shift for the difference calculation.
    scale_col (str, optional): The name of the column to scale the difference by. Defaults to None.
    add_to_df (bool, optional): Whether to add the difference as a new column to the DataFrame. Defaults to False.

    Returns:
    Series: The difference of the column.
    """
    if groupby_col is None: 
        if pct_change: diff = df[col].pct_change(d)
        else: diff = df[col].diff(d) 
    else: 
        if pct_change: diff = df.groupby(groupby_col, dropna=False)[col].pct_change(d) * 100
        else: diff = df.groupby(groupby_col, dropna=False)[col].diff(d)
    # if scale_col is not None: diff = diff / df[scale_col] * 100 # compute as a %age
    if scale_col is not None: diff = diff / df[scale_col] * 1e5 # compute as # cases per 100,000 people
    if add_to_df: 
        if pct_change: df.loc[:, (f'{col}_pctchange({d})',)] = diff
        else: df.loc[:, (f'{col}_diff({d})',)] = diff
    return diff

def difference_outcomes(df, outcome_columns, d, groupby_col=None, scale_col=None, pct_change=False, add_to_df=False):
    all_diff_dfs = []
    for outcome_col in outcome_columns:
        diff_df = difference(df, outcome_col, d, groupby_col=groupby_col, scale_col=scale_col, add_to_df=add_to_df) 
        if not add_to_df: all_diff_dfs.append(diff_df)

    suffix = f'pctchange({d})' if pct_change else f'diff({d})'
    diff_outcome_columns = [f'{col}_{suffix}' for col in outcome_columns]
    diff_df = None if not all_diff_dfs else pd.concat(all_diff_dfs, axis=1) 
    if diff_df is not None: diff_df.columns = diff_outcome_columns 
    return diff_df, diff_outcome_columns

def _add_forecast_outcomes(df, groupby_column, outcome_column, outcome_ts_ahead, T):
    # Add forecast outcomes for each ts value
    for ts in outcome_ts_ahead:
        # Lag the outcomes
        temp1 = df[[groupby_column, outcome_column]].groupby(groupby_column, dropna=False).head(ts)
        temp2 = df[[groupby_column, outcome_column]].groupby(groupby_column, dropna=False).tail(T-ts)
        temp1.index += (T-ts)
        temp2.index -= (ts)
        temp3 = pd.concat((temp1, temp2), axis=0)
        temp3.sort_index(inplace=True)
        
        # Update dataframe
        temp_lagged_outcome = temp3.iloc[:, 1]
        df.loc[:, f'{outcome_column}_{ts}_ts_ahead'] = temp_lagged_outcome

def add_forecast_outcomes(df, groupby_column, outcome_columns, outcome_ts_ahead):
    # Add true outcome columns (0 days ahead, 7 days ahead, 14 days ahead)
    assert df.groupby(groupby_column, dropna=False).size().all(), "Not all same length. Please preprocess."
    T = df.groupby(groupby_column, dropna=False).size().iloc[0]

    # diff_outcome_columns = list(diff_outcome_col_to_lagged_diff_outcome_col.keys()) + list(diff_outcome_col_to_lagged_diff2_outcome_col.keys())
    outcome_ts_ahead_columns_map = {x: [f'{x}_{ts}_ts_ahead' for ts in outcome_ts_ahead] for x in outcome_columns}

    for outcome_col in outcome_columns:
        _add_forecast_outcomes(df, groupby_column, outcome_col, outcome_ts_ahead, T)
    
    return outcome_ts_ahead_columns_map

def compute_plot_monthly_stats(x_df, ax=None, title=None):
    """ Computes and plots mean, variance per month.
    x_df: series with index=dates
    """
    # if fig is None: fig = go.Figure()
    # if fig_params is None: fig_params = {}

    # Compute monthly mean/variance
    # x_df.index = df[columns_map['date'][0]].iloc[1:]

    chunk_by_month = [x for x in x_df.groupby(by=[x_df.index.month, x_df.index.year])]
    months, chunk_by_month = list(zip(*chunk_by_month))
    months = [datetime.strptime(f'{m:02d}-{Y}', '%m-%Y') for (m, Y) in months] # format as datetime

    monthly_mean = [np.mean(x) for x in chunk_by_month]
    monthly_var = [np.var(x) for x in chunk_by_month]

    monthly_stats = pd.DataFrame(
        np.array([months, monthly_mean, monthly_var]).T,
        columns=['date', 'mean', 'var']
    )
    monthly_stats.sort_values(by='date', inplace=True) # sort by date

    # Plot as boxplot over months
    temp = pd.DataFrame(x_df)
    temp['month'] = [pd.to_datetime(datetime.strptime(f'{m:02d}-{Y}', '%m-%Y')) for (m, Y) in zip(x_df.index.month, x_df.index.year)]
    temp['month'] = temp.month.infer_objects()
    # temp.index = [datetime.strptime(f'{m:02d}-{Y}', '%m-%Y') for (m, Y) in zip(x_df.index.month, x_df.index.year)]
    y_name = 0 if x_df.name is None else x_df.name
    if title is not None: 
        temp = temp.rename(columns={y_name: title})
        y_name = title
    # sns.boxplot(x='month',y=y_name,data=temp);plt.show();plt.close()
    # temp.boxplot(column=y_name, by='month', ax=fig, rot=90);plt.tight_layout()
    temp.boxplot(column=y_name, by='month', ax=ax, rot=90);plt.tight_layout()
    if ax is None: plt.show()

    return monthly_stats

def plot_distribution(df, col, ax=None, set_title=False):
    ax1, ax2 = None if ax is None else ax[2], None if ax is None else ax[3]
    df[col].hist(bins=100, ax=ax1)
    if set_title and ax1 is not None: ax1.set_title('Data distribution')
    plt.tight_layout()
    qqplot((df[col]-df[col].mean()), line='s', ax=ax2)
    if set_title and ax2 is not None: ax2.set_title('QQ-plot')
    plt.tight_layout()

# def adf_test(df, col):
#     adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(df[col].dropna())
#     print(f'ADF Test, p-value={pvalue:.2e} (test statistic={adf:.2f})')
#     return {'adf': adf, 'pvalue': pvalue, 'usedlag': usedlag, 'nobs': nobs, 'critical_values': critical_values, 'icbest': icbest}

def stationarity_test(df, col):
    adf_result = adf_test(df[col].dropna())
    kpss_result = kpss_test(df[col].dropna())
    return {'adf_result': adf_result, 'kpss_test': kpss_test}

def adf_test(timeseries):
    """ Source: https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html """
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    # for key, value in dftest[4].items():
    #     dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    return dfoutput

def kpss_test(timeseries):
    """ Source: https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html """
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)
    return kpss_output

def run_stationarity_checks(df, x_col, y_col, vlines=None):
    # Function to check stationarity 
    # Plots: run-sequence, summary stats, histogram, qq-plot
    _, ax = plt.subplots(2, 2, figsize=(17,9)); ax_=ax.flatten()
    plot_lines(df, x_col, [y_col], vlines=vlines, fig=ax_[0], title=f'Run-sequence')
    monthly_stats_df = compute_plot_monthly_stats(df[y_col].dropna(), ax=ax_[1], title='Monthly summary statistics')
    plot_distribution(df, y_col, ax=ax_, set_title=True)
    plt.suptitle(f'Stationarity checks for {y_col}')

    # Tests: ADF & KPSS Tests
    # adf_test_dict = adf_test(df, y_col)
    test_dict = stationarity_test(df, y_col)

    return monthly_stats_df, test_dict

# Maps timestamp to timestamp at the end of the month
to_month_end = lambda ts: pd.Timestamp(year=ts.year, month=ts.month, day=1)-pd.Timedelta('1 day')

def decompose(df, col, dtype='standard', model='additive'):
    if dtype == 'standard': result_cases = seasonal_decompose(df[col].dropna(), model=model, extrapolate_trend='freq')
    elif model == 'multiple': result_cases = MSTL(df[col].dropna()).fit()
    else: result_cases = STL(df[col].dropna()).fit()
    df.loc[:, f'{col}_trend'] = result_cases.trend
    df.loc[:, f'{col}_seasonality'] = result_cases.seasonal
    df.loc[:, f'{col}_residual'] = result_cases.resid
    df.loc[:, f'{col}_seasonally_adjusted'] = result_cases.trend+result_cases.resid
    df.loc[:, f'{col}_log_residual'] = result_cases.resid.apply(lambda x: np.log(x+10)) # Handle heteroscedasticity by applying log transform
    result_cases.plot()







def plot(x, y, params, fig):
    fig.add_trace(go.Scatter(x=x, y=y), **params)
    return fig