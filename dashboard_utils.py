import os
import re
import pickle
import importlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import utils 
import tsa_helper
importlib.reload(utils)
importlib.reload(tsa_helper)

PLOTLY_COLORS = px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly 
PLOTLY_COLORS_SUBSET = px.colors.qualitative.Plotly 
policy_category_to_color = dict(zip(['C', 'E', 'H', 'V'], PLOTLY_COLORS))

# Load artifcts
artifacts_file = os.path.join('assets', 'artifacts.pkl')
with open(artifacts_file, 'rb') as f:
    artifacts = pickle.load(f)
data_to_column_map_unfiltered_processed_columns = artifacts['unfiltered']['data_to_column_map_processed_columns']

region_level_to_engine = { 
    region_level: utils.create_engine(f'sqlite:///assets/{region_level}_unfiltered.db', echo=False)
    for region_level in ['subnational', 'national']
}

#------------------------------------------------------------------------
income_bracket_columns = \
[ 'Demographic_Less than $10,000',
  'Demographic_$10,000 to $14,999',
  'Demographic_$15,000 to $24,999',
  'Demographic_$25,000 to $34,999',
  'Demographic_$35,000 to $49,999']
income_bracket_labels = list(map(lambda x: x.replace('Demographic_', '').replace('Less than', '<'), income_bracket_columns))
median_income_column = 'Demographic_Median income (dollars)'
race_columns = \
[ 'Demographic_White',
  'Demographic_Black or African American',
  'Demographic_Hispanic or Latino (of any race)',]
age_bracket_columns = \
['Demographic_65 years and over',]

population_columns = \
['Misc_Population Density  - weighted [per km2]',
 'Demographic_Population']
political_columns = \
['Misc_Democratic', 'Misc_Republican']
#------------------------------------------------------------------------


data_category_to_title = {
    'case-death-rate': 'Confirmed Cases & Deaths',
    'mobility': 'Mobility by location category',
    'vax-rate': 'Vaccination Rate',
    'policy-index': 'Policy Index',
    'policy-raw': 'Policy Indicator',
    'policy-index-intervention': 'Policy Index Intervention',
    'policy-raw-intervention': 'Policy Indicator Intervention',
    'hospital': 'Weekly COVID 19 Admissions',
}

def process_column_name(col, data_category):
    if data_category == 'case-death-rate':
        return col.replace('OxCGRT_', '').replace('_diff7', '').replace('Confirmed', 'Confirmed ')
    elif data_category == 'mobility':
        return col.replace('Mobility_', '').replace('_', ' ').replace(' percent change from baseline', '')
    elif data_category == 'vax-rate':
        return col.replace('OxCGRT_', '').replace('_', ' ').replace('people ', '')
    elif data_category == 'policy-index':
        return col.replace('OxCGRT_', '').replace('_', ' ').replace('Index', ' Index').replace(' Average', '')
    elif data_category == 'policy-index-intervention':
        return col.replace('OxCGRT_', '').replace('_', ' ').replace('Index', ' Index').replace(' Average', '').replace('_intervention', '')
    elif data_category == 'policy-raw-intervention':
        return col.replace('OxCGRT_', '').replace('_', ' ').replace('_intervention', '')
    elif data_category == 'hospital': 
        return col.replace('Hospital_', '').replace('_', ' ').replace('100 000', '100k')
    return col

def _get_processed_column_names(data_category, return_visible_booleans=False):
    data_columns = []
    key1, key2s, funcs, _, _, = data_category_to_args[data_category]
    for key2, f in zip(key2s, funcs):
        data_columns.extend(f(data_to_column_map_unfiltered_processed_columns[key1][key2])) 

    processed_data_columns = [process_column_name(col, data_category) for col in data_columns]
    if return_visible_booleans:
        N = len(data_columns) 
        visible_booleans = [[True for _ in range(N)]] + [[False for _ in range(i)]+[True]+[False for _ in range(i+1,N)] for i in range(N)]
        processed_data_columns = [data_category_to_title[data_category]] + processed_data_columns
    else:
        visible_booleans = None

    return processed_data_columns, visible_booleans

y_axis_label_by_data_category = ['per 100k people', '% change from baseline', 'per 100 people', 
                                 'scaled index (0-100)', None, None, None, ['total admissions', 'per 100k people', 'total admissions']]
# is_2plot_by_data_category = [(True, ('Cases', 'Deaths')), (False, None), (False, None), (False, None),
#                              (False, None), (False, None), (False, None), (True, ('Weekly Admissions per 100k', 'Weekly Admissions'))]
intervention_artifacts_by_data_category = [('policy-index-intervention',  0.5), 
                                           ('policy-raw-intervention', 5),]
data_category_to_intervention_artifacts = {
    'policy-index-intervention': 0.5,
    'policy-raw-intervention': 5
}

# key 1, key 2s, column transformation funcs, ymin/max, func to process each column name, yaxis label, indicator for 1 plot or 2
data_category_to_args = { 
    'case-death-rate': ('OxCGRT', ['outcome_columns_diff'], [lambda x:x], [-10, 3500] ),
    'mobility': ('mobility', ['mobility'], [lambda x:x], [-100, 400] ),
    'vax-rate': ('OxCGRT', ['vax_status_columns'], [lambda x:x], [-10, 110] ),
    'policy-index': ('OxCGRT', ['policy_index_columns'], [lambda x:x], [-10, 110]) ,
    'policy-raw': ('OxCGRT', ['policy_raw_columns_intervention'], [ lambda cdict: [ v.replace('_diff7_intervention', '') for _,v in cdict.items()] ], [-1, 6] ),
    # 'policy-raw': ('OxCGRT', ['policy_raw_columns_intervention'], [lambda x:list(map(lambda col: col.strip('_diff(7)'), x))], [0, 6] ),
    'policy-index-intervention': ('OxCGRT', ['policy_index_columns_intervention'], [ lambda cdict: [ v for _,v in cdict.items()] ], [-100, 100] ),
    'policy-raw-intervention': ('OxCGRT', ['policy_raw_columns_intervention'], [ lambda cdict: [ v for _,v in cdict.items()] ], [-6, 6] ),
    'hospital': ('hospital', ['per_100_000_columns', 'other'], [lambda x: [x[2]], lambda x: [x[2]]], [-10, 75] ),
}
data_category_to_args = {
    k: v+(y_axis_label_by_data_category[i], ) for i,(k,v) in enumerate(data_category_to_args.items())
}

update_layouts_y_values = [1.3] + ((1.2-(np.log(np.arange(1,52, 0.25))/np.log(12000)))[2:]).tolist()
# update_layouts_y_values = [1.3, 1.15, 1.12, 1.1, 1.1]
data_category_to_tab_update_layouts = {
    data_category: [dict(
        # type="buttons",
        active=0,
        # direction="right",
        showactive=True,
        xanchor="left",
        y=0,
        yanchor="top",
        buttons=list([
            dict(
                label=processed_col,#"All",
                method="update",
                args=[{"visible": visible},
                    {
                        # "title": processed_col,
                    "annotations": []
                    }]
            )
            for processed_col,visible in zip(*_get_processed_column_names(data_category, return_visible_booleans=True))
        ])
    )]
    for data_category in data_category_to_args.keys()
}

# def update_visible(buttons_arg, interventions_visible):
#     buttons_arg['args'][0]['visible'] = interventions_visible + buttons_arg['args'][0]['visible']


index_columns = np.concatenate([data_to_column_map_unfiltered_processed_columns['OxCGRT'][c] for c in ['date', 'location']]).tolist()
region_code_column = 'OxCGRT_RegionCode'

def query_data(data_category, region_level, region_code=None):
    data_columns = []
    key1, key2s, funcs, _, _, = data_category_to_args[data_category]
    for key2, f in zip(key2s, funcs):
        data_columns.extend(f(data_to_column_map_unfiltered_processed_columns[key1][key2])) 

    command = f"SELECT " + ','.join(index_columns + data_columns) + f" FROM {region_level}_unfiltered;"
    if region_level == 'subnational': 
        command = command[:-1] + f' WHERE {region_code_column} IS \'{region_code}\''

    engine = region_level_to_engine[region_level]
    with engine.connect() as conn:
        queried_data = conn.execute(utils.text(command)).fetchall()

    return queried_data, index_columns, data_columns

### Plotting

# colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
#           'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
#           'rgba(190, 192, 213, 1)'] # purple gradient

covid_wave_map = {
    'Wave_1': (('2020-03-01', '2020-06-01'), 'red'),
    'Wave_2': (('2020-06-01', '2020-09-15'), 'blue'),
    'Wave_3': (('2020-09-15', '2021-03-15'), 'green'),
    'Wave_4': (('2021-03-15', '2021-07-01'), 'orange'),
    'Wave_5': (('2021-07-01', '2022-01-01'), 'pink'),
    'Wave_6': (('2022-01-01', '2022-02-15'), 'yellow'),
    # 'Wave_6': _get_wave_fn('2022-01-01', '2022-03-15'),
}

def plot_stacked_bar_chart(x_data, y_data, top_labels, title):
    # Source: Plotly 
    fig = go.Figure()

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                text=top_labels[i],
                textfont_size=9,
                textfont_color='white',
                marker=dict(
                    color=PLOTLY_COLORS_SUBSET[i],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            ))

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            # showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        # paper_bgcolor='rgb(248, 248, 255)',
        # plot_bgcolor='rgb(248, 248, 255)',
        margin=dict(l=120, r=10, t=140, b=80),
        showlegend=False,
    )

    # fig.update_layout(annotations=annotations)
    fig.update_layout(dict(autosize=False,width=750,height=x_data.shape[0]*125,title=title))
    # fig.show()
    return fig

def _add_shading(fig, x0, x1, params):
    params['x0'] = x0; params['x1'] = x1
    fig.add_vrect(**params)
    return fig


def add_wave_shading(fig):
    for wave, ((x0,x1), color) in covid_wave_map.items():
        fig = _add_shading(fig, x0, x1, dict(
            annotation_text=wave.replace('_', ' '), annotation_position="top right",
            annotation=dict(font_size=11, font_family="Arial", font_color='black'),
            fillcolor=color, opacity=0.05, line_width=0            
        ))
    return fig

def plot_interventions(data_category, region_level, region_code, date_column, ymin, ymax, fig, subcategory=None):
    threshold = data_category_to_intervention_artifacts[data_category]

    # Query data
    queried_data, index_columns, data_columns  = query_data(data_category, region_level, region_code=region_code)
    temp_df = pd.DataFrame(
        list(map(lambda x: (pd.Timestamp(x[0]),) + x[-len(data_columns):], queried_data)),
        columns=index_columns[:1]+data_columns
    )
    temp_df.index = temp_df[date_column]
    y_columns = temp_df.columns[1:]

    # Get tuples of (date, intervention value, name, color)
    if 'raw' in data_category: # Discrete policy
        p = re.compile(f'OxCGRT_{subcategory}')
        # color = policy_category_to_color[subcategory]

        columns = list(filter(lambda x: p.match(x) is not None, y_columns))
        interventions = [ ]
        for col in columns:
            try:
                filtered = temp_df[col].loc[temp_df[col].abs() >= threshold]
                interventions.append(filtered)
            except:
                continue
        
        interventions = [zip(df.index, df.values, 
                                [process_column_name(df.name, data_category) for _ in range(df.shape[0])]) for df in interventions]

        temp = [list(zipped) for zipped in interventions]
        intervention_tuples = temp[0]
        for a2 in temp[1:]:
            intervention_tuples+=a2

    else: # Continuous interventions
        interventions = [ ]
        for col in y_columns:
            try:
                filtered = temp_df[col].loc[temp_df[col].abs() >= threshold]
                interventions.append(filtered)
            except:
                continue

        interventions = [zip(df.index, df.values, 
                                [process_column_name(df.name, data_category) for _ in range(df.shape[0])]) for df in interventions]

        temp = [list(zipped) for zipped in interventions]
        intervention_tuples = temp[0]
        for a2 in temp[1:]:
            intervention_tuples+=a2

    _get_color = lambda val: 'blue' if val<0 else 'red'

    # Plot 
    delta = ymax-ymin
    for date, val, name in intervention_tuples:
        fig.add_trace(
            go.Scatter(
                x=[date, date],
                y=[ymin-delta*0.2, ymax+delta*0.2],
                mode='lines',
                line_width=1,
                line_color=_get_color(val),
                line_dash="dot",
                # opacity=0.5,
                # line_style='dot',
                name=name,
                showlegend=False,
                hovertemplate=f'{val:.2f}'))
    
    return fig, len(intervention_tuples)

# def plot_interventions(data_category, region_level, region_code, date_column, symbol, threshold, fig):
#     # Query data
#     queried_data, index_columns, data_columns  = query_data(data_category, region_level, region_code=region_code)
#     temp_df = pd.DataFrame(
#         list(map(lambda x: (pd.Timestamp(x[0]),) + x[-len(data_columns):], queried_data)),
#         columns=index_columns[:1]+data_columns
#     )

#     y_columns = temp_df.columns[1:]
#     # ymin2, ymax2 = temp_df.iloc[:,1:].min().min(), temp_df.iloc[:,1:].max().max()

#     for y_col, color in zip(y_columns, PLOTLY_COLORS):
#         dates = temp_df[date_column].loc[temp_df[y_col].abs() >= threshold]
#         interventions = temp_df[y_col].loc[temp_df[y_col].abs() >= threshold]

#         # For each type of treatment: plot
#         for date,treat_val in zip(dates,interventions): 
#             fig.add_trace(
#                 go.Scatter(
#                     x=[date, date],
#                     y=[0, treat_val],
#                     mode='lines',
#                     line_width=3,
#                     line_color='black',
#                     showlegend=False,
#                     hoverinfo='none',))
#                     # layout=dict()
#             # ), secondary_y=True)
#             fig.add_trace(
#                 go.Scatter(
#                     x=[date],
#                     y=[treat_val],
#                     mode='markers',
#                     marker=dict(
#                         size=7,
#                         color=color,
#                         symbol=symbol
#                     ),
#                     name=y_col,))
#                     # hovertemplate=f'{treat_val:.2f}'
#             # ), secondary_y=True)

#     return fig, temp_df

# def adjust_yaxes(ymin1, ymax1, ymin2, ymax2, fig):
#     unit1_per_unit2 = (ymax1-ymin1) / (ymax2-ymin2)
#     delta_lt0 = (-ymin1) if (-ymin1) > (-ymin2*unit1_per_unit2) else (-ymin2*unit1_per_unit2)
#     delta_gt0 = (ymax1) if (ymax1) > (ymax2*unit1_per_unit2) else (ymax2*unit1_per_unit2)

#     ymin1, ymax1 = delta_lt0*1.1, delta_gt0*1.1
#     ymin2, ymax2 = ymin1/unit1_per_unit2, ymax1/unit1_per_unit2

#     fig.update_yaxes(range=[-ymin1, ymax1], secondary_y=False) 
#     fig.update_yaxes(range=[-ymin2, ymax2], secondary_y=True) 

#     return fig

