"""
Source: 
* Tabs: https://dash-bootstrap-components.opensource.faculty.ai/examples/graphs-in-tabs
"""
import os
import time
import dash
import pickle
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_leaflet as dl

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import constants as C
import dashboard_utils
from tsa_helper import plot_lines
from dashboard_utils import query_data, data_category_to_args, data_category_to_title, \
                            add_wave_shading, data_category_to_title, process_column_name, \
                            data_category_to_tab_update_layouts, update_layouts_y_values, \
                            plot_interventions,population_columns,age_bracket_columns, \
                            income_bracket_columns, race_columns, political_columns, median_income_column


# ---------------------------------------------------------------------------------------------------------------

dash.register_page(__name__, path='/dashboard', name='Dashboard', title='Dashboard', id='dashboard-link')

data_category_menu_options = [
    {"label": data_category_to_title[data_category], "value": data_category}
    for data_category in data_category_to_args.keys()
]

data_category_menu_options_general = list(filter(
    lambda option: 'policy' not in option['value'],
    data_category_menu_options
))
data_category_menu_options_policy = list(filter(
    lambda option: ('policy' in option['value']) and ('intervention' not in option['value']) and ('raw' not in option['value']),
    data_category_menu_options
))

def _get_labels(columns, prefix):
    return list(map(lambda x: x.replace(prefix, '').replace('Less than', '<'), columns))

demographics_category_to_artifacts = {
    'income': dict(
        columns=income_bracket_columns,
        labels=_get_labels(income_bracket_columns, 'Demographic_'),
        misc_info_artifacts=dict(
            column=median_income_column,
            format_func=lambda col: f'\n (${col})'
        )
    ),
    'race': dict(
        columns=race_columns,
        labels=_get_labels(race_columns, 'Demographic_'),
        misc_info_artifacts=None
    ),
    'political': dict(
        columns=political_columns,
        labels=_get_labels(political_columns, 'Misc_'),
        misc_info_artifacts=None
    )
}
demographics_category_menu_options = list(demographics_category_to_artifacts.keys())

# ---------------------------------------------------------------------------------------------------------------

artifacts_file = os.path.join('assets', 'artifacts.pkl')
with open(artifacts_file, 'rb') as f:
    artifacts = pickle.load(f)

national_df, subnational_df = artifacts['unfiltered']['national_df'], artifacts['unfiltered']['subnational_df']
region_code_column, date_column = 'OxCGRT_RegionCode', 'OxCGRT_Date'
date0 = subnational_df[date_column].iloc[0]
subnational_df_head1 = subnational_df.loc[subnational_df[date_column] == date0]
national_df_head1 = national_df.head(1)
rcode_to_rname = artifacts['unfiltered']['rcode_to_rname']
all_data_columns_map = artifacts['unfiltered']['all_data_columns_map']

# ---------------------------------------------------------------------------------------------------------------
# Map 

fig = go.Figure(data=go.Choropleth(
    locations=subnational_df_head1[region_code_column].apply(lambda x: x.split('_')[-1]),
    z=subnational_df_head1['Demographic_Population'].astype(float),
    locationmode='USA-states',
    colorscale='Reds',
    autocolorscale=False,
    text=[f'Population: {row[1]} \nPopulation Density: {row[0]}\n65 years+ (%): {row[2]}' 
          for _,row in subnational_df_head1[population_columns+age_bracket_columns].iterrows()], # hover text
    marker_line_color='white', # line markers between states
    # colorbar_title="Millions USD"
))

fig.update_layout(
    title_text=f'US Population (based on 2020 Census)',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
    clickmode='event+select' # can (de)accumulate selection by pressing Shift
)

geographic_map_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            # dcc.Store(id="locations-store"),#, data=dict),
            dcc.Graph(
                figure=fig,
                id='choropleth'
            ),
            dcc.Loading(
                id="loading-0",
                type="default",
                children=html.Div(id="choropleth")
            ),
        ], id='col-xl-12')
    ])
])

# ---------------------------------------------------------------------------------------------------------------
# Tabs 

tab_id_to_assets = {
    'demographics': {
        'label': 'Demographics',
    },
    'time-series': {
        'label': "Time series",
        'label_style': {}, # optional: add styling
        'active_label_style': {},
        'tab-update-layouts':  data_category_to_tab_update_layouts,
    },
    'policy': {
        'label': "Policy",
        'label_style': {}, # optional: add styling
        'active_label_style': {}
    },
    # 'policy-effect': {
    #     'label': "Policy effect",
    # },
    # 'policy-strategy': {
    #     'label': "Policy strategy"
    # }
}

tabs = dbc.CardHeader(
    dbc.Tabs(
        [   
            dbc.Tab(label=asset['label'], tab_id=tab_id, label_style=asset.get('label_style', {}))
            for (tab_id, asset) in tab_id_to_assets.items()
        ],
        id="tabs",
        active_tab="demographics",
    )
)

content_with_tabs = dbc.Card(
    [
        dcc.Store(id="store"),#, data=dict),
        tabs,
        dbc.CardBody(html.P(id="tab-content", className="card-text")),
        # html.Div(id="tab-content", className="p-4"),
    ]
)

# intervention_subcategory_to_name = dict(zip(['C', 'H', 'E', 'V', 'null'], ['Containment', 'Health', 'Economic', 'Vaccine', 'Index']))
# intervention_name_to_subcategory = dict(zip(['Containment', 'Health', 'Economic', 'Vaccine', 'Index'], ['C', 'H', 'E', 'V', 'null']))
# intervention_dropdown = dcc.Dropdown(['Containment', 'Health', 'Economic', 'Vaccine', 'Index'], 'Containment', id='intervention-dropdown')

@callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        if active_tab == "demographics":
            dropdown_menu = html.Div([
                # dbc.Label("Data Category"),
                dcc.Dropdown(
                    id="demographics-category-button-group",
                    options=demographics_category_menu_options,
                    value="income",
                ),
            ])

            return dbc.Container([
                dropdown_menu,
                # intervention_dropdown,
                dbc.CardBody(html.P(id='demographics-stack')),
                dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=html.Div(id="demographics-stack")
                ),
            ])
        
        elif active_tab == "time-series":
            dropdown_menu = html.Div([
                # dbc.Label("Data Category"),
                dcc.Dropdown(
                    id="time-series-category-button-group",
                    options=data_category_menu_options_general,
                    value="case-death-rate",
                ),
            ])

            return dbc.Container([
                dropdown_menu,
                # intervention_dropdown,
                dbc.CardBody(html.P(id='time-series-stack')),
                dcc.Loading(
                    id="loading-2",
                    type="default",
                    children=html.Div(id="time-series-stack")
                ),
            ])
        
        elif active_tab == "policy":
            dropdown_menu = html.Div([
                # dbc.Label("Data Category"),
                dcc.Dropdown(
                    id="policy-category-button-group",
                    options=data_category_menu_options_policy,
                    value="policy-index",
                ),
            ])

            return dbc.Container([
                dropdown_menu,
                # intervention_dropdown,
                dbc.CardBody(html.P(id='policy-stack')),
                dcc.Loading(
                    id="loading-3",
                    type="default",
                    children=html.Div(id="policy-stack")
                ),
            ])

    return "No tab selected"

# ---------------------------------------------------------------------------------------------------------------
# Update store data based on selected locations
 
STATE_REGION_CODES = subnational_df_head1[region_code_column].unique()
@callback(
        [Output("store", "data")], 
        [Input('choropleth', 'selectedData'),
         Input("store", "data")])
        # [Input("button", "n_clicks")]
def generate_graphs(selectedData, store_data):
    """
    This callback generates three simple graphs from random data.
    """
    # A. Query national data when app loads 
    if not selectedData and not store_data:
        # Query data, build dataframe, plot
        store_data = [{ 'national': {}, 'locations': [] }]
        for data_category in data_category_to_args.keys():
            store_data[0]['national'][data_category] = store_data[0]['national'].get(data_category, {})

            # Query data
            queried_data, index_columns, data_columns  = query_data(
                data_category, 'national', region_code=None)
            data_columns = [process_column_name(col, data_category) for col in data_columns] 
            temp_df = pd.DataFrame(
                list(map(lambda x: (pd.Timestamp(x[0]),) + x[-len(data_columns):], queried_data)),
                columns=index_columns[:1]+data_columns
            )

            # Build time series plot
            fig = go.Figure()
            # intervention_min_max = {}
            # for intervention_category, (threshold) in intervention_artifacts_by_data_category: 
            #     fig, intervention_df = plot_interventions(intervention_category, 'national', None, 
            #                                               index_columns[0], symbol, threshold, fig)
            #     intervention_min_max[intervention_category] = dict(zip(intervention_df.iloc[:, 1:].max().index, 
            #                                                            zip(intervention_df.iloc[:, 1:].min().values, intervention_df.iloc[:, 1:].max().values)))
            
            # Plot time series
            fig = plot_lines(temp_df, index_columns[0], data_columns, title=f'{C.COUNTRY_NAME}', 
                             fig=fig, legend=False, dimensions=(None, 300))#, dimensions=(None, 300), vlines=vlines, title=f'{policy_col}', fig=fig)
            
            # Plot interventions
            _, _, _, (ymin,ymax),_ = data_category_to_args[data_category]
            intervention_category_to_N = {'policy-index-intervention': {}, 'policy-raw-intervention': {}}
            for intervention_category, subcategory in zip(['policy-index-intervention']+['policy-raw-intervention']*4, [None, 'C', 'H', 'E', 'V']):
                fig, N_interventions = plot_interventions(intervention_category, 'national', None, index_columns[0], 
                                                          ymin, ymax, fig, None, None, subcategory=subcategory)
                intervention_category_to_N[intervention_category][subcategory] = N_interventions

            fig.update_layout(title_text=data_category_to_title[data_category])

            store_data[0]['national'][data_category]['time-series'] = fig
            store_data[0]['national'][data_category]['interventions'] = intervention_category_to_N
        
        return store_data#, locations
    
    # B. Query additional state data.
    # Note: Input (when data is not None, is a dictionary)
    elif selectedData is not None:
        # Process selected states
        selected = [f'US_' + sdata['location'] for sdata in selectedData['points']]
        for i,rcode in enumerate(STATE_REGION_CODES): 
            # Update if no longer selected
            if  (rcode not in selected) and (rcode in store_data['locations']): store_data['locations'].remove(rcode)
            # Add if newly selected
            elif (rcode in selected) and (rcode not in store_data['locations']): 
                store_data['locations'].append(rcode)

                # Query data for state if not already in store
                if rcode in store_data: continue
                store_data[rcode] = {}
                for data_category in data_category_to_args.keys():
                    store_data[rcode][data_category] = {}

                    # Query data
                    queried_data, index_columns, data_columns  = query_data(
                        data_category, 'subnational', region_code=rcode)
                    data_columns = [process_column_name(col, data_category) for col in data_columns]
                    temp_df = pd.DataFrame(
                        list(map(lambda x: (pd.Timestamp(x[0]),) + x[-len(data_columns):], queried_data)),
                        columns=index_columns[:1]+data_columns
                    )

                    # Build time series plot
                    fig = go.Figure()
                    fig = plot_lines(temp_df, index_columns[0], data_columns, title=f'{rcode_to_rname[rcode]}', 
                                     fig=fig, legend=False, dimensions=(None, 300))#, vlines=vlines, title=f'{policy_col}', fig=fig)

                    # Plot interventions
                    _, _, _, (ymin,ymax),_ = data_category_to_args[data_category]; intervention_category_to_N = {}
                    intervention_category_to_N = {'policy-index-intervention': {}, 'policy-raw-intervention': {}}
                    for intervention_category, subcategory in zip(['policy-index-intervention']+['policy-raw-intervention']*4, [None, 'C', 'H', 'E', 'V']):
                        fig, N_interventions = plot_interventions(intervention_category, 'subnational', rcode, index_columns[0], 
                                                                  ymin, ymax, fig, None, None, subcategory=subcategory)
                        intervention_category_to_N[intervention_category][subcategory] = N_interventions

                    fig.update_layout(title_text=data_category_to_title[data_category])

                    store_data[rcode][data_category]['time-series'] = fig
                    store_data[rcode][data_category]['interventions'] = intervention_category_to_N
    
    return [store_data]#, locations


# ---------------------------------------------------------------------------------------------------------------
# Render tabs

def _render_tab(value, data):
    N_rows = len(data['locations'])+1
    subplot_titles = [C.COUNTRY_NAME] + [rcode_to_rname[rcode] for rcode in data['locations']]
    fig = make_subplots(
        rows=N_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": True}] for _ in range(N_rows)]
    )

    for trace in data['national'][value]["time-series"]['data']:
        fig.add_trace(trace, row=1, col=1)
    
    for i, rcode in enumerate(data['locations']):
        for trace in data[rcode][value]["time-series"]['data']:
            fig.add_trace(trace, row=i+2, col=1)

    # Add shading by wave
    _, _, _, (ymin,ymax),_ = data_category_to_args[value]
    fig = add_wave_shading(fig)

    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified", 
                      hoverlabel=dict(font_size=9),
                    #   yaxis=dict(title=yaxis_title, side='left')
                      )
    
    # Add menu to update visible elements
    tab_id_to_assets['time-series']['tab-update-layouts'][value][0]['y'] = update_layouts_y_values[N_rows-1]
    # intervention_visible_buttons = []
    # selected_subcategory = intervention_name_to_subcategory[intervention_type]
    # # Iterate over intervention by category to indicate which ones are visible
    # for intervention_category, subcategory in zip(['policy-index-intervention']+['policy-raw-intervention']*4, ['null', 'C', 'H', 'E', 'V']):
    #     N_interventions = data['national'][value]['interventions'][intervention_category][subcategory]
    #     intervention_visible_buttons.extend([(selected_subcategory==subcategory)] * N_interventions)
    # print(intervention_visible_buttons)
    # for i, buttons_arg in enumerate(tab_id_to_assets['time-series']['tab-update-layouts'][value][0]['buttons']):
    #     tab_id_to_assets['time-series']['tab-update-layouts'][value][0]['buttons'][i]['args'][0]['visible'] = \
    #     buttons_arg['args'][0]['visible'] + intervention_visible_buttons

    fig.update_layout(updatemenus=tab_id_to_assets['time-series']['tab-update-layouts'][value])

    # Adjust y axes
    (ymin, ymax), yaxis_title = data_category_to_args[value][3], data_category_to_args[value][4]
    fig.update_yaxes(range=[ymin, ymax], secondary_y=False) 

    # Set y axis label
    yaxis_titles = [yaxis_title for _ in range(N_rows)]
    if type(yaxis_title)==list: yaxis_titles = yaxis_title
    yaxis_keys = ['yaxis'] + [f'yaxis{i}' for i in range(1,N_rows)]
    
    for i,(yaxis_key,yaxis_title) in enumerate(zip(yaxis_keys,yaxis_titles)):
        fig.layout[yaxis_key].update(title=yaxis_title)
        fig.layout.annotations[i].update(text=subplot_titles[i])

    fig.update_layout(
        height=(N_rows+1)*200, # HACK
        showlegend=False,
        # title_text="",
    )

    return dbc.Container([ # init with national plot
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig)], id='col-xl-12')
        ])
    ])

@callback(
    Output("time-series-stack", "children"),
    [Input("time-series-category-button-group", "value"),
     Input("store", "data")]
)
def _render_time_series_tab(value, data):
    return _render_tab(value, data)

@callback(
    Output("policy-stack", "children"),
    [Input("policy-category-button-group", "value"),
     Input("store", "data")]
)
def _render_policy_tab(value, data):
    return _render_tab(value, data)

layout = dbc.Container([
    dbc.Row([
        # dbc.Col([], width = 2),
        dbc.Col([
            html.H3(['Dashboard']),
            # html.P([html.B('App Overview')]),
            geographic_map_content,
            content_with_tabs
        ], width=12),
        # dbc.Col([], width = 2)
    ]),
])

@callback(
    Output("demographics-stack", "children"),
    [Input("demographics-category-button-group", "value"),
     Input("store", "data")]
)
def _render_demographics_tab(value, data):
    N_rows = len(data['locations'])+1
    locations = data['locations']
    y_data0 = [C.COUNTRY_NAME] + [rcode_to_rname[rcode] for rcode in locations]

    artifacts = demographics_category_to_artifacts[value]
    columns, labels = artifacts['columns'], artifacts['labels']
    misc_info_artifacts = artifacts['misc_info_artifacts']
    
    if misc_info_artifacts is None:
        y_data = y_data0
    else:
        column, f = misc_info_artifacts['column'], misc_info_artifacts['format_func']
        x2_data = pd.concat((
            national_df_head1[column],
            subnational_df_head1.loc[subnational_df_head1[region_code_column].isin(locations)][column]
        )).values
        y_data = [region+f(x) for region,x in zip(y_data0,x2_data)]

    x_data = pd.concat((
        national_df_head1[columns],
        subnational_df_head1[columns].loc[subnational_df_head1[region_code_column].isin(locations)]
    )).values

    x_data = x_data[::-1]
    y_data = y_data[::-1]

    fig = dashboard_utils.plot_stacked_bar_chart(x_data, y_data, labels, '')
    fig.update_layout(
        height=(N_rows+1)*150, # HACK
        showlegend=False,
        # title_text="",
    )

    return dbc.Container([ # init with national plot
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig)], id='col-xl-12')
        ])
    ])
