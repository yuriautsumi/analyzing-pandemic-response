import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

import dash_leaflet as dl

dash.register_page(__name__, path='/dashboard', name='Dashboard', title='Dashboard', id='dashboard-link')

option_menu = html.Div([
    html.Div(children=[
        html.Label('Dropdown'),
        dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),

        html.Br(),
        html.Label('Multi-Select Dropdown'),
        dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'],
                     ['Montréal', 'San Francisco'],
                     multi=True),

        html.Br(),
        html.Label('Radio Items'),
        dcc.RadioItems(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),
    ], style={'padding': 10, 'flex': 1}),

    html.Div(children=[
        html.Label('Checkboxes'),
        dcc.Checklist(['New York City', 'Montréal', 'San Francisco'],
                      ['Montréal', 'San Francisco']
        ),

        html.Br(),
        html.Label('Text Input'),
        dcc.Input(value='MTL', type='text'),

        html.Br(),
        html.Label('Slider'),
        dcc.Slider(
            min=0,
            max=9,
            marks={i: f'Label {i}' if i == 1 else str(i) for i in range(1, 6)},
            value=5,
        ),
    ], style={'padding': 10, 'flex': 1})
], style={'display': 'flex', 'flexDirection': 'row'})

layout = dbc.Container([
    dbc.Row([
        # dbc.Col([], width = 2),
        dbc.Col([
            html.H3(['Dashboard']),
            html.P([html.B('App Overview')]),
            option_menu,
            dl.Map(dl.TileLayer(), center=[56,10], zoom=6, style={'height': '50vh'}),
            html.P([html.B('1) Built-in dataset'),html.Br(),
                    'The default dataset used is Air Passenger. The app could work with any .csv file.'], className='guide'),
            html.P([html.B('2) Apply transformations to make the data stationary'),html.Br(),
                    'The tools available on the page are: log and differencing, the Box-Cox plot and the A. Dickey Fuller test.',html.Br(),
                    'Once the data is stationary, check the ACF and PACF plots for suitable model parameters.'], className='guide'),
            html.P([html.B('3) Perform a SARIMA model grid search'),html.Br(),
                    'Choose the train-test split and provide from-to ranges for any parameter.'
                    'The seasonality component of the model can be excluded by leaving all right-hand parameters to 0.',html.Br(),
                    'The 10 top-performer models (according to the AIC score), are shown.'], className='guide'),
            html.P([html.B('4) Set up your final model'),html.Br(),
                    'The parameters for the best model from the previous step are suggested.',html.Br(),
                    'The SARIMA model with the input parameters is automatically fitted to the train data; predictions are made for the train and test sets',html.Br(),
                    'The model residuals ACF and PACF are shown.'], className='guide')
        ], width = 8),
        # dbc.Col([], width = 2)
    ]),
])