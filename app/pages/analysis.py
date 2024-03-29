import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/analysis', name='Analysis', title='Analysis', id='analysis-link')

layout = dbc.Container([
    dbc.Row([
        # dbc.Col([], width = 2),
        dbc.Col([
            html.H3(['Analysis']),
            html.P([html.B('Part 1: Analyzing Policy Effectiveness')]),
            html.P(['To analyze individual policies\' effectiveness, I use methods such as covariate adjustment and IPTW to estimate intervention effect ' + \
                    'in 1 to 2 weeks on the outcome (e.g. case rate). ']),
            html.P([html.B('Part 2: Analyzing Response Strategy')]),
            html.P(['To analyze the dynamic response strategy, I use g-methods (in particular g-computation) to simulate covariates and outcomes under ' + \
                    'national and state level strategies to measure how much more (or less) effect the state level strategy had over the national level strategy. ']),
        ], width = 8),
        # dbc.Col([], width = 2)
    ]),
])