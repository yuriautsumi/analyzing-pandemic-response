import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='About', title='About', id='about-link')

layout = dbc.Container([
    dbc.Row([
        # dbc.Col([], width = 2),
        dbc.Col([
            html.H3(['About']),
            # html.P([html.B('App Overview')]),
            html.P([
                'This app was created to summarize the results of a project involving analyzing responses to the COVID-19 pandemic, and how they affected outcomes such as case rates and mobility trends in the United States.'
            ])
        ], width = 8),
        # dbc.Col([], width = 2)
    ]),
])