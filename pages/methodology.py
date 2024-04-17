import dash
import sqlite3
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/methodology', name='Methodology', title='Methodology', id='methodology-link')

# Get content for page
con = sqlite3.connect("assets/content.db")
cur = con.cursor()
res = cur.execute("SELECT page_content FROM content where page_id=\'methodology-link\'")
methodology_md = res.fetchall()[0][0]
methodology_md1, methodology_md2 = methodology_md.split('### Analysis')

layout = dbc.Container([
    dbc.Col([], width=2),
    dbc.Col([
        dbc.Row([
            dbc.Col([dcc.Markdown(methodology_md1, mathjax=True)])
        ]),
        dbc.Row([
            html.Div(
                [html.Img(src=dash.get_asset_url('causal_graph_Acont_ch.png'), width="75%"),
                 dcc.Markdown("(a) Causal DAG with containment and health intervention shown as Acont_ch.")]
            ),
            html.Div(
                [html.Img(src=dash.get_asset_url('causal_graph_Acont_ev.png'), width="75%"),
                 dcc.Markdown("(b) Causal DAG with economic and vaccine intervention shown as Acont_ev.")]
            ),
        ]),
        dbc.Row([
            dbc.Col([dcc.Markdown('### Analysis \n' + methodology_md2, mathjax=True)])
        ]),
    ], width = 8),
    dbc.Col([], width=2),
])
