"""
This app creates a collapsible, responsive sidebar layout with
dash-bootstrap-components and some custom css with media queries.

When the screen is small, the sidebar moved to the top of the page, and the
links get hidden in a collapse element. We use a callback to toggle the
collapse when on a small screen, and the custom CSS to hide the toggle, and
force the collapse to stay open when the screen is large.

dcc.Location is used to track the current location. There are two callbacks,
one uses the current location to render the appropriate page content, the other
uses the current location to toggle the "active" properties of the navigation
links.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls

Source: 
* Collapsible Sidebar: https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
"""
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


def _order(page_registry):
    # dash.page_registry.values()
    ix = -1
    ordered_page_registry = [None for _ in range(len(page_registry))]
    for page in page_registry:
        if page['id'] == 'about-link':
            ordered_page_registry[0] = page
        elif page['id'] == 'dataset-link':
            ordered_page_registry[1] = page
        elif page['id'] == 'methodology-link':
            ordered_page_registry[2] = page
        elif page['id'] == 'analysis-link':
            ordered_page_registry[3] = page
        elif page['id'] == 'dashboard-link':
            ordered_page_registry[4] = page
        else:
            ordered_page_registry[ix] = page; ix -= 1
    return ordered_page_registry


app = dash.Dash(
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, 'assets/style.css'],
    # these meta_tags ensure content is scaled correctly on different devices
    # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)

# we use the Row and Col components to construct the sidebar header
# it consists of a title, and a toggle, the latter is hidden on large screens
sidebar_header = dbc.Row(
    [
        dbc.Col(html.P("COVID-19 Data Viz", className="h4")),
        dbc.Col(
            [
                html.Button(
                    # use the Bootstrap navbar-toggler classes to style
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    # the navbar-toggler classes don't set color
                    style={
                        "color": "rgba(0,0,0,.5)",
                        "border-color": "rgba(0,0,0,.1)",
                    },
                    id="navbar-toggle",
                ),
                html.Button(
                    # use the Bootstrap navbar-toggler classes to style
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    # the navbar-toggler classes don't set color
                    style={
                        "color": "rgba(0,0,0,.5)",
                        "border-color": "rgba(0,0,0,.1)",
                    },
                    id="sidebar-toggle",
                ),
            ],
            # the column containing the toggle will be only as wide as the
            # toggle, resulting in the toggle being right aligned
            width="auto",
            # vertically align the toggle in the center
            align="center",
        ),
    ]
)

sidebar = html.Div(
    [
        sidebar_header,
        # we wrap the horizontal rule and short blurb in a div that can be
        # hidden on a small screen
        html.Div(
            [
                html.Hr(),
                html.P(
                    "Analyzing response to COVID-19 in the United States.",
                    className="lead",
                ),
            ],
            id="blurb",
        ),
        # use the Collapse component to animate hiding / revealing links
        dbc.Collapse(
            dbc.Nav([
                dbc.NavLink(
                    page["name"],
                    href=page["path"],
                    id=page["id"],
                    # active='exact'
                    # active=True,
                )
                for page in _order(dash.page_registry.values())
            ], 
            vertical=True,
            pills=True),
            id="collapse",
        ),
    ],
    id="sidebar",
)

# content = html.Div(id="page-content")
content = html.Div(
    dash.page_container,
    id="page-content"
)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(
    Output("sidebar", "className"),
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className")],
)
def toggle_classname(n, classname):
    if n and classname == "":
        return "collapsed"
    return ""


@app.callback(
    Output("collapse", "is_open"),
    [Input("navbar-toggle", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(port=8000, debug=True, host='0.0.0.0')
