import pandas as pd

import plotly.express as px
import plotly.graph_objects as go # or plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html

# === APP ===
mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
external_scripts = [
    mathjax
]
app = dash.Dash(__name__, external_scripts=external_scripts)

# === Figures ===
def general_plot(filename, title, value_names, xy_labels):
    df = pd.read_csv(filename, index_col=0)
    fig = px.line(df, 
        x=df.index, y=df.columns,
        labels={"value": value_names[0], "variable": value_names[1]},
        title=title)
    fig.update_layout(
        title_x = 0.5,
        title_y = 0.85,
        xaxis_title=xy_labels[0],
        yaxis_title=xy_labels[1],
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="black",
        ),
    )
    return fig

# Age
def ages_fig():
    return general_plot(
        filename = "log/ages.csv",
        title = r'Deathes in each age group',
        value_names = ["Ages", "Age groups"],
        xy_labels = ["Days", "Infections"])

# County
def countys_fig():
    return general_plot(
        filename = "log/county.csv",
        title = r'Deathes in each county',
        value_names = ["Conty", "County"],
        xy_labels = ["Days", "Infections"])

# === Helper functions ===
def create_dropdown_options(series):
    options = [{'label': i, 'value': i} for i in series]
    return options
def create_dropdown_value(series):
    value = series
    return value
def create_slider_marks(values):
    marks = {i: {'label': str(i)} for i in values}
    return marks

# === Layout ===
app.layout = html.Div(children=[
    # Left: control panel
    html.Div(children = [
        html.H1(children='Simulation dashboard'),
        html.P('Rényi metapop research group'),
        html.Img(src="assets/covid19.png"),
        html.Label("Simulations", className='dropdown-labels'),
        dcc.Dropdown(multi=True, className='dropdown', id='class-dropdown',
                     options=create_dropdown_options(['Sim1', 'Sim2']),
                     value=create_dropdown_value(['Sim1', 'Sim2'])),
        html.Button("Update", id="update-button"),
    ], id='left-container'),
    # Right: plots
    html.Div(children=[
        html.Div(children=[
            dcc.Graph(figure=ages_fig()),
            dcc.Graph(figure=countys_fig()),
        ], id="visualization")
    ], id="right-container")
], id='container')
app.run_server(debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter