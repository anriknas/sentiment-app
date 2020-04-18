import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import model


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
    html.H1('Sentiment App',
    style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    html.Div('''
        Sentiment App: Enter your text to identify sentiment.
    ''',
    style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Input(
        id="input-id",
        type="text",
        placeholder="Enter your text",
        value ="Wow"
    ),
    html.Div(id='my-sent'),
    html.Div(id='my-conf'),
    html.Div(id='my-tag')
]
)


@app.callback(
    [Output(component_id='my-sent', component_property='children'),
    Output(component_id='my-conf', component_property='children'),
    Output(component_id='my-tag', component_property='children')
    ],
    [Input(component_id='input-id', component_property='value')]
)
def update_output_div(input_value):
    tag, tag_prob, sent_prob = model.predict_response([input_value])
    return tag, tag_prob, sent_prob


if __name__ == "__main__":
    app.run_server(debug=True)