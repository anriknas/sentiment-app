import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')

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
    )    

]
)


if __name__ == "__main__":
    app.run_server(debug=True)