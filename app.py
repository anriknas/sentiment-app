import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import model


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(
    __name__, 
    #external_stylesheets=external_stylesheets,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
    )
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div(
    [
        html.Div(
            [html.Img(src=app.get_asset_url("dash-logo.png"))], className="app__banner"
    ),
    html.Div(
        [
        html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "Sentiment classifier",
                            className="uppercase title",
                        ),
                        html.Span(
                            "This is a multi-class text-classification model that can classify your into the following tags:", style={'padding': '10px'}
                        ),
                        html.Br(),                        
                        html.Span(
                            "Tag confidence is the confidence for each tag", style={'padding': '10px'}
                        ),
                        html.Br(),                        
                        html.Span(
                            "Sentiment confidence is the confidence of the sentiment: Positive/Neutral/Negative", style={'padding': '10px'}
                        ),  
                        html.Br(),
                        html.Br(),                        
                        html.Span("Very Negative", style={'color': '#e01b1b', 'border-radius': '5px', 'border-width': '2px', 'border-style': 'solid', 'border-color': '#b8c2d4', 'padding': '2px','margin-right': '5px'} ),
                        html.Span("Negative",      style={'color': '#db3a2c', 'border-radius': '5px', 'border-width': '2px', 'border-style': 'solid', 'border-color': '#b8c2d4', 'padding': '2px', 'margin-right': '5px'} ),
                        html.Span("Neutral",       style={'color': '#ebcd0e', 'border-radius': '5px', 'border-width': '2px', 'border-style': 'solid', 'border-color': '#b8c2d4', 'padding': '2px', 'margin-right': '5px'} ),
                        html.Span("Positive",      style={'color': '#83db1f', 'border-radius': '5px', 'border-width': '2px', 'border-style': 'solid', 'border-color': '#b8c2d4', 'padding': '2px', 'margin-right': '5px'} ),
                        html.Span("Very Positive", style={'color': '#4ce80e', 'border-radius': '5px', 'border-width': '2px', 'border-style': 'solid', 'border-color': '#b8c2d4', 'padding': '2px'} ),                                                                               
                    ], style = {'background':'#ffffff'}
                ),
            ],
        className="app__header",
    ),
    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3('Enter your text here:'),
                            dcc.Textarea(
                                id="input-id",
                                value ="Wow, what a fabulous story!",
                                style={'height': '300px', 'width': '100%'}
                            )
                        ], className ='six columns'
                    ),
                    
                    html.Div(
                        [
                            html.H3('Tag'),
                            html.Div(
                                id='my-sent', 
                                className="row"
                                ),
                            html.Div(
                                id='my-conf', 
                                className="row"
                                ),
                            html.Div(
                                id='my-tag', 
                                className="row"
                                ),
                        ], className='six columns'
                    ),                    
                ],
            )
        ], className="container card app__content bg-white",
    )
    ], className="app__container",
)
],style = {'background':'#f5f7fa'}
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
    return f"Tag: {tag}", f"Tag confidence: {tag_prob*100:.1f}%", f"Sentiment Confidence: {sent_prob*100:.1f}%"



if __name__ == "__main__":
    app.run_server(debug=True)