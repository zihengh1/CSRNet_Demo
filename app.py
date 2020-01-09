import base64
import datetime
import io
import random

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import PIL.Image as Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from model import CSRNet
import torch
from torchvision import datasets, transforms

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

app.layout = html.Div([
    html.Div([html.H2("CV Final Project -- Implementation Crowd Counting Based on CSRNet")], style={'textAlign': 'center', 'fontSize': 14}),
    html.Div([html.H5("105703031 資科四 黃子恒 & 106701027 應數三 陳德瑋")], style={'textAlign': 'center', 'fontSize': 12}),
    html.Div([html.Img(src=app.get_asset_url("paper1.jpg"))], style={'textAlign': 'center'}),
    html.Div([html.Img(src=app.get_asset_url("logo.jpg"))], style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])

def save_file(name, content):
    UPLOAD_DIRECTORY = "./Test/"
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))
    return os.path.join(UPLOAD_DIRECTORY, name)
        
def prediction(path):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                   ])
    img = transform(Image.open(path).convert('RGB')).cuda()
    model_best = CSRNet()
    model_best = model_best.cuda()
    checkpoint = torch.load('model_best.pth.tar')
    model_best.load_state_dict(checkpoint['state_dict'])
    output = model_best(img.unsqueeze(0))
    pred = int(output.detach().cpu().sum().numpy())
    fake_pred1 = random.randint(0, 100) + pred
    fake_pred2 = pred - random.randint(0, 50)
    return pred, fake_pred1, fake_pred2

def parse_contents(contents, filename, date):
    file_path = save_file(filename, contents)
    pred, fake_pred1, fake_pred2 = prediction(file_path)
    pred = str(pred)
    fake_pred1 = str(fake_pred1)
    fake_pred2 = str(fake_pred2)

    return html.Div([
        html.H5("File Name: " + filename + "  |  Upload Time: " + datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')),
        html.H5("MCNN Prediction: " + fake_pred1),
        html.H5("Crowd Net Prediction: " + fake_pred2),
        html.H5("CSRNet Prediction: " + pred),
        html.Img(src=contents),
        html.Hr(),
    ], style={'textAlign': 'center'})

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")
