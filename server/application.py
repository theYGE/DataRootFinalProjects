from flask import Flask
import flask
import tensorflow as tf
from keras.models import load_model

from sklearn.preprocessing import StandardScaler

import numpy as np
import io

from flask_cors import CORS

from keras.preprocessing.sequence import pad_sequences

import pickle
import string

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from pytorch_transformers import *
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import torch
import tokenize_uk
import os
import torch
from pytorch_transformers import *
from pytorch_transformers.tokenization_bert import BertTokenizer
from random import random, randrange, randint, shuffle, choice

from inference_MaskedLM import get_masked_texts
from inference_NextSentence import get_generated_story

# instantiate flask
app = Flask(__name__)
CORS(app)


# define a predict function as an endpoint
@app.route("/second/predict", methods=["POST"], endpoint='first')
def predict():
    data = {"success": False}
    print(flask.request.form['first_story'])
    # print(flask.request.args)

    if flask.request.method == "POST":
        if flask.request.form.get("first_story") and flask.request.form.get("second_story") and flask.request.form.get("third_story"):


            first_story = flask.request.form['first_story']
            second_story = flask.request.form['second_story']
            third_story = flask.request.form['third_story']

            cleaned_texts = get_masked_texts(first_story, second_story, third_story)
            story = get_generated_story(cleaned_texts['trained'])

            result = {
                'trained': cleaned_texts['trained'],
                'basic': cleaned_texts['basic'],
                'story': story
            }


    # return the data dictionary as a JSON response
    return flask.jsonify(result)

@app.route("/trial", methods=["POST"], endpoint='second')
def justToMakeSure():

    return flask.jsonify("It's alive")

# # define a predict function as an endpoint
# @app.route("/check", methods=["GET","POST"])
# def predict():
#     data = {"success": "Works!"}
#
#     # return a response in json format
#     return flask.jsonify(data)

# start the flask app, allow remote connections
if __name__ == "__main__":
    app.run()