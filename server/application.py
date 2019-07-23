# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model

from sklearn.preprocessing import StandardScaler

from PIL import Image
import numpy as np
import io

from flask_cors import CORS

import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from keras.preprocessing.sequence import pad_sequences

import pickle
import string

# instantiate flask
application = app = flask.Flask(__name__)
CORS(app)

# load the model, and pass in the custom metric function
global graph_AlexNet, graph_SentimentAnalysis

graph_AlexNet = tf.Graph()
with graph_AlexNet.as_default():
    session1 = tf.Session()
    with session1.as_default():
        model_AlexNet = load_model('keras.h5')
tf.reset_default_graph()

graph_SentimentAnalysis = tf.Graph()
with graph_SentimentAnalysis.as_default():
    session2 = tf.Session()
    with session2.as_default():
        model_SentimentAnalysis = load_model('second.h5')
tf.reset_default_graph()

global scaler
scaler = StandardScaler()
mean = np.load('mean.npy')
scale = np.load('scale.npy')
var = np.load('var.npy')
scaler.mean_ = mean
scaler.scale_ = scale
scaler.var_ = var

global tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

global stopwords
with open('stop_words.pickle', 'rb') as handle:
    stopwords = pickle.load(handle)


def prepare_image(image):

    image = Image.open(io.BytesIO(image))
    image.thumbnail((227, 227))
    image_array = np.array(image)
    image_array = image_array.flatten()
    test_images = []
    test_images.append(image_array)
    image_array = scaler.transform(test_images)
    image = np.reshape(image_array, (227, 227, 3))
    return image


    # # if the image mode is not RGB, convert it
    # if image.mode != "RGB":
    #     image = image.convert("RGB")
    #
    # # resize the input image and preprocess it
    # image = image.resize(target)
    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)
    #
    # # return the processed image
    # return image

def prepare_review(lines):
    predict_lines = list()
    word_tokenizer = RegexpTokenizer(r'\w+')

    for line in lines:
        tokens = word_tokenizer.tokenize(line)

        # convert to lower case
        tokens = [w.lower() for w in tokens]

        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]

        # filter out stop words
        stop_words = set(stopwords)
        words = [w for w in words if not w in stop_words]
        predict_lines.append(words)

    max_length = 2795
    predict_sequences = tokenizer.texts_to_sequences(predict_lines)
    predict_pad = pad_sequences(predict_sequences, maxlen=max_length)
    return predict_pad

# define a predict function as an endpoint
@app.route("/first/predict", methods=["POST"], endpoint='first')
def predict():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            # image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image)
            # classify the input image and then initialize the list
            # of predictions to return to the client
            test_images_list = []
            test_images_list.append(image)
            test_images = np.array(test_images_list)

            with graph_AlexNet.as_default():
                with session1.as_default():
                    preds = model_AlexNet.predict(test_images)
            tf.reset_default_graph()
            label = np.argmax(preds[0])
            print(label)

            data["success"] = True
            data['label'] = str(label)

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# define a predict function as an endpoint
@app.route("/second/predict", methods=["POST"], endpoint='second')
def predict():
    data = {"success": False}
    # print(flask.request.form['review'])
    # print(flask.request.args)

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.form.get("review"):

            review = flask.request.form['review']

            # preprocess the image and prepare it for classification
            predict_pad = prepare_review([review])
            # classify the input image and then initialize the list
            # of predictions to return to the client

            with graph_SentimentAnalysis.as_default():
                with session2.as_default():
                    preds = model_SentimentAnalysis.predict(predict_pad)
            # label = np.argmax(preds[0])
            prob  = preds[0][0]
            print(preds)

            data["success"] = True
            data['prob'] = str(prob)

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# # define a predict function as an endpoint
# @app.route("/check", methods=["GET","POST"])
# def predict():
#     data = {"success": "Works!"}
#
#     # return a response in json format
#     return flask.jsonify(data)

# start the flask app, allow remote connections
app.run(host='0.0.0.0')