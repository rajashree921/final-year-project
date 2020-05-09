from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from bokeh.plotting import figure
from bokeh.embed import components
from numpy import pi

import keras
import numpy as np
import pandas as pd
from PIL import Image
from joblib import load
from sklearn.externals import joblib
from keras.preprocessing import image
from keras.models import Sequential, Model, load_model, model_from_json
from keras.applications.inception_v3 import InceptionV3, preprocess_input

work_dir='C:/Users/Rajasree/Code/Final-Year-Project/Book_project/pre_trained_models/'

Choice_matrix=pd.read_csv(work_dir + "Choice_matrix.csv", sep=";",index_col=0)
Choice_matrix_img=pd.read_csv(work_dir + 'Choice_matrix_img.csv', sep=";",index_col=0)

def load_cnn():
    if os.path.isfile(work_dir + 'new_inception.json') == True and \
    os.path.isfile(work_dir + 'new_inception.h5') == True:

        print("New_inception model loading...")
        json_file = open(work_dir + "new_inception.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        new_inception = model_from_json(loaded_model_json)
        new_inception.load_weights(work_dir + "new_inception.h5")
        print("New_inception model loaded")
        return new_inception
    else:
        print("New_inception model can not be loaded, please check the file name or the filepath")
        shutdown_server()

def load_svm():
    if os.path.isfile(work_dir + 'SVM_new_inception') == True and \
   os.path.isfile(work_dir + 'pca_pre_SVM') == True:
        print('SVM_new_inception model loading...')
        pca_pre_svm = load(work_dir + 'pca_pre_SVM')
        clf_SVM_new_inception = load(work_dir + 'SVM_new_inception')
        print("SVM_new_inception model loaded")
        return pca_pre_svm, clf_SVM_new_inception
    else:
        print("SVM_New_inception model can not be loaded, please check the file name or the filepath")
        shutdown_server()

def load_textmodel():
    if os.path.isfile(work_dir + 'clf_textmining') == True and \
   os.path.isfile(work_dir + 'stopwords') == True and \
   os.path.isfile(work_dir + 'countvectorizer') == True and \
   os.path.isfile(work_dir + 'tfidf_transformer') == True:
        print('Text Mining model loading...')

        stop_words=load(work_dir + "stopwords")
        countv=load(work_dir + "countvectorizer")
        tformer=joblib.load(work_dir + "tfidf_transformer")
        clf_TextMining=joblib.load(work_dir + "clf_textmining")

        print("Text Mining model loaded")
        return stop_words, countv, tformer, clf_TextMining
    else:
        print("Text Mining model can not be loaded, please check the file name or the filepath")
        shutdown_server()        

app = Flask(__name__)

@app.route('/')
def upload_file():
   return render_template('index.html')
	
@app.route('/upload', methods = ['GET', 'POST'])
def upload_file1():
   if request.method == 'POST':
        f = request.files['file']
        filename = "static/uploaded/"+secure_filename(f.filename)
        f.save(filename)
        # new_inception = load_cnn()
        # pca_pre_svm, clf_SVM_new_inception = load_svm()
        # stop_words, countv, tformer, clf_TextMining = load_textmodel()

        # img = image.load_img(filename, target_size=(299, 299))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        # # print(type(x),flush=True)
        # predictions = new_inception.predict(x)

        # os.remove(filename)
        return render_template('predict.html',fn = filename)

from flask import request
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'  

IMAGE_LABELS = [ 'Shoes', 'Watch', 'Laptop']

def generate_barplot(predictions):
    """ Generates script and `div` element of bar plot of predictions using
    Bokeh
    """
    plot = figure(y_range=IMAGE_LABELS, plot_height=100)
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.hbar(y=IMAGE_LABELS, right=predictions, height=0.6)
    plot.yaxis.major_label_orientation = pi / 2.
    plot.sizing_mode = 'scale_width'
    return components(plot)

@app.route('/predict1')
def predict():
    predictions = [0.1, 0.5, 0.3]
    script, div = generate_barplot(predictions)
    return render_template(
        'predict1.html',
        plot_script=script,
        plot_div=div,
    )        

if __name__ == '__main__':
   app.run()