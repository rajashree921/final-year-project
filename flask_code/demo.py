import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import csv
import nltk
import pickle
import itertools
import cv2 as cv
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
from joblib import load, dump
import matplotlib.pylab as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from keras.preprocessing import image
from keras.backend import clear_session
from keras.models import Model, load_model, model_from_json
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import re
import string

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
work_dir='C:/Users/Rajasree/Code/Final-Year-Project/Book_project/pre_trained_models/'

Choice_matrix=pd.read_csv(work_dir + "Choice_matrix.csv", sep=";",index_col=0)
Choice_matrix_img=pd.read_csv(work_dir + 'Choice_matrix_img.csv', sep=";",index_col=0)
classe=['Arts & Photography',
 'Biographies & Memoirs',
 'Business & Money',
 'Calendars',
 "Children's Books",
 'Comics & Graphic Novels',
 'Computers & Technology',
 'Cookbooks, Food & Wine',
 'Crafts, Hobbies & Home',
 'Christian Books & Bibles',
 'Engineering & Transportation',
 'Health, Fitness & Dieting',
 'History',
 'Humor & Entertainment',
 'Law',
 'Literature & Fiction',
 'Medical Books',
 'Mystery, Thriller & Suspense',
 'Parenting & Relationships',
 'Politics & Social Sciences',
 'Reference',
 'Religion & Spirituality',
 'Romance',
 'Science & Math',
 'Science Fiction & Fantasy',
 'Self-Help',
 'Sports & Outdoors',
 'Teen & Young Adult',
 'Test Preparation',
 'Travel']             

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

def Final_textreader(filepath):
    Corpus=[]
    for i in range(len(filepath)):
        img=plt.imread(filepath[i])
        if(len(img.shape)==3):
            if(img.shape[2]==3):
                img=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
            else:
                img=cv.cvtColor(img,cv.COLOR_RGBA2GRAY)
        Text=[]
        IMG0=Image.fromarray(img)
        Text.append(pytesseract.image_to_string(IMG0).replace("\n"," ").split())
        ret,thresh1 = cv.threshold(img,200,255,cv.THRESH_BINARY_INV)
        IMG1=Image.fromarray(thresh1)
        Text.append(pytesseract.image_to_string(IMG1).replace("\n"," ").split())
        ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
        IMG2=Image.fromarray(thresh2)
        Text.append(pytesseract.image_to_string(IMG2).replace("\n"," ").split())
        if(Text==[[],[],[]]):
            ret,thresh3 = cv.threshold(img,225,255,cv.THRESH_BINARY_INV)
            IMG3=Image.fromarray(thresh3)
            Text.append(pytesseract.image_to_string(IMG3).replace("\n"," ").split())
        Text=list(itertools.chain.from_iterable(Text))
        for i in range(len(Text)):
            word = Text[i]
            word = word.upper()
            Text[i] = word.translate(word.maketrans('', '', string.punctuation + '‘’“”—' + string.whitespace + string.digits))
        Text=list(set(Text))
        regex = re.compile(r'[A-Za-z]{4,}')
        Text = [i for i in Text if regex.search(i)]
        Text1 = Text.copy()
        for word in Text1:
            if word in ['BESTSELLING','AUTHOR','YORK','TIMES']:
                Text.remove(word)
        Corpus.append(" ".join(Text))
        return Corpus

def flatten(l):
  out = []
  for item in l:
    if isinstance(item, (list, tuple)):
      out.extend(flatten(item))
    else:
      out.append(item)
  return out

def word_splitter(corpus):
    out=[]
    for line in corpus:
        splitted=line.replace("\n"," ").split()
        out.append(splitted)
    return out

def count_words(corpus):
    return len(flatten(word_splitter(corpus)))

def count_empty_lines(corpus):
    empties=0
    for line in corpus:
        if(line==""): empties+=1
    return empties

def Corpus_dropna(Corpus):
    
    Test=pd.DataFrame()
    Test["Text"]=Corpus
    Test_dropna=pd.DataFrame()
    Test_dropna["Text"]=Test.Text[Test.Text!=""]
    Test_dropna["Category"]="unknown"
    return Test_dropna

def stop_words_filtering(liste,stop_words):
        
   filtered=[]
   for word in liste:
       if word not in stop_words: filtered.append(word)
   return(filtered)

def stemming(words,stemmer):
    stems=[]
    for word in words:
        stems.append(stemmer.stem(word))
    return stems

def text_processer(list_of_sentences,stop_words=None,stemmer=None):
    
    sortie=[]
    for sentence in list_of_sentences:
        tokens=word_tokenize(sentence.lower(),language="english")
        if(stop_words!=None):
            filtered=stop_words_filtering(tokens,stop_words)
        else:
            filtered=tokens
        if(stemmer!=None):
            stemmed=stemming(filtered,stemmer)
        else:
            stemmed=filtered
        sortie.append(" ".join(stemmed))
    return sortie

def extract_features_keras(list_images, new_inception):
    ## Extract the transfer values of the last Avg_pool layer of new_inception
    ## model. Weights are kept from new inception model
    ## A new model is made with Avg_pool as the last layer. Image are processed
    ## in the CNN and features extracted

    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    model = Model(inputs=new_inception.input, outputs=new_inception.get_layer('avg_pool').output)

    for i, image_path in enumerate(list_images):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = model.predict(x)
        features[i,:]=np.squeeze(predictions)
    return features

def inception_one_image(image_path, new_inception):

    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = new_inception.predict(x)
    return predictions

def classement_predictions(predictions, classe=classe):
    pred = pd.DataFrame(np.transpose(predictions))
    maximum = pred.sort_values(by=0, ascending = False)
    max_3 = maximum.index[0:3]
    classe_3 = []
    for i in max_3:
        classe_3.append(classe[i])
    return classe_3, maximum.index[0:3]

def prediction(img,  
              classe, 
              stopwords,
              countv,
              tformer, 
              clf_Text,
              new_inception, 
              clf_SVM_new_inception,
              pca):
    image_path = list([img])
    text_img = Final_textreader(image_path)
    pred_clf_inception = inception_one_image(img, new_inception = new_inception) #new_inception prediction
    #pred_clf_inception = pre_pred_clf_inception.argmax(axis=1)

    features_img = extract_features_keras(image_path, new_inception = new_inception)
    features_img_pca = pca.transform(features_img)
    pred_clf_svm_inception = clf_SVM_new_inception.predict_proba(features_img_pca) #svm_inception prediction
    if text_img == ['']:
        print('Text Mining classifier did not find any text on the image')
        total_pred = pd.DataFrame(index=['Top 1', 'Top 2', 'Top 3'],
                            columns=['inception', 'SVM_inception'])
        total_pred_best_pred = pd.DataFrame(index=['Top 1', 'Top 2', 'Top 3'],
                            columns=['inception', 'SVM_inception'])
        total_pred.inception, total_pred_best_pred.inception = classement_predictions(pred_clf_inception)
        total_pred.SVM_inception, total_pred_best_pred.SVM_inception = classement_predictions(pred_clf_svm_inception)

    else :
        print('Some text has been found on the image: ',text_img)
        df_text_img=Corpus_dropna(text_img)
        Filtered_text_img=text_processer(df_text_img.Text,stop_words=stopwords,stemmer=None)
        Text_img_count=countv.transform(Filtered_text_img)
        text_img_to_pred=tformer.transform(Text_img_count)

        pred_clf_textmining = clf_Text.predict_proba(text_img_to_pred)#text prediction
        total_pred = pd.DataFrame(index=['Top 1', 'Top 2', 'Top 3'],
                            columns=['Text', 'inception', 'SVM_inception'])

        total_pred_best_pred = pd.DataFrame(index=['Top 1', 'Top 2', 'Top 3'],
                            columns=['Text', 'inception', 'SVM_inception'])

        total_pred.Text, total_pred_best_pred.Text = classement_predictions(pred_clf_textmining)
        total_pred.inception, total_pred_best_pred.inception = classement_predictions(pred_clf_inception)
        total_pred.SVM_inception, total_pred_best_pred.SVM_inception = classement_predictions(pred_clf_svm_inception)
    return total_pred, total_pred_best_pred

def best_pred(img,
            classe, 
            stopwords,
            countv,
            tformer, 
            clf_Text, 
            new_inception, 
            clf_SVM_new_inception, 
            pca,
            choice_matrix=Choice_matrix,
            choice_matrix_img=Choice_matrix_img):
    nope,pred=prediction(img, classe=classe, stopwords=stopwords, countv=countv, tformer=tformer, clf_Text = clf_Text, new_inception = new_inception, clf_SVM_new_inception = clf_SVM_new_inception, pca = pca)
    resultats=pd.DataFrame(columns=['Best Predictions'],
                            index=['Top 1', 'Top 2', 'Top 3'])
    if(len(pred.columns)==3):
        Top1f=choice_matrix[str(pred.Text[0])][pred.SVM_inception[0]]
        if(pred.Text[0]!=pred.SVM_inception[0]):
            if(Top1f!=classe[pred.Text[0]]):
                Top2f=classe[pred.Text[0]]
            else:
                Top2f=classe[pred.SVM_inception[0]]
            Top2Text=classe[pred.Text[1]]
            Top2SVM=classe[pred.SVM_inception[1]]
            if(Top2Text!=Top1f and Top2Text!=Top2f):
                if(Top2SVM!=Top1f and Top2SVM!=Top2f):
                    Top3f=choice_matrix[str(pred.Text[1])][pred.SVM_inception[1]]
                else:
                    Top3f=Top2Text
            else:
                if(Top2SVM!=Top1f and Top2SVM!=Top2f):
                    Top3f=Top2SVM
                else:
                    Top3f=choice_matrix[str(pred.Text[2])][pred.SVM_inception[2]]
        else:
            Top2f=choice_matrix[str(pred.Text[1])][pred.SVM_inception[1]]
            if(pred.Text[1]!=pred.SVM_inception[1]):
                if(Top2f!=classe[pred.Text[1]]):
                    Top3f=classe[pred.Text[1]]
                else:
                    Top3f=classe[pred.SVM_inception[1]]
            else:
                Top3f=choice_matrix[str(pred.Text[2])][pred.SVM_inception[2]]
    else:
        Top1f=choice_matrix_img[str(pred.SVM_inception[0])][pred.inception[0]]
        if(pred.SVM_inception[0]!=pred.inception[0]):
            if(Top1f!=classe[pred.SVM_inception[0]]):
                Top2f=classe[pred.SVM_inception[0]]
            else:
                Top2f=classe[pred.inception[0]]
            Top2Inc=classe[pred.inception[1]]
            Top2SVM=classe[pred.SVM_inception[1]]
            if(Top2SVM!=Top1f and Top2SVM!=Top2f):
                if(Top2Inc!=Top1f and Top2Inc!=Top2f):
                    Top3f=choice_matrix_img[str(pred.SVM_inception[1])][pred.inception[1]]
                else:
                    Top3f=Top2SVM
            else:
                if(Top2Inc!=Top1f and Top2Inc!=Top2f):
                    Top3f=Top2Inc
                else:
                        Top3f=choice_matrix_img[str(pred.SVM_inception[2])][pred.inception[2]]
        else:
            Top2f=choice_matrix[str(pred.SVM_inception[1])][pred.inception[1]]
            if(pred.SVM_inception[1]!=pred.inception[1]):
                if(Top2f!=classe[pred.SVM_inception[1]]):
                    Top3f=classe[pred.SVM_inception[1]]
                else:
                    Top3f=classe[pred.inception[1]]
            else:
                Top3f=choice_matrix_img[str(pred.SVM_inception[2])][pred.inception[2]]
    resultats["Best Predictions"][0]=Top1f
    resultats["Best Predictions"][1]=Top2f
    resultats["Best Predictions"][2]=Top3f
    return resultats.to_html()

app = Flask(__name__)

@app.route('/')
def upload_file():
   return render_template('index.html')



@app.route('/upload', methods = ['GET', 'POST'])
def predict():
  if request.method == 'POST':
    f = request.files['file']
    filename = "static/uploaded/"+secure_filename(f.filename)
    f.save(filename)
    clear_session()
    new_inception = load_cnn()
    pca_pre_svm, clf_SVM_new_inception = load_svm()
    stop_words, countv, tformer, clf_TextMining = load_textmodel()
    result_table = best_pred(filename,
                            classe=classe,
                            stopwords=stop_words,
                            countv=countv,
                            tformer=tformer,
                            clf_Text = clf_TextMining,
                            new_inception = new_inception, 
                            clf_SVM_new_inception = clf_SVM_new_inception,
                            pca = pca_pre_svm
                             )
    # os.remove(filename)
    return render_template('predict.html',fn = filename, result_table = result_table)

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

if __name__ == '__main__':
   app.run()