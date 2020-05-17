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
from joblib import load
import matplotlib.pylab as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from keras.preprocessing import image
from keras.models import Model,load_model,model_from_json
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import re,string,time
import tensorflow as tf

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
work_dir = 'static/models/'

# Choice_matrix=pd.read_csv(work_dir + "Choice_matrix.csv", sep=";",index_col=0)
# Choice_matrix_img=pd.read_csv(work_dir + 'Choice_matrix_img.csv', sep=";",index_col=0)
classe = ['Arts & Photography',
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

new_inception,\
feature_model,\
pca,\
clf_SVM_new_inception,\
stopwords,\
countv,\
tformer,\
clf_Text,\
choice_matrix,\
choice_matrix_img,\
graph = (None,)*11

def load_data():
  global new_inception, feature_model, pca, clf_SVM_new_inception, stopwords
  global countv, tformer, clf_Text, choice_matrix, choice_matrix_img, graph
  print("New_inception model loading...")
  json_file = open(work_dir + "new_inception.json", 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  new_inception = model_from_json(loaded_model_json)
  new_inception.load_weights(work_dir + "new_inception.h5")
  print("New_inception model loaded")
  feature_model = Model(inputs=new_inception.input, 
                        outputs=new_inception.get_layer('avg_pool').output)
  graph = tf.get_default_graph()
  print('SVM_new_inception model loading...')
  pca = load(work_dir + 'pca_pre_SVM')
  clf_SVM_new_inception = load(work_dir + 'SVM_new_inception')
  print("SVM_new_inception model loaded")
  print('Text Mining model loading...')
  stopwords = load(work_dir + "stopwords")
  countv = load(work_dir + "countvectorizer")
  tformer = joblib.load(work_dir + "tfidf_transformer")
  clf_Text = joblib.load(work_dir + "clf_textmining")
  print("Text Mining model loaded")
  choice_matrix = pd.read_csv(work_dir + "Choice_matrix.csv", sep=";",index_col=0)
  choice_matrix_img = pd.read_csv(work_dir + 'Choice_matrix_img.csv', sep=";",index_col=0)

def Final_textreader(img):
  Corpus = []
  if len(img.shape) == 3:
    if img.shape[2] == 3:
      img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    else:
      img = cv.cvtColor(img,cv.COLOR_RGBA2GRAY)
  Text=[]
  config = ("-l eng --oem 1 --psm 6")
  IMG0=Image.fromarray(img)
  Text.append(pytesseract.image_to_string(IMG0, config=config).replace("\n"," ").split())
  ret,thresh1 = cv.threshold(img,200,255,cv.THRESH_BINARY_INV)
  IMG1=Image.fromarray(thresh1)
  Text.append(pytesseract.image_to_string(IMG1, config=config).replace("\n"," ").split())
  ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
  IMG2=Image.fromarray(thresh2)
  Text.append(pytesseract.image_to_string(IMG2, config=config).replace("\n"," ").split())
  if(Text == [[],[],[]]):
    ret,thresh3 = cv.threshold(img,225,255,cv.THRESH_BINARY_INV)
    IMG3=Image.fromarray(thresh3)
    Text.append(pytesseract.image_to_string(IMG3, config=config).replace("\n"," ").split())
  Text=list(set(itertools.chain.from_iterable(Text)))
  # print("Text_1: ",Text)
  # for i in range(len(Text)):
  #   word = Text[i]
  #   word = word.upper()
  #   Text[i] = word.translate(word.maketrans('', '', string.punctuation + '‘’“”—' + string.whitespace + string.digits))
  # Text=list(set(Text))
  regex = re.compile(r'[A-Za-z]{4,}')
  # strip_special_chars = re.compile(r"([^A-Z ]+)|(\b[A-Z]{,3}\b)")
  # Text = [re.sub(strip_special_chars, "", i.upper()) for i in Text]
  Text = [i.translate(i.maketrans('', '', string.punctuation + '‘’“”—|' + string.whitespace + string.digits)).upper() for i in Text if regex.search(i)]
  # print("Text_2: ",Text)
  # Text = [i for i in Text if regex.search(i.upper())]
  # Text1 = Text.copy()
  # for word in Text1:
  #   if word in :
  #     Text.remove(word)
  discard = ['BESTSELLING','BESTSELLER','AUTHOR','YORK','TIMES']
  dis = []
  for i, word in enumerate(Text):
    if word in discard:
      dis.append(word)
  for word in dis:
    Text.remove(word)
  # print("Text_3: ", Text)
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
    if(line == ""): empties+=1
  return empties

def Corpus_dropna(Corpus):    
  Test=pd.DataFrame()
  Test["Text"]=Corpus
  Test_dropna=pd.DataFrame()
  Test_dropna["Text"]=Test.Text[Test.Text!=""]
  # Test_dropna["Category"]="unknown"
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

def extract_features_keras(x):
    ## Extract the transfer values of the last Avg_pool layer of new_inception
    ## model. Weights are kept from new inception model
    ## A new model is made with Avg_pool as the last layer. Image are processed
    ## in the CNN and features extracted

  nb_features = 2048
  features = np.empty((1,nb_features))
  
  global graph, feature_model
  with graph.as_default():
    #   for i, image_path in enumerate(list_images):
    #       img = image.load_img(image_path, target_size=(299, 299))
    # x = image.img_to_array(img_load)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    predictions = feature_model.predict(x)
    features[0,:]=np.squeeze(predictions)
  return features

def inception_one_image(x):
#   img = image.load_img(image_path, target_size=(299, 299))
  # x = image.img_to_array(img_load)
  # x = np.expand_dims(x, axis=0)
  # x = preprocess_input(x)
  global graph, new_inception
  with graph.as_default():
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

def predict_genre(filename):
  img = plt.imread(filename)
  text_img = Final_textreader(img)
  img_load = image.load_img(filename, target_size=(299, 299))     
  x = image.img_to_array(img_load)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  start = time.time()
  pred_clf_inception = inception_one_image(x) #new_inception prediction
  print("Inception done, {}".format(time.time()-start))
  #pred_clf_inception = pre_pred_clf_inception.argmax(axis=1)
  text_detected = False
  features_img = extract_features_keras(x)
  features_img_pca = pca.transform(features_img)
  pred_clf_svm_inception = clf_SVM_new_inception.predict_proba(features_img_pca) #svm_inception prediction
  print("SVM done, {}".format(time.time()-start))
  print(text_img)
  if text_img  ==  ['']:
  #   print('Text Mining classifier did not find any text on the image')
    total_pred = pd.DataFrame(index=['Top 1', 'Top 2', 'Top 3'],
                        columns=['inception', 'SVM_inception'])
    total_pred_best_pred = pd.DataFrame(index=['Top 1', 'Top 2', 'Top 3'],
                        columns=['inception', 'SVM_inception'])
    total_pred.inception, total_pred_best_pred.inception = classement_predictions(pred_clf_inception)
    total_pred.SVM_inception, total_pred_best_pred.SVM_inception = classement_predictions(pred_clf_svm_inception)

  else :
  #   print('Some text has been found on the image: ',text_img)
    text_detected = " ".join(text_img)
    df_text_img=Corpus_dropna(text_img)
    Filtered_text_img=text_processer(df_text_img.Text,stop_words=stopwords,stemmer=None)
    Text_img_count=countv.transform(Filtered_text_img)
    text_img_to_pred=tformer.transform(Text_img_count)

    pred_clf_textmining = clf_Text.predict_proba(text_img_to_pred)#text prediction
    print("Text done, {}".format(time.time()-start))

    total_pred = pd.DataFrame(index=['Top 1', 'Top 2', 'Top 3'],
                        columns=['Text', 'inception', 'SVM_inception'])

    total_pred_best_pred = pd.DataFrame(index=['Top 1', 'Top 2', 'Top 3'],
                        columns=['Text', 'inception', 'SVM_inception'])

    total_pred.Text, total_pred_best_pred.Text = classement_predictions(pred_clf_textmining)
    total_pred.inception, total_pred_best_pred.inception = classement_predictions(pred_clf_inception)
    total_pred.SVM_inception, total_pred_best_pred.SVM_inception = classement_predictions(pred_clf_svm_inception)
  pred = total_pred_best_pred
  resultats=pd.DataFrame(columns=['Best Predictions'],
                          index=['Top 1', 'Top 2', 'Top 3'])
  if(len(pred.columns) == 3):
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
  end = time.time()
  print("Time taken: ",(end-start))
  return resultats.to_html(justify='left', border = 0), text_detected

app = Flask(__name__)

@app.route('/')
def home_page():
  return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
  if request.method  ==  'POST' and request.files['file']:
    f = request.files['file']
    filename = "static/images/uploaded/"+secure_filename(f.filename)
    f.save(filename)
    try:
      # clear_session()
      # image = filename
      # image_path = list([filename])
      table, text_detected = predict_genre(filename)
      # os.remove(filename)
      return render_template('predict.html',img = filename, result_table = table, text = text_detected)
    except :
      return render_template('error.html')
    
    # return render_template('predict.html',fn = filename, result_table = '<p>Blank</p>')

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

if __name__  ==  '__main__':
    load_data()
    app.run()