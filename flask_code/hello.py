from flask import Flask, render_template, request
# import cv2
import os
from werkzeug.utils import secure_filename
# from keras.preprocessing import image
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
# import numpy as np
# import PIL
from bokeh.plotting import figure
from bokeh.embed import components
from numpy import pi

app = Flask(__name__)

@app.route('/')
def upload_file():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file1():
   if request.method == 'POST':
      f = request.files['file']
      filename = "static/uploaded/"+secure_filename(f.filename)
      f.save(filename)
      # os.remove(filename)
      # img = image.load_img(filename, target_size=(299, 299))
      # x = image.img_to_array(img)
      # x = np.expand_dims(x, axis=0)
      # x = preprocess_input(x)
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
   app.run(debug=True)