from flask import Flask, render_template, request
# import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def upload_file():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file1():
   if request.method == 'POST':
      f = request.files['file']
      filename = "static/"+secure_filename(f.filename)
      f.save(filename)
    #   os.remove(filename)
      return render_template('predict.html',fn = filename)
		
if __name__ == '__main__':
   app.run()