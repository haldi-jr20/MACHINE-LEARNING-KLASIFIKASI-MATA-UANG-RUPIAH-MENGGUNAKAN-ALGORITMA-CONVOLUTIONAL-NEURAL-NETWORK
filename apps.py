
from flask import Flask, render_template, request, jsonify
from keras.models import load_model

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_applications.mobilenet_v2 import preprocess_input
from keras_applications.mobilenet_v2 import decode_predictions
from keras_applications.mobilenet_v2 import MobileNetV2

import tensorflow as tf
from tensorflow import keras
from skimage import transform, io
import numpy as np
import os
from PIL import Image
from datetime import datetime
from keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)

# load model for prediction
modelcnn = load_model("MobileNetV2_uang_89.52.h5")
modelvgg = load_model("VGG16_uang87.61.h5")
modelxception = load_model("Xception_uang_85.71.h5")
modelnasnet = load_model("NASNetMobile_uang_84.76.h5")


UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("cnn.html")

@app.route("/classification", methods = ['GET', 'POST'])
def classification():
	return render_template("classifications.html")

@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('file')
    filename = "temp_image.png"
    namafile = "rgb.png"
    errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors["message"] = 'File type of {} is not allowed'.format(file.filename)

    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp
    img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    im_url = os.path.join(app.config['UPLOAD_FOLDER'], namafile)
    
    # convert image to RGB
    im = np.array(Image.open(img_url))

    im_R = im.copy()
    im_R[:, :, (1,2)] = 0

    im_G = im.copy()
    im_G[:,:,(0,2)] = 0

    im_B = im.copy()
    im_B[:,:,(0,1)] = 0

    im_RGB = np.concatenate((im_R,im_G,im_B), axis=1)
    pil_img = Image.fromarray(im_RGB)
    pil_img.save(im_url)


    # prepare image for prediction
    pil_img = image.load_img(img_url, target_size=(224, 224, 3))
    x = image.img_to_array(pil_img)
    x = x/127.5-1
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # predict
    prediction_array_nasnet = modelnasnet.predict(images)
    prediction_array_vgg = modelvgg.predict(images)
    prediction_array_xception = modelxception.predict(images)
    prediction_array_cnn = modelcnn.predict(images)

    # prepare api response
    class_names = ['Rp 100.000','Rp 10.000','Rp 1000','Rp 20.000','Rp 2000','Rp 50.000','Rp 5000']
	
    return render_template("classifications.html", img_path = img_url, 
                        predictionnasnet = class_names[np.argmax(prediction_array_nasnet)],
                        confidencenasnet = '{:2.0f}%'.format(100 * np.max(prediction_array_nasnet)),
                        predictionvgg = class_names[np.argmax(prediction_array_vgg)],
                        confidencvgg = '{:2.0f}%'.format(100 * np.max(prediction_array_vgg)),
                        predictionxception = class_names[np.argmax(prediction_array_xception)],
                        confidenceexception = '{:2.0f}%'.format(100 * np.max(prediction_array_xception)),
                        predictioncnn = class_names[np.argmax(prediction_array_cnn)],
                        confidencecnn = '{:2.0f}%'.format(100 * np.max(prediction_array_cnn))  
                        )

if __name__ =='__main__':
	app.run(debug = True)