import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

class_names = ['Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy']

model =tf.keras.models.load_model("corn_model")

def test_on_image(img_path):
    test_image = image.load_img(img_path, target_size=(240,240))
    test_image = image.img_to_array(test_image)
    test_image = test_image.reshape(240, 240, 3)
    test_image = np.expand_dims(test_image, axis=0) 
    result = model.predict(test_image)
    p = max(result.tolist())
    return(class_names[p.index(max(p))])



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = test_on_image(file_path)
        
        return preds
    return None


if __name__ == '__main__':
    http_server = WSGIServer(('', 8000), app)
    http_server.serve_forever()
    app.run()
