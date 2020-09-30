import os

from flask import Flask,request,render_template
from tensorflow.python.keras.applications.densenet import preprocess_input
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

model_path = 'model_resnet.h5'

model = load_model(model_path)


def model_predict(img_path,model):
    img = image.load_img(img_path,target_size = (224,224))
    x = image.img_to_array(img)
    x = x/225
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    preds = np.argmax(preds,axis=1)
    if preds==0:
        preds = 'The car is Audi'
    elif preds==1:
        preds = 'The car is Lamborgini'
    else:
        preds = 'Th car is Mercedes'
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path,model)
        result = preds
        return result
    return None

app.run(debug=True)

