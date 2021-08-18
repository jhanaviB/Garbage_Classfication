
import numpy as np

from flask import Flask,render_template,url_for,request

import tensorflow as tf 
import tensorflow.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import  os

model = tensorflow.keras.models.load_model("./web_apps/static/trained_seq_model.h5")

# model = tensorflow.keras.models.load_model("./web_apps/static/inceptionv3_model.h5")


static = os.path.join('web_apps', 'static')
template = os.path.join('web_apps', 'templates')
UPLOAD_FOLDER =  os.path.join('web_apps','uploads')

app = Flask(__name__,static_folder=static, template_folder=template)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
answer=""
error=""

@app.route("/",methods=['GET'])
def home():    
    return render_template('index.html' )

@app.route("/predict", methods=['GET', 'POST'])
def GetClass():
    if request.method=="POST":
        try:
            file= request.files["myfile"]
            if file:
                #upload
                file_path  = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path )
                print(file.filename)
                result,error=process(file_path)
                if result=="":
                    error="Sorry!"
        except(SyntaxError) as e:
            error ="Could not understand"
            print("Error:" + str(e))                    
        try:
                    if result!="Sorry!":
                        answer="Classified as: " + result
        except Exception as e:
                print(e)
        return render_template('index.html', file=file,
                            answer=answer, error=error)
    else :
        return render_template('index.html', 
                            answer="", error="")

def process(img_path):
    labels={0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    img = image.load_img(img_path, target_size=(256, 256))
    p=[]
    img = np.expand_dims(img, axis=0)
    p=model.predict(img)
    print("p.shape:",p.shape)
    print("p:",p)
    o=np.argmax(p, axis=-1)
    print("o: ",o)
    predicted_class = labels[o[0]]
    os.remove(img_path)
    print("classified label:",predicted_class)
    error=""
    return(str(predicted_class),error)


if __name__ == "__main__":
    app.run(debug=True)


