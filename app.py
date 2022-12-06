# coding=utf8
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename, redirect
from tensorflow import keras
from keras.preprocessing import image
from flask import send_from_directory
from keras.utils import img_to_array
import cv2
import math
import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import sys
# import os
import base64
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
# from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


unicode_mapping_rows = [
    [ '000', u'\u0B85' ],
    [ '001', u'\u0B86' ],
    [ '002', u'\u0B87' ],
    [ '003', u'\u0B88' ],
    [ '004', u'\u0B89' ],
    [ '005', u'\u0B8A' ],
    [ '006', u'\u0B8E' ],
    [ '007', u'\u0B8F' ],
    [ '008', u'\u0B90' ],
    [ '009', u'\u0B92' ],
    [ '010', u'\u0B93' ],
    [ '011', u'\u0B83' ],
    [ '012', u'\u0B95' ],
    [ '013', u'\u0B99' ],
    [ '014', u'\u0B9A' ],
    [ '015', u'\u0B9E' ],
    [ '016', u'\u0B9F' ],
    [ '017', u'\u0BA3' ],
    [ '018', u'\u0BA4' ],
    [ '019', u'\u0BA8' ],
    [ '020', u'\u0BAA' ],
    [ '021', u'\u0BAE' ],
    [ '022', u'\u0BAF' ],
    [ '023', u'\u0BB0' ],
    [ '024', u'\u0BB2' ],
    [ '025', u'\u0BB5' ],
    [ '026', u'\u0BB4' ],
    [ '027', u'\u0BB3' ],
    [ '028', u'\u0BB1' ],
    [ '029', u'\u0BA9' ],
    [ '030', u'\u0BB8' ],
    [ '031', u'\u0BB7' ],
    [ '032', u'\u0B9C' ],
    [ '033', u'\u0BB9' ],
    [ '034', u'\u0B95\u0BCD\u0BB7' ],
    [ '035', u'\u0B95\u0BBF' ],
    [ '036', u'\u0B99\u0BBF' ],
    [ '037', u'\u0B9A\u0BBF' ],
    [ '038', u'\u0B9E\u0BBF' ],
    [ '039', u'\u0B9F\u0BBF' ],
    [ '040', u'\u0BA3\u0BBF' ],
    [ '041', u'\u0BA4\u0BBF' ],
    [ '042', u'\u0BA8\u0BBF' ],
    [ '043', u'\u0BAA\u0BBF' ],
    [ '044', u'\u0BAE\u0BBF' ],
    [ '045', u'\u0BAF\u0BBF' ],
    [ '046', u'\u0BB0\u0BBF' ],
    [ '047', u'\u0BB2\u0BBF' ],
    [ '048', u'\u0BB5\u0BBF' ],
    [ '049', u'\u0BB4\u0BBF' ],
    [ '050', u'\u0BB3\u0BBF' ],
    [ '051', u'\u0BB1\u0BBF' ],
    [ '052', u'\u0BA9\u0BBF' ],
    [ '053', u'\u0BB8\u0BBF' ],
    [ '054', u'\u0BB7\u0BBF' ],
    [ '055', u'\u0B9C\u0BBF' ],
    [ '056', u'\u0BB9\u0BBF' ],
    [ '057', u'\u0B95\u0BCD\u0BB7\u0BBF' ],
    [ '058', u'\u0B95\u0BC0' ],
    [ '059', u'\u0B99\u0BC0' ],
    [ '060', u'\u0B9A\u0BC0' ],
    [ '061', u'\u0B9E\u0BC0' ],
    [ '062', u'\u0B9F\u0BC0' ],
    [ '063', u'\u0BA3\u0BC0' ],
    [ '064', u'\u0BA4\u0BC0' ],
    [ '065', u'\u0BA8\u0BC0' ],
    [ '066', u'\u0BAA\u0BC0' ],
    [ '067', u'\u0BAE\u0BC0' ],
    [ '068', u'\u0BAF\u0BC0' ],
    [ '069', u'\u0BB0\u0BC0' ],
    [ '070', u'\u0BB2\u0BC0' ],
    [ '071', u'\u0BB5\u0BC0' ],
    [ '072', u'\u0BB4\u0BC0' ],
    [ '073', u'\u0BB3\u0BC0' ],
    [ '074', u'\u0BB1\u0BC0' ],
    [ '075', u'\u0BA9\u0BC0' ],
    [ '076', u'\u0BB8\u0BC0' ],
    [ '077', u'\u0BB7\u0BC0' ],
    [ '078', u'\u0B9C\u0BC0' ],
    [ '079', u'\u0BB9\u0BC0' ],
    [ '080', u'\u0B95\u0BCD\u0BB7\u0BC0' ],
    [ '081', u'\u0B95\u0BC1' ],
    [ '082', u'\u0B99\u0BC1' ],
    [ '083', u'\u0B9A\u0BC1' ],
    [ '084', u'\u0B9E\u0BC1' ],
    [ '085', u'\u0B9F\u0BC1' ],
    [ '086', u'\u0BA3\u0BC1' ],
    [ '087', u'\u0BA4\u0BC1' ],
    [ '088', u'\u0BA8\u0BC1' ],
    [ '089', u'\u0BAA\u0BC1' ],
    [ '090', u'\u0BAE\u0BC1' ],
    [ '091', u'\u0BAF\u0BC1' ],
    [ '092', u'\u0BB0\u0BC1' ],
    [ '093', u'\u0BB2\u0BC1' ],
    [ '094', u'\u0BB5\u0BC1' ],
    [ '095', u'\u0BB4\u0BC1' ],
    [ '096', u'\u0BB3\u0BC1' ],
    [ '097', u'\u0BB1\u0BC1' ],
    [ '098', u'\u0BA9\u0BC1' ],
    [ '099', u'\u0B95\u0BC2' ],
    [ '100', u'\u0B99\u0BC2' ],
    [ '101', u'\u0B9A\u0BC2' ],
    [ '102', u'\u0B9E\u0BC2' ],
    [ '103', u'\u0B9F\u0BC2' ],
    [ '104', u'\u0BA3\u0BC2' ],
    [ '105', u'\u0BA4\u0BC2' ],
    [ '106', u'\u0BA8\u0BC2' ],
    [ '107', u'\u0BAA\u0BC2' ],
    [ '108', u'\u0BAE\u0BC2' ],
    [ '109', u'\u0BAF\u0BC2' ],
    [ '110', u'\u0BB0\u0BC2' ],
    [ '111', u'\u0BB2\u0BC2' ],
    [ '112', u'\u0BB5\u0BC2 ' ],
    [ '113', u'\u0BB4\u0BC2 ' ],
    [ '114', u'\u0BB3\u0BC2 ' ],
    [ '115', u'\u0BB1\u0BC2 ' ],
    [ '116', u'\u0BA9\u0BC2' ],
    [ '117', u'\u0BBE' ],
    [ '118', u'\u0BC6' ],
    [ '119', u'\u0BC7' ],
    [ '120', u'\u0BC8' ],
    [ '121', u'\u0BB8\u0BCD\u0BB0\u0BC0' ],
    [ '122', u'\u0BB8\u0BC1' ],
    [ '123', u'\u0BB7\u0BC1' ],
    [ '124', u'\u0B9C\u0BC1' ],
    [ '125', u'\u0BB9\u0BC1' ],
    [ '126', u'\u0B95\u0BCD\u0BB7\u0BC1' ],
    [ '127', u'\u0BB8\u0BC2' ],
    [ '128', u'\u0BB7\u0BC2' ],
    [ '129', u'\u0B9C\u0BC2' ],
    [ '130', u'\u0BB9\u0BC2' ],
    [ '131', u'\u0B95\u0BCD\u0BB7\u0BC2 ' ],
    [ '132', u'\u0B95\u0BCD' ],
    [ '133', u'\u0B99\u0BCD' ],
    [ '134', u'\u0B9A\u0BCD' ],
    [ '135', u'\u0B9E\u0BCD' ],
    [ '136', u'\u0B9F\u0BCD' ],
    [ '137', u'\u0BA3\u0BCD' ],
    [ '138', u'\u0BA4\u0BCD' ],
    [ '139', u'\u0BA8\u0BCD' ],
    [ '140', u'\u0BAA\u0BCD' ],
    [ '141', u'\u0BAE\u0BCD' ],
    [ '142', u'\u0BAF\u0BCD' ],
    [ '143', u'\u0BB0\u0BCD' ],
    [ '144', u'\u0BB2\u0BCD' ],
    [ '145', u'\u0BB5\u0BCD' ],
    [ '146', u'\u0BB4\u0BCD' ],
    [ '147', u'\u0BB3\u0BCD' ],
    [ '148', u'\u0BB1\u0BCD' ],
    [ '149', u'\u0BA9\u0BCD' ],
    [ '150', u'\u0BB8\u0BCD' ],
    [ '151', u'\u0BB7\u0BCD' ],
    [ '152', u'\u0B9C\u0BCD' ],
    [ '153', u'\u0BB9\u0BCD' ],
    [ '154', u'\u0B95\u0BCD\u0BB7\u0BCD' ],
    [ '155', u'\u0B94' ]
]

unicode_mapping_df = pd.DataFrame(unicode_mapping_rows, columns=["label-idx", "unicode-values"])
UPLOAD_FOLDER = os.getcwd()
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model= keras.models.load_model("assets/vit190/kaggle/working/VIT_Wei")

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route('/canvas')
def canvaspage():
    return render_template("canvas.html")

@app.route('/word')
def wordpage():
    return render_template("word.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        print(f)
        filepath = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filepath))

        upload_img = os.path.join(UPLOAD_FOLDER, filepath)
        image=cv2.imread(upload_img,cv2.IMREAD_GRAYSCALE)
        image=cv2.resize(image,(64,64))

        (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # plt.imshow(im_bw)
        # plt.xticks([]), plt.yticks([]) # optional line of code, just to hide tick values on X and Y axis if needed
        # plt.show()
        test_sample_image=img_to_array(im_bw)
        test_sample_image=np.asarray(test_sample_image)
        test_sample_image=(np.expand_dims (test_sample_image, 0))
        x=test_sample_image.shape
        prediction=model.predict(test_sample_image)
        predicted_class=np.argmax(prediction)
        print(predicted_class)
        return render_template('predict.html', num=str(unicode_mapping_df.loc[predicted_class][1]),num1="Character")
    else:
        return redirect("/")

@app.route('/predict2', methods=['GET', 'POST'])
def predictword():
    if request.method == "POST":
        f = request.files["image"]
        print(f)
        ans=""
        filepath = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filepath))

        upload_img = os.path.join(UPLOAD_FOLDER, filepath)
        img=cv2.imread(upload_img)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,20)
        ret,thresh=cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)
        plt.imshow(thresh)
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(cnts)
        cnt_t = []
        # removing repeated contours for same symbol
        for i, cnt in enumerate(cnts):
            if hierarchy[0][i][3] == -1:
                cnt_t.append(cnt)
        cnts = sorted(cnt_t, key=lambda b: b[0][0][0], reverse=False)
        plt.imshow(cv2.drawContours(thresh, cnts, -1, (0,255,0), 3),cmap='gray',vmin=0)


        # Extracting each symbol to predict
        for cnt in cnts:
            
            x, y, w, h = cv2.boundingRect(cnt)
            img_temp = thresh[y:y + h, x:x + w]
            top_bottom_padding = int((max(w, h) * 1.2 - h) / 2)
            left_right_padding = int((max(w, h) * 1.2 - w) / 2)
            img_temp = cv2.copyMakeBorder(img_temp, top_bottom_padding, top_bottom_padding, left_right_padding,
                                            left_right_padding,cv2.BORDER_CONSTANT,value=0)
            
            if(cv2.contourArea(cnt)<200):
                continue
            img_temp = cv2.resize(img_temp, (64, 64))
            #img_temp=thinning(img_temp)
            img_temp=255-img_temp
            img_temp=cv2.adaptiveThreshold(img_temp,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)
            fig=plt.figure()
            img_temp=np.asarray(img_temp) 
            non_zeros=img_temp.size-np.count_nonzero(img_temp)
            if(non_zeros==0):
                continue
            
            plt.imshow(img_temp,cmap='gray')
            img_temp=(np.expand_dims (img_temp, 0))
        #     img_temp = np.array(img_temp, dtype="float") / 255.0
            #print(prediction)
            
            prediction=model.predict(img_temp)
            predicted_class=np.argmax(prediction)
            ans+=(unicode_mapping_df.loc[predicted_class][1])
            print(ans)
            # prediction_score=max(prediction[0])
            # predicted_class=np.argmax(prediction)
       
        return render_template('predict.html', num=ans,num1="Word")
    else:
        return redirect("/")



@app.route('/predict1', methods=['GET', 'POST'])
def something():
    if request.method == "POST":
        # final_pred = None
        draw = request.form['url']
        #Removing the useless part of the url.
        draw = draw.split(',')[1]
        # encoded_data = uri.split(',')[1]
        #Decoding
        draw_decoded = base64.b64decode(draw)
        
       
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
       
        image = cv2.imdecode(image,  cv2.IMREAD_GRAYSCALE)
      
        resized = cv2.resize(image, (64,64), interpolation = cv2.INTER_AREA)
        resized=cv2.bitwise_not(resized)
        cv2.imwrite("result2.jpg", resized)
       
        vect = np.asarray(resized, dtype="uint8")
        test_sample_image=(np.expand_dims(vect, 0))
      
        prediction=model.predict(test_sample_image)
        predicted_class=np.argmax(prediction)
        print(predicted_class)          
        return render_template('predict.html', num=str(unicode_mapping_df.loc[predicted_class][1]),num1="Character")
    else:
        return redirect("/")



if __name__=="__main__":
    app.run("localhost",5000,debug=True)
