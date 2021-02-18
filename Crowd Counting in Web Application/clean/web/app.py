# coding=utf-8
import os
from flask import Flask
from flask import render_template

# app = Flask(__name__, template_folder="./", static_folder='./', static_url_path='')
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import time
from datetime import timedelta
import cv2
import pickle,random
from src.utils import *
import os
import torch
import numpy as np

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

TMP_FOLDER = 'static/tmp'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.secret_key = 'test'
app.config['TMP_FOLDER'] = TMP_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT']=timedelta(seconds=1)



@app.route('/')
def index():
    return render_template('index0.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def img_detect():
    print('img_detect')
    model_path = 'static/mcnn_shtechA_1000.h5'

    output_dir = './output/'
    model_name = os.path.basename(model_path).split('.')[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    net = CrowdCounter()

    trained_model = os.path.join(model_path)
    load_net(trained_model, net)
    net.cuda()
    net.eval()

    img = cv2.imread('static/tmp/a.jpg', 0)
    img = img.astype(np.float32, copy=False)
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = int((ht / 4) * 4)
    wd_1 = int((wd / 4) * 4)
    img = cv2.resize(img, (wd_1, ht_1))
    img = img.reshape((1, 1, img.shape[0], img.shape[1]))

    im_data =img
    gt_data =[]
    density_map = net(im_data, gt_data)
    density_map = density_map.data.cpu().numpy()
    et_count = np.sum(density_map)

    return int(et_count)

#img_detect()

@app.route('/updetectimg', methods=['GET', 'POST'])
def upload_file():
    print('get upload')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return render_template('index.html')
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return render_template('index.html')
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['TMP_FOLDER'], 'a.jpg'))

            time.sleep(2)
            print('saved...')
            ans=0
            try:
                ans=img_detect()
            except:
                ans=0
            strans=str(ans)#'Detect Result:'+str(ans)
            return render_template('index1.html', p_num=strans)
    else:
        return render_template('index0.html')



if __name__ == '__main__':
    app.debug = True  # Set mode of debug
    app.run(host='127.0.0.1')
    pass