
"""Here is the code to train the CSRnet network"""
from PIL import Image

import os,cv2,time,random,sys,pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from utils_gen import gen_paths_img_dm, gen_var_from_paths
from utils_imgproc import norm_by_imagenet
from keras.optimizers import Adam
from model import CSRNet
import sys
from time import time, ctime
from utils_imgproc import image_preprocessing,generator_data
from utils_callback import eval_loss, callbacks_during_train


dataset = "A"
# Generate paths of (train, test) x (img, dm)
(test_img_paths, train_img_paths), (test_dm_paths, train_dm_paths) = gen_paths_img_dm(
    path_file_root='data/paths_train_val_test',
    dataset=dataset
)


# Generate raw images (standardized by imagenet rgb) and density maps
test_x, test_y = gen_var_from_paths(test_img_paths[:], unit_len=None), gen_var_from_paths(test_dm_paths[:], stride=8, unit_len=None)
test_x = norm_by_imagenet(test_x)  # The normalization and standardization of the original images in the test set.
# The normalization of the original images in the training set are below the image_preprocessing.

train_x, train_y = gen_var_from_paths(train_img_paths[:], unit_len=None), gen_var_from_paths(train_dm_paths[:], stride=8, unit_len=None)


optimizer = Adam(lr=1e-7)

# Create model
model = CSRNet(input_shape=(None, None, 3))
model.compile(optimizer=optimizer, loss='MSE', metrics=['mae'])

train_gen = generator_data(train_x,train_y,1)


model.load_weights("{}.hdf5".format(dataset))

epochs = 100  # Set how many round during the training

time_st = time()
for x in range(epochs):
    model.fit_generator(train_gen,steps_per_epoch= 100, verbose=1,)
    loss = eval_loss(model, test_x, test_y, quality=False)
    print("MAE-RMSE-SFN-MAPE={}, time consuming={}m-{}s\r".format(np.round(np.array(loss), 2),
            int((time() - time_st)/60), int((time() - time_st)-int((time() - time_st)/60)*60)))
    

model.save_weights("{}.hdf5".format(dataset))











