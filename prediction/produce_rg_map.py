import sys
import os
f = open("../env_path","r")
env_path = f.readline()
env_path = env_path[:-1]
f.close()
sys.path.append(os.path.abspath(env_path+"/src/uil"))
sys.path.append(os.path.abspath(env_path+"/src/data"))
sys.path.append(os.path.abspath(env_path+"/src/model"))

import h5py
import torch
import numpy as np
import geo_uil_model_branch as geo
import exp_uil as uil
from osgeo import gdal

##########################################################################
#
# Parameter setting
#
model_directory = '../experiments/test/unet_regression_test_debug_run_outcome_2020-12-16_09-56-38/model'
data_directory  = '../data/23083_autumn.tif'
out_map_directory = 'new_york_regression_map.tif'
patch_size = 32
model_type = model_directory.split('/')[-2].split('_')[0]
cudaNow = torch.device("cuda:0")


'''
step 1: initial segmentation map
'''
print('initial geotiff file for segmentation map')
geo.intial_geotiff_segmentation_prob_map(data_directory, out_map_directory)


'''
step 2: cut data patch
'''
import load_data
print('generate data patches for inference')
data_patches, img_coord = geo.cut_patch_4_inference(data_directory, patch_size)
pred_dat = load_data.data_4_prediction(data_patches)

'''
step 3: inferencing
'''
print('inferencing ...')
import unet
import train

if model_type == 'unet':
    predict_model = unet.UNet(pred_dat.data.shape[1], 3).to(cudaNow)

predict_model.load_state_dict(torch.load(model_directory, map_location=cudaNow))
predictions = train.prediction_rg(predict_model, cudaNow, pred_dat, batch_size=256)
predictions = predictions.numpy()

'''
step 4: save inferencing
'''

final_map = geo.save_regression_inference(out_map_directory, predictions, img_coord)









