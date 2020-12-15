import sys
import os
f = open("../../env_path","r")
env_path = f.readline()
env_path = env_path[:-1]
f.close()
sys.path.append(os.path.abspath(env_path+"/src/uil"))
sys.path.append(os.path.abspath(env_path+"/src/data"))
sys.path.append(os.path.abspath(env_path+"/src/model"))

import h5py
import torch
import numpy as np
import exp_uil as uil

'''
PARAMETER SETTING
'''
print("parameter setting ...")
paraDict = {
        ### network parameters
        "batch_size": 1,
        "n_epoch": 100,
        "training_patient": 20,
        "lr_rate": 0.1,
        "lr_rate_patient": 5,
        "val_percent": 0.2,
        "save_best_model": 1,
        ### data loading parameters
        #"trainData": "lcz42", # training data could be the training data of LCZ42 data, or data of one of the cultural-10 city
        "trainData": env_path+"/data/00010_23083_NewYork_Patch_32_LabPerc_70_bal.h5",
        #"testData": "cul10",  # testing data could be all the data of the cultural-10 cities, or one of them.
        "testData": env_path+"/data/00010_23083_NewYork_Patch_32_LabPerc_70_bal.h5",
        ### model name
        "backbone_model": 'unet',
        "exper_description": 'test_debug_run',
        }
cudaNow = torch.device("cuda:2")

'''
record experiment parameters
'''
outcomeDir = uil.initialOutputFolder(paraDict)
model_dir = os.path.join(outcomeDir,'model')
print("Experiments outcomes are saving in the directory: "+outcomeDir)
uil.recordExpParameters(outcomeDir,paraDict)

'''
STEP ONE: data loading
'''
print('loading data ...')
import load_data
from torch.utils.data import DataLoader, random_split
data_training = load_data.data_set(paraDict["trainData"])
n_val = int(len(data_training) * paraDict["val_percent"])
n_tra = len(data_training) - n_val
tra_dat, val_dat = random_split(data_training, [n_tra, n_val])
pred_dat = load_data.data_set(paraDict["testData"])
train_loader = DataLoader(tra_dat, batch_size=paraDict["batch_size"], shuffle=True)
valid_loader = DataLoader(val_dat, batch_size=paraDict["batch_size"], shuffle=True)


'''
STEP TWO: initial deep model
'''
import unet
if paraDict["backbone_model"] == 'unet':
    model = unet.UNet(data_training.data.shape[1], 4).to(cudaNow)
    predict_model = unet.UNet(data_training.data.shape[1], 4).to(cudaNow)

'''
STEP THREE: Define a loss function and optimizer
'''
import torch.optim as optim
import torch.nn as nn
classWeight = torch.tensor([0,1/3,1/3,1/3]).to(cudaNow)
criterion = nn.CrossEntropyLoss(weight=classWeight)
optimizer = optim.Adam(model.parameters(), lr=paraDict["lr_rate"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=paraDict["lr_rate_patient"], threshold=0.0001)

'''
STEP FOUR: Train the network
'''
# import modelOperation
import train
print('Start training ...')
traLoss, traArry, valLoss, valArry, model_epoch = train.train_alpha(model, cudaNow, train_loader, valid_loader, paraDict["n_epoch"], criterion, optimizer, scheduler, save_best_model=paraDict["save_best_model"], model_dir=model_dir, patient=paraDict["training_patient"])

'''
STEP FIVE: Prediction and calculate the metrics
'''
print('Start prediction ...')
predict_model.load_state_dict(torch.load(model_dir, map_location=cudaNow))
test_loss, test_accuracy = train.test_alpha(predict_model, cudaNow, pred_dat, criterion)

predictions = train.prediction(predict_model, cudaNow, pred_dat)
predictions = predictions.numpy()
label_tmp = pred_dat.label.copy()

y_true = label_tmp[label_tmp>0]
y_pred = predictions[label_tmp>0]

print('Calculating metrics ...')
# average accuracy
from sklearn.metrics import balanced_accuracy_score
aa = balanced_accuracy_score(y_true, y_pred)
# overall accuracy
from sklearn.metrics import accuracy_score
oa = accuracy_score(y_true, y_pred)
# kappa coefficient
from sklearn.metrics import cohen_kappa_score
ka = cohen_kappa_score(y_true, y_pred)
# iou
# from sklearn.metrics import jaccard_score
# iou = jaccard_score(y_true[0], y_pred[0])
# producer accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
cm_tmp = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
pa = cm_tmp.diagonal()



'''
STEP SIX: Save the model and accuracy
'''
print('Saving evaluation metrics ...')
fid = h5py.File(os.path.join(outcomeDir,'training_history.h5'),'w')
fid.create_dataset('traLoss',data=traLoss)
fid.create_dataset('traArry',data=traArry)
fid.create_dataset('valLoss',data=valLoss)
fid.create_dataset('valArry',data=valArry)
fid.create_dataset('model_epoch',data=model_epoch)
fid.close()

fid = h5py.File(os.path.join(outcomeDir,'test_accuracy.h5'),'w')
fid.create_dataset('cm',data=cm)
fid.create_dataset('oa',data=oa)
fid.create_dataset('aa',data=aa)
fid.create_dataset('ka',data=ka)
fid.create_dataset('pa',data=pa)
fid.create_dataset('model_epoch',data=model_epoch)
fid.close()

print('Training finished ')



