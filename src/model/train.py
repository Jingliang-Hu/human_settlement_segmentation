import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn


def save_best_model_func(model, save_best_model, model_dir, epoch, val_loss):
    if epoch == 0 and save_best_model:
        torch.save(model.state_dict(), model_dir)
        print('The model after %d th epoch training has the lowest validation loss and saved' % (epoch+1))
        return 1
    elif val_loss[epoch] < val_loss[:epoch].min() and save_best_model:
        torch.save(model.state_dict(), model_dir)
        print('The model after %d th epoch training has the lowest validation loss and saved' % (epoch+1))
        return 1
    return 0

def train_rg(model, device, train, valid, n_epoch, criterion, optimizer, scheduler, save_best_model=1, model_dir='./model', patient=10):
    print('training started ...')
    tra_loss = np.zeros(n_epoch)
    val_loss = np.zeros(n_epoch)
    lr_rate = np.zeros(n_epoch)
    model_epoch = 0
    #softmax = nn.Softmax(dim=1)
    for epoch in range(n_epoch):
        running_loss = 0.0
        correct = 0.0
        total_pred = 0.0
        lr_rate[epoch] = optimizer.param_groups[0]['lr']
        for batch in tqdm(train):
            # data and label for forward proporgation
            data = batch['data'].type(torch.float32).to(device)
            label = batch['label'].type(torch.float32).to(device)
            pred = model(data).type(torch.float32)
            # back proporgation
            optimizer.zero_grad()
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item() * train.batch_size
        tra_loss[epoch] = running_loss / len(train) / train.batch_size
        val_loss[epoch] = test_rg(model, device, valid.dataset.dataset, criterion, train.batch_size)
        # update learning rate
        scheduler.step(val_loss[epoch])
        print('epoch %d: training loss: %.6f; validation loss: %.6f; learning rate: %.6f' % (epoch+1, tra_loss[epoch], val_loss[epoch], lr_rate[epoch]))
        # save the model that has the lowest validation loss
        model_save_flag = save_best_model_func(model, save_best_model, model_dir, epoch, val_loss)
        model_epoch = epoch if model_save_flag else model_epoch
        # training early stop
        if patient>=n_epoch:
            continue
        elif epoch <= patient:
            continue
        elif (val_loss[epoch-patient]<=val_loss[epoch-patient+1:epoch+1]).all():
            print('Early stopping patient (%d) reached' % (patient))
            break
    return tra_loss, val_loss, model_epoch, lr_rate






def train_alpha(model, device, train, valid, n_epoch, criterion, optimizer, scheduler, save_best_model=1, model_dir='./model', patient=10):
    print('training started ...')
    tra_loss = np.zeros(n_epoch)
    val_loss = np.zeros(n_epoch)

    tra_acc = np.zeros(n_epoch)
    val_acc = np.zeros(n_epoch)
    lr_rate = np.zeros(n_epoch)

    model_epoch = 0
    model.train()

    for epoch in range(n_epoch):
        running_loss = 0.0
        correct = 0.0
        total_pred = 0.0
        lr_rate[epoch] = optimizer.param_groups[0]['lr']
        for batch in tqdm(train):
            # data and label for forward proporgation
            data = batch['data'].type(torch.float32).to(device)
            label = batch['label'].type(torch.long).to(device)
            pred = model(data).type(torch.float32)
            _, predicted = torch.max(pred.data, 1)
            # back proporgation
            optimizer.zero_grad()
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item() * label.numel()
            correct += torch.sum(label==predicted).item()
            total_pred += label.numel()
        tra_loss[epoch] = running_loss / total_pred
        tra_acc[epoch] = correct/total_pred
        val_loss[epoch], val_acc[epoch] = test_alpha(model, device, valid.dataset.dataset, criterion)


        # update learning rate
        scheduler.step(val_loss[epoch])
        print('epoch %d: training loss: %.6f; training acc: %.6f; validation loss: %.6f; validation acc: %.6f; learning rate: %.6f' % (epoch+1, tra_loss[epoch], tra_acc[epoch], val_loss[epoch], val_acc[epoch], lr_rate[epoch]))

        # save the model that has the lowest validation loss
        model_save_flag = save_best_model_func(model, save_best_model, model_dir, epoch, val_loss)
        model_epoch = epoch if model_save_flag else model_epoch

        # training early stop
        if patient>=n_epoch:
            continue
        elif epoch <= patient:
            continue
        elif (val_loss[epoch-patient]<val_loss[epoch-patient+1:epoch+1]).all():
            print('Early stopping patient (%d) reached' % (patient))
            break

    return tra_loss, tra_acc, val_loss, val_acc, model_epoch, lr_rate






def train_cl(model, device, train, valid, n_epoch, criterion, optimizer, scheduler, save_best_model=1, model_dir='./model', patient=10):
    print('training started ...')
    tra_loss = np.zeros(n_epoch)
    val_loss = np.zeros(n_epoch)
    tra_acc = np.zeros(n_epoch)
    val_acc = np.zeros(n_epoch)
    lr_rate = np.zeros(n_epoch)
    model_epoch = 0 
    model.train()
    for epoch in range(n_epoch):
        running_loss = 0.0
        correct = 0.0
        total_pred = 0.0
        lr_rate[epoch] = optimizer.param_groups[0]['lr']
        for batch in tqdm(train):
            # data and label for forward proporgation
            data = batch['data'].type(torch.float32).to(device)
            label = batch['label'].type(torch.long).to(device)
            pred = model(data).type(torch.float32)
            _, predicted = torch.max(pred.data, 1)
            # back proporgation
            optimizer.zero_grad()
            loss = criterion(pred, label-1)
            loss.backward()
            optimizer.step()
            # statistics
            correct += predicted.eq(label-1).sum().item()
            total_pred += data.shape[0] * data.shape[2] * data.shape[3]
            running_loss += loss.item() * data.shape[0]
        tra_loss[epoch] = running_loss / total_pred
        tra_acc[epoch] = correct/total_pred
        val_loss[epoch], val_acc[epoch] = test_cl(model, device, valid.dataset.dataset, criterion)
        # update learning rate
        scheduler.step(val_loss[epoch])
        print('epoch %d: training loss: %.6f; training acc: %.6f; validation loss: %.6f; validation acc: %.6f; learning rate: %.6f' % (epoch+1, tra_loss[epoch], tra_acc[epoch], val_loss[epoch], val_acc[epoch], lr_rate[epoch]))
        # save the model that has the lowest validation loss
        model_save_flag = save_best_model_func(model, save_best_model, model_dir, epoch, val_loss)
        model_epoch = epoch if model_save_flag else model_epoch
        # training early stop
        if patient>=n_epoch:
            continue
        elif epoch <= patient:
            continue
        elif (val_loss[epoch-patient]<val_loss[epoch-patient+1:epoch+1]).all():
            print('Early stopping patient (%d) reached' % (patient))
            break
    return tra_loss, tra_acc, val_loss, val_acc, model_epoch, lr_rate



def prediction(model, device, test_dat, batch_size):
    model.eval()
    pred_loader = DataLoader(test_dat, batch_size=batch_size)
    out_prediction = torch.zeros((test_dat.data.shape[0], test_dat.data.shape[2], test_dat.data.shape[3]))
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(pred_loader)):
            data = batch['data'].type(torch.float32).to(device)
            pred = model(data).type(torch.float32)
            _, predicted = torch.max(pred.data, 1)
            out_prediction[i_batch*batch_size:(i_batch+1)*batch_size,:,:,:] = predicted
    return out_prediction

def prediction_rg(model, device, test_dat, batch_size):
    model.eval()
    pred_loader = DataLoader(test_dat, batch_size=batch_size)
    out_prediction = torch.zeros((test_dat.data.shape[0],3,test_dat.data.shape[2], test_dat.data.shape[3]))
    
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(pred_loader)):
            data = batch['data'].type(torch.float32).to(device)
            pred = model(data).type(torch.float32)
            pred = softmax(pred)
            out_prediction[i_batch*batch_size:(i_batch+1)*batch_size,:,:,:] = pred
    return out_prediction



def test_rg(model, device, test_dat, criterion, batch_size):
    model.eval()
    pred_loader = DataLoader(test_dat, batch_size=batch_size)
    running_loss = 0.0
    #softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for i_batch, batch in enumerate(pred_loader):
            # data and label for forward proporgation
            data = batch['data'].type(torch.float32).to(device)
            label = batch['label'].type(torch.float32).to(device)
            pred = model(data).type(torch.float32)
            # reshape label and prediction to exclude no label pixels
            #pred = softmax(pred)
            loss = criterion(pred, label)
            # statistics
            running_loss += loss.item() * batch_size 
        testLoss = running_loss/len(pred_loader)/batch_size
    return testLoss





def test_cl(model, device, test_dat, criterion, batch_size=32):
    model.eval()
    pred_loader = DataLoader(test_dat, batch_size=batch_size)
    running_loss = 0.0
    correct = 0.0
    total_pred = 0.0
    with torch.no_grad():
        for i_batch, batch in enumerate(pred_loader):
            # data and label for forward proporgation
            data = batch['data'].type(torch.float32).to(device)
            label = batch['label'].type(torch.long).to(device)
            pred = model(data).type(torch.float32)
            _, predicted = torch.max(pred.data, 1)
            loss = criterion(pred, label-1)
            # statistics
            running_loss += loss.item()*data.shape[0]
            correct += predicted.eq(label-1).sum().item()
            total_pred += data.shape[0]*data.shape[2]*data.shape[3]
    testLoss = running_loss/total_pred
    accuracy = correct/total_pred
    return testLoss, accuracy



def test_alpha(model, device, test_dat, criterion):

    model.eval()

    pred_loader = DataLoader(test_dat, batch_size=1)

    running_loss = 0.0
    correct = 0.0
    total_pred = 0.0

    with torch.no_grad():
        for i_batch, batch in enumerate(pred_loader):
            # data and label for forward proporgation
            data = batch['data'].type(torch.float32).to(device)
            label = batch['label'].type(torch.long).to(device)
            pred = model(data).type(torch.float32)

            _, predicted = torch.max(pred.data, 1)
            loss = criterion(pred, label)

            # statistics
            running_loss += loss.item() * label.numel()
            correct += torch.sum(label==predicted).item()
            total_pred += label.numel()

    testLoss = running_loss/total_pred
    accuracy = correct/total_pred


    return testLoss, accuracy







