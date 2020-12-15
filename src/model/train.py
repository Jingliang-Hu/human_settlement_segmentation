import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def save_best_model_func(model, save_best_model, model_dir, epoch, val_loss):
    if val_loss[epoch] == val_loss[:epoch+1].min() and save_best_model:
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
            running_loss += loss.item() * label.shape[0]
        tra_loss[epoch] = running_loss / len(train) 
        val_loss[epoch], val_acc[epoch] = test_rg(model, device, valid.dataset.dataset, criterion)
               
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
        
    return tra_loss, tra_acc, val_loss, val_acc, model_epoch






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

    return tra_loss, tra_acc, val_loss, val_acc, model_epoch






def train(model, device, train, valid, n_epoch, criterion, optimizer, scheduler, save_best_model=1, model_dir='./model', patient=10):
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

            # reshape label and prediction to exclude no label pixels
            label = label.reshape(label.shape[0],label.shape[1]*label.shape[2])
            pred  = pred.reshape(pred.shape[0],pred.shape[1],pred.shape[2]*pred.shape[3])
            mask = label[0,:]>0
            label = label[:,mask]
            pred  = pred[:,:,mask]

            _, predicted = torch.max(pred.data, 1)
            # back proporgation
            optimizer.zero_grad()
            loss = criterion(pred, label-1)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * torch.sum(mask).item()
            correct += torch.sum((label-1)==predicted).item()
            total_pred += torch.sum(mask).item()

        tra_loss[epoch] = running_loss / total_pred
        tra_acc[epoch] = correct/total_pred
        val_loss[epoch], val_acc[epoch] = test(model, device, valid.dataset.dataset, criterion)


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

    return tra_loss, tra_acc, val_loss, val_acc, model_epoch


def prediction(model, device, test_dat):
    model.eval()
    pred_loader = DataLoader(test_dat, batch_size=16)
    out_prediction = torch.zeros((test_dat.data.shape[0], test_dat.data.shape[2], test_dat.data.shape[3]))

    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(pred_loader)):
            data = batch['data'].type(torch.float32).to(device)
            pred = model(data).type(torch.float32)
            _, predicted = torch.max(pred.data, 1)
            out_prediction[i_batch:i_batch+data.shape[0],:,:] = predicted
    print(torch.unique(out_prediction))
    return out_prediction

def prediction_rg(model, device, test_dat):
    model.eval()
    pred_loader = DataLoader(test_dat, batch_size=16)
    out_prediction = torch.zeros((test_dat.data.shape[0],3,test_dat.data.shape[2], test_dat.data.shape[3]))
    
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(pred_loader)):
            data = batch['data'].type(torch.float32).to(device)
            pred = model(data).type(torch.float32)
            out_prediction[i_batch:i_batch+data.shape[0],:,:,:] = predicted
    return out_prediction



def test_rg(model, device, test_dat, criterion):
    model.eval()
    pred_loader = DataLoader(test_dat, batch_size=1)

    running_loss = 0.0
    with torch.no_grad():
        for i_batch, batch in enumerate(pred_loader):
            # data and label for forward proporgation
            data = batch['data'].type(torch.float32).to(device)
            label = batch['label'].type(torch.float32).to(device)
            pred = model(data).type(torch.float32)
            # reshape label and prediction to exclude no label pixels
            loss = criterion(pred, label-1)
            # statistics
            running_loss += loss.item() * label.shape[0]
            
        testLoss = running_loss/len(pred_loader)
    return testLoss





def test(model, device, test_dat, criterion):

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

            # reshape label and prediction to exclude no label pixels
            label = label.reshape(label.shape[0],label.shape[1]*label.shape[2])
            pred  = pred.reshape(pred.shape[0],pred.shape[1],pred.shape[2]*pred.shape[3])
            mask = label[0,:]>0
            label = label[:,mask]
            pred  = pred[:,:,mask]

            _, predicted = torch.max(pred.data, 1)
            loss = criterion(pred, label-1)

            # statistics
            running_loss += loss.item() * torch.sum(mask).item()
            correct += torch.sum((label-1)==predicted).item()
            total_pred += torch.sum(mask).item()

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







