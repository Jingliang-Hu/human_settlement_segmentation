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

def train(model, device, train, valid, n_epoch, criterion, optimizer, scheduler, save_best_model=1, model_dir='./model', patient=10):
    print('training started ...')
    tra_loss = np.zeros(n_epoch)
    val_loss = np.zeros(n_epoch)

    tra_acc = np.zeros(n_epoch)
    val_acc = np.zeros(n_epoch)
    lr_rate = np.zeros(n_epoch)

    model_epoch = 0 

    for epoch in range(n_epoch):
        model.train()
        running_loss = 0.0
        correct = 0.0
        total_pred = 0.0
        lr_rate[epoch] = optimizer.param_groups[0]['lr']
        for batch in tqdm(train):
            # data and label for forward proporgation
            data = batch['data'].type(torch.float32).to(device)
            label = batch['label'].type(torch.long).to(device)
            mask = batch['mask'].to(device)
            pred = model(data).type(torch.float32)
            pred[mask==0] = 0
            _, predicted = torch.max(pred.data, 1)

            # back proporgation
            optimizer.zero_grad()
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * data.size(0)
            correct += torch.sum(label[label>0]==predicted[label>0]).item()
            total_pred += torch.sum(label>0).item()

        tra_loss[epoch] = running_loss / len(train)
        tra_acc[epoch] = correct/total_pred
        _, val_loss[epoch], val_acc[epoch] = test(model, device, valid.dataset.dataset, criterion)


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


def test(model, device, test_dat, criterion):

    model.eval()

    pred_loader = DataLoader(test_dat, batch_size=1)
    nb_img_patches = len(pred_loader.dataset)

    running_loss = 0.0
    correct = 0.0
    total_pred = 0.0

    out_prediction = torch.zeros(test_dat.label.shape)

    with torch.no_grad():
        for i_batch, batch in enumerate(pred_loader):
            # data and label for forward proporgation
            data = batch['data'].type(torch.float32).to(device)
            label = batch['label'].type(torch.long).to(device)
            mask = batch['mask'].to(device)
            pred = model(data).type(torch.float32)
            pred[mask==0] = 0
            loss = criterion(pred, label)
            _, predicted = torch.max(pred.data, 1)

            # statistics
            running_loss += loss.item() * data.size(0)
            correct += torch.sum(label[label>0]==predicted[label>0]).item()
            total_pred += torch.sum(label>0).item()

            out_prediction[i_batch:i_batch+data.shape[0],:,:] = predicted

    testLoss = running_loss/nb_img_patches
    accuracy = correct/total_pred


    return out_prediction, testLoss, accuracy







