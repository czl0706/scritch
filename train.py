import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchmetrics.classification import Precision, Recall
import os 
from utils import *

config = {
  'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
  'batch_size': 200,
  'learning_rate': 1e-5,
  'epochs': 100000,
  'early_stop': 100,
  'save_path': './models',
  'seed': 42,
  'log_step': 50
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(config['seed'])
    
# full_dataset = ScritchData(['./data/data1.csv', 
#                             './data/data2.csv',
#                             './data/data3.csv',
#                             './data/data4.csv',
#                             './data/data5.csv',
#                             './data/data6.csv'])

full_dataset = get_dataset()

train_ds, valid_ds = torch.utils.data.random_split(full_dataset, [0.7, 0.3])

train_dl = DataLoader(train_ds, config['batch_size'], shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
valid_dl = DataLoader(valid_ds, config['batch_size'], shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
  
device, n_epochs, save_path, log_step, early_stop = \
    config['device'], config['epochs'], config['save_path'], config['log_step'], config['early_stop']

model = Scritch().to(device)

# loss_func = nn.BCELoss()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

precision = Precision(task='binary').to(device)
recall = Recall(task='binary').to(device)

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial 'min' to infinity
valid_acc_max = 0.0
# initialize history for recording what we want to know
history = []

early_stop_count = 0

for epoch in range(n_epochs):
    # monitor training loss, validation loss and learning rate
    train_loss = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    train_size = 0
    valid_size = 0
    lrs    = []
    saved_target, saved_preds = [], []
    result = {'train_loss': [], 'val_loss': [], 'lrs': []}

    # prepare model for training
    model.train()

    #######################
    # train the model #
    #######################
    # for batch_idx, item in enumerate(tqdm(train_dl)):
    for batch_idx, item in enumerate(train_dl):
        x, y = item
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(x).squeeze(dim=1)
        # calculate the loss
        loss = loss_func(output, y)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        # record learning rate
        lrs.append(optimizer.param_groups[0]['lr'])

        # update running training loss
        train_loss += loss.item()*x.size(0)
        train_size += x.size(0)

    ######################
    # validate the model #
    ######################
    model.eval()
    with torch.no_grad():
        # for batch_idx, item in enumerate(tqdm(valid_dl)):
        for batch_idx, item in enumerate(valid_dl):
            x, y = item
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # compute predicted outputs by passing inputs to the model
            output = model(x).squeeze(dim=1)
            # calculate the loss
            loss = loss_func(output,y)

            # update running validation loss
            valid_loss += loss.item()*x.size(0)

            # pred = (output > THRESHOLD)
            # valid_acc += pred.eq(y).sum().item()

            y = y.argmax(dim=1)
            pred = output.argmax(dim=1)
            valid_acc += pred.eq(y).sum().item()

            saved_target.append(y.cpu())
            saved_preds.append(pred.cpu())

            valid_size += x.size(0)

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = train_loss / train_size
    result['train_loss'] = train_loss
    valid_loss = valid_loss / valid_size
    result['val_loss'] = valid_loss
    leaning_rate = lrs
    result['lrs'] = leaning_rate
    history.append(result)

    valid_acc = (100. * valid_acc) / valid_size

    saved_target, saved_preds = torch.vstack(saved_target), torch.vstack(saved_preds)

    if (epoch+1) % log_step == 0:
        # print('\nEpoch {:2d}, lr: {:.6f} Train Loss: {:.6f} Valid Loss: {:.6f} Valid Acc: {:.2f}%\nPrecision: {:.2f}%, Recall: {:.2f}%\n'.format(
        #     epoch+1,
        #     leaning_rate[-1],
        #     train_loss,
        #     valid_loss,
        #     valid_acc,
        #     100.0 * precision(saved_target, saved_preds),
        #     100.0 * recall(saved_target, saved_preds)
        #     ))
        print('\n'.join(['', 
                         'Epoch {:2d}, lr: {:.6f} Train Loss: {:.6f} Valid Loss: {:.6f} Valid Acc: {:.2f}%', 
                         'Precision: {:.2f}%, Recall: {:.2f}%', 
                         '']).format(epoch+1,
                                     leaning_rate[-1],
                                     train_loss,
                                     valid_loss,
                                     valid_acc,
                                     100.0 * precision(saved_preds, saved_target), 
                                     100.0 * recall(saved_preds, saved_target)))

    # Apply early stopping and save model if validation loss has decreased
    # Use validation loss as the metric
    if valid_loss <= valid_loss_min:
        print(f'Validation loss decreased({valid_loss_min:.6f} -> {valid_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
        valid_loss_min = valid_loss
        valid_acc_max = valid_acc
        # print('Saving checkpoint...')
        # state = {
        #     'state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'epoch': epoch,
        #     'valid_loss_min': valid_loss_min }
        # if not os.path.isdir(save_path): os.mkdir(save_path)
        # torch.save(state, save_path + 'checkpoint.pth')
        
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= early_stop:
        print(f'\nModel is not improving after {early_stop} epochs, so we halt the training session.')
        break

print(f'Model accuracy: {valid_acc_max:.2f}%')