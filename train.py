import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from torchsummary import summary
from tqdm import tqdm
import os 
from model import *

config = {
  "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
  # you can set your own training configurations
  "batch_size": 100,
  "learning_rate": 1e-5,
  'epochs': 1000,
  'save_path': './models'
}

class ScritchData(Dataset):
    def __init__(self, filenames):
        arr = []
        for filename in filenames:
            arr.append(np.loadtxt(filename, delimiter=",", dtype=np.float32))
            # print(arr[-1].shape)
        arr = np.vstack(arr)
        # print(arr.shape)
        z_data, label = arr[:, 2], arr[:, 3]

        window = lambda a, w, o: np.lib.stride_tricks.as_strided(a, strides = a.strides * 2, shape = (a.size - w + 1, w))[::o]
        window_size, sliding_size = int(WINDOW_LENGTH/SAMPLING_PERIOD), int(STRIDE_LENGTH/SAMPLING_PERIOD)

        inp_feat = window(z_data, window_size, sliding_size)
        out_feat = np.sum(
            window(label, window_size, sliding_size),
            axis=1) > sliding_size//2
        out_feat = out_feat.astype('float32')

        self.inp_feat = inp_feat
        self.out_feat = out_feat
    
    def __len__(self):
        return len(self.out_feat)
    
    def __getitem__(self, idx):
        return self.inp_feat[idx], self.out_feat[idx]
    
# full_dataset = ScritchData(['./data/data1.csv', './data/data2.csv', './data/data3.csv'])
full_dataset = ScritchData(['./data/data_finger.csv'])

train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_ds, valid_ds = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

train_dl = DataLoader(train_ds, config["batch_size"], shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(valid_ds, config["batch_size"], shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
  
model = Scritch().to(config['device'])

summary(model, input_size=(1, int(WINDOW_LENGTH/SAMPLING_PERIOD)))

loss_func = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity
# initialize history for recording what we want to know
history = []
device, n_epochs, save_path = config['device'], config['epochs'], config['save_path']

early_stop_count = 0

for epoch in range(n_epochs):
    # monitor training loss, validation loss and learning rate
    train_loss = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    lrs    = []
    result = {'train_loss': [], 'val_loss': [], 'lrs': []}

    # prepare model for training
    model.train()

    #######################
    # train the model #
    #######################
    for batch_idx, item in enumerate(tqdm(train_dl)):
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

    ######################
    # validate the model #
    ######################
    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(tqdm(val_dl)):
            x, y = item
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # compute predicted outputs by passing inputs to the model
            output = model(x).squeeze(dim=1)
            # calculate the loss
            loss = loss_func(output,y)

            # update running validation loss
            valid_loss += loss.item()*x.size(0)

            pred = output > 0.5
            valid_acc += pred.eq(y).sum().item()

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_dl.dataset)
    result['train_loss'] = train_loss
    valid_loss = valid_loss/len(val_dl.dataset)
    result['val_loss'] = valid_loss
    leaning_rate = lrs
    result['lrs'] = leaning_rate
    history.append(result)

    valid_acc = (100. * valid_acc) / len(val_dl.dataset)

    print('Epoch {:2d}, lr: {:.6f} Train Loss: {:.6f} Valid Loss: {:.6f} Valid Acc: {:.2f}%'.format(
        epoch+1,
        leaning_rate[-1],
        train_loss,
        valid_loss,
        valid_acc
        ))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print("Validation loss decreased({:.6f}-->{:.6f}). Saving model ..".format(
            valid_loss_min,
            valid_loss
        ))
        torch.save(model.state_dict(), save_path + 'model.pt')
        valid_loss_min = valid_loss
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

    if early_stop_count >= 100:
        print('\nModel is not improving, so we halt the training session.')
        break