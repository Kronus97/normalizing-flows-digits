import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from utils import load_vectorized_data, reshape_conditioned_vectors, color_features
from flow import train

if torch.cuda.is_available():
  device = torch.device('cuda:0')
  torch.cuda.set_device(device)

epochs = 50
wd_rate = 1.0e-6
conditioning_criteria = 'color'
conditional = True

# load data
train_data, validation_data, test_data = load_vectorized_data()

train_x = train_data[0]
train_labels = train_data[1]
train_y = train_data[2]

val_x = validation_data[0]
val_labels = validation_data[1]
val_y = validation_data[2]

test_x = test_data[0]
test_labels = test_data[1]
test_y = test_data[2]

if conditioning_criteria == 'color':
  (train_x, train_y) = color_features(train_x, randomize=True)
  (val_x, val_y) = color_features(val_x, randomize=True)

tt_train_x = torch.tensor(train_x, dtype=torch.float32)
tt_train_y = torch.tensor(train_y, dtype=torch.float32)
tt_train_labels = torch.tensor(train_labels, dtype=torch.float32)

tt_val_x = torch.tensor(val_x, dtype=torch.float32)
tt_val_y = torch.tensor(val_y, dtype=torch.float32)
tt_val_labels = torch.tensor(val_labels, dtype=torch.float32)

if torch.cuda.is_available():
  tt_train_x = tt_train_x.to(device)
  tt_train_y = tt_train_y.to(device)
  tt_train_labels = tt_train_labels.to(device)

  tt_val_x = tt_val_x.to(device)
  tt_val_y = tt_val_y.to(device)
  tt_val_labels = tt_val_labels.to(device)

train_set = TensorDataset(tt_train_x, tt_train_y, tt_train_labels)
val_set = TensorDataset(tt_val_x, tt_val_y, tt_val_labels)

batch_size = 100
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

# train the model
model = train(tt_train_x, 
              train_labels, 
              train_loader, 
              val_loader, 
              maxepochs=epochs, 
              weight_decay=wd_rate,
              show_epoch_loss_progress=True,
              conditional=conditional,
              conditioning_criteria=conditioning_criteria)