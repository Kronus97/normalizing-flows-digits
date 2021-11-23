import torch
from torch import nn
from torch import optim

from utils import load_vectorized_data, color_features, get_data_loader
from flow import train, build_flow, build_cond_flow

if torch.cuda.is_available():
  device = torch.device('cuda:0')
  torch.cuda.set_device(device)


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

# parameters for simulation
num_dim = train_x.shape[1]
batch_size = 100
weight_decay = 1.0e-6
lr = 1.0e-4
patience = 30
conditioning_criteria = 'color'
conditional = True

# adjust for conditioning
if conditioning_criteria == 'color':
  (train_x, train_y) = color_features(train_x)
  (val_x, val_y) = color_features(val_x)

train_loader = get_data_loader((train_x, train_y, train_labels), batch_size)
val_loader = get_data_loader((val_x, val_y, val_labels), batch_size)

flow = build_cond_flow(num_dim, condition=conditioning_criteria) if conditional else build_flow(num_dim)

if torch.cuda.is_available():
  flow = flow.to(device)

optimizer = optim.Adam(flow.parameters(),
                       lr=lr,
                       weight_decay=0 if weight_decay is None else weight_decay)

# train the model
model = train(flow,
              optimizer,
              train_loader, 
              val_loader, 
              patience=patience,
              show_epoch_loss_progress=True,
              conditional=True,
              conditioning_criteria=conditioning_criteria)