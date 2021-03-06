from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import torch
from torch import nn

from datetime import date

from utils import plot_loss_progress

def build_flow(num_dim, hidden_features=1024, layers=5, batch_norm=False):
  """
  Build a MAF
  """
  base_dist = StandardNormal(shape=[num_dim])
  transforms = []
  
  for _ in range(layers):
    transforms.append(ReversePermutation(features=num_dim))
    transforms.append(MaskedAffineAutoregressiveTransform(features=num_dim,
                                                          hidden_features=hidden_features,
                                                          use_batch_norm=batch_norm))
  transform = CompositeTransform(transforms)
  return Flow(transform, base_dist)


def build_cond_flow(num_dim, condition='digit', hidden_features=1024, layers=5, batch_norm=False):
  posible_conditioning = { 'digit': 10, 'color': 3 }
  context_features = posible_conditioning.get(condition, 1)
  base_dist = ConditionalDiagonalNormal(shape=[num_dim],
                                        context_encoder=nn.Linear(context_features, 2 * num_dim))
  transforms = []
  
  for _ in range(layers):
    transforms.append(ReversePermutation(features=num_dim))
    transforms.append(MaskedAffineAutoregressiveTransform(features=num_dim,
                                                          hidden_features=hidden_features,
                                                          use_batch_norm=batch_norm,
                                                          context_features=context_features))
  
  transform = CompositeTransform(transforms)
  return Flow(transform, base_dist)


def train(flow, optimizer, train_loader, val_loader, patience=30, monitor_every_batch=False, show_epoch_loss_progress=False, show_flow=False, conditional=False, conditioning_criteria='digit'):
  best_val_loss = float('inf')
  maxepochs = 999
  best_epoch = None
  counter_step = 0
  progress_epoch = []
  progress_trn_loss = []
  progress_val_loss = []

  for epoch in range(1, maxepochs + 1):
    iter = 1

    t_loss = 0.0
    for x, y, l in train_loader:
      # print(y[0])
      optimizer.zero_grad()
      train_loss = -flow.log_prob(inputs=x, context=y).mean() if conditional else -flow.log_prob(inputs=x).mean()
      train_loss.backward()
      optimizer.step()

      t_loss = train_loss.cpu().detach().numpy() if torch.cuda.is_available() else train_loss.detach().numpy()
      
      if monitor_every_batch:
        print('Epoch {}, Iteration {}, Train loss {:.5f}'.format(epoch, iter, t_loss))

      iter += 1

    v_loss = 0.0
    flow.eval()
    with torch.no_grad():
      for x, y, l in val_loader:
        val_loss = -flow.log_prob(inputs=x, context=y).mean() if conditional else -flow.log_prob(inputs=x).mean()
        v_loss = val_loss.cpu().detach().numpy() if torch.cuda.is_available() else val_loss.detach().numpy()
      
    progress_trn_loss.append(t_loss)
    progress_val_loss.append(v_loss)
    progress_epoch.append(epoch)

    print('Epoch: {}, Train loss: {:.5f}, Validation loss: {:.5f}'.format(epoch, t_loss, v_loss))

    if best_val_loss > v_loss:
      best_val_loss = v_loss
      today = date.today().strftime("%d-%m-%Y")
      filename = "MAF-MNIST-{}-{}-best.pth".format(conditioning_criteria if conditional else '', today)
      torch.save(flow, "/content/drive/MyDrive/Colab Notebooks/{}".format(filename))
      best_epoch = epoch
      counter_step = 0
    else:
      if counter_step == patience - 1:
        print('Stopped training after {} epochs'.format(epoch))
        break
      counter_step += 1
    
  if show_epoch_loss_progress:
    plot_loss_progress(progress_trn_loss, progress_val_loss, progress_epoch, best_epoch)

  return flow