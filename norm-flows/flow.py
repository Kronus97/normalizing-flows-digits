from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import torch
from torch import nn
from torch import optim

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


def build_cond_flow(num_dim, hidden_features=1024, layers=5, batch_norm=False):
  """
  Build a conditioned normalizing flow
  """
  base_dist = ConditionalDiagonalNormal(shape=[num_dim],
                                        context_encoder=nn.Linear(1, 2 * num_dim))
  transforms = []
  
  for _ in range(layers):
    transforms.append(ReversePermutation(features=num_dim))
    transforms.append(MaskedAffineAutoregressiveTransform(features=num_dim,
                                                          hidden_features=hidden_features,
                                                          use_batch_norm=batch_norm,
                                                          context_features=1))
  
  transform = CompositeTransform(transforms)
  return Flow(transform, base_dist)


def train(train_x, train_labels, train_loader, val_loader, maxepochs=1, monitor_every_batch=False, show_epoch_loss_progress=False, show_flow=False, weight_decay=None, conditional=False):
  num_dim = train_x.shape[1]
  lr = 1.0e-4
  best_val_loss = float('inf')
  best_epoch = None
  progress_epoch = []
  progress_trn_loss = []
  progress_val_loss = []
  flow = build_cond_flow(num_dim) if conditional else build_flow(num_dim)

  if torch.cuda.is_available():
    flow = flow.to(device)
  
  optimizer = optim.Adam(flow.parameters(),
                         lr=lr,
                         weight_decay=0 if weight_decay is None else weight_decay)

  # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min(1., epoch / 10))

  for epoch in range(1, maxepochs + 1):
    iter = 1

    t_loss = 0.0
    for x, y, l in train_loader:
      optimizer.zero_grad()
      train_loss = -flow.log_prob(inputs=x, context=l.reshape(-1, 1)).mean() if conditional else -flow.log_prob(inputs=x).mean()
      train_loss.backward()
      optimizer.step()

      t_loss = train_loss.cpu().detach().numpy() if torch.cuda.is_available() else train_loss.detach().numpy()
      
      if monitor_every_batch:
        print('Epoch {}, Iteration {}, Train loss {:.5f}'.format(epoch, iter, t_loss))

      iter += 1

    v_loss = 0.0
    with torch.no_grad():
      for x, y, l in val_loader:
        val_loss = -flow.log_prob(inputs=x, context=l.reshape(-1, 1)).mean() if conditional else -flow.log_prob(inputs=x).mean()
        v_loss = val_loss.cpu().detach().numpy() if torch.cuda.is_available() else val_loss.detach().numpy()
      
    progress_trn_loss.append(t_loss)
    progress_val_loss.append(v_loss)
    progress_epoch.append(epoch)

    print('Epoch: {}, Train loss: {:.5f}, Validation loss: {:.5f}'.format(epoch, t_loss, v_loss))

    if best_val_loss > v_loss:
      best_val_loss = v_loss
      # this uses the mounted google drive
      torch.save(flow, "/content/drive/MyDrive/Colab Notebooks/MAF-MNIST-digit-cond-15-11-2021-best.pth")
      best_epoch = epoch

    
  if show_epoch_loss_progress:
    plot_loss_progress(progress_trn_loss, progress_val_loss, progress_epoch, best_epoch)

  return flow