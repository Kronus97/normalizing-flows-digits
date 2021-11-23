import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data():
  """
  Load the mnist dataset
  """
  f = gzip.open('mnist.pkl.gz', 'rb')

  # fix for encoding of pickle
  u = pickle._Unpickler(f)
  u.encoding = 'latin1'

  train_data, validation_data, test_data = u.load()
  f.close()

  return (train_data, validation_data, test_data)


def preprocess_data(data, rng, alpha=1.0e-6, logit=False, should_dequantize=True):
  """
  Processes the dataset
  """
  x = dequantize(data[0], rng) if should_dequantize else data[0]  # dequantize pixels
  x = logit_transform(x, alpha) if logit else x                   # logit
  labels = data[1]                                                # numeric labels
  encoded_labels = one_hot_encode(labels, 10)                     # 1-hot encoded labels
  return (x, labels, encoded_labels)


def dequantize(x, rng):
  """
  Adds noise to pixels to dequantize them
  """
  return x + rng.rand(*x.shape) / 256.0

def logit_transform(x, alpha=1.0e-6):
  """
  Transforms pixel values with logit to reduce the impact of boundary effects
  """
  a = alpha + (1 - 2*alpha) * x
  return np.log(a / (1.0 - a))


def one_hot_encode(labels, nr_labels):
  """
  Transforms numeric labels to 1-hot encoded labels
  """
  y = np.zeros([labels.size, nr_labels])
  y[range(labels.size), labels] = 1
  return y


def load_vectorized_data():
  """
  Load the data inside vectors and return it as a tuple containing the training data, validation data, test data
  """
  train_data, validation_data, test_data = load_data()
  rng = np.random.RandomState(42)

  processed_train_data = preprocess_data(train_data, rng, logit=True)
  processed_validation_data = preprocess_data(validation_data, rng, logit=True)
  processed_test_data = preprocess_data(test_data, rng, logit=True)
  
  return (processed_train_data, processed_validation_data, processed_test_data)


def plot_loss_progress(progress_trn_loss, progress_val_loss, progress_epoch, best_epoch):
  """
  Plots the progress of the loss of the flow
  """
  fig, ax = plt.subplots(1, 1)
  ax.semilogx(progress_epoch, progress_trn_loss, 'b', label='training')
  ax.semilogx(progress_epoch, progress_val_loss, 'r', label='validation')
  ax.vlines(best_epoch, ax.get_ylim()[0], ax.get_ylim()[1], color='g', linestyles='dashed', label='best')
  ax.set_xlabel('epochs')
  ax.set_ylabel('loss')
  ax.legend()
  plt.show(block=False)


def color_features(input, randomize=False):
  np.random.seed(42)
  colors = []
  colored_input = np.zeros((input.shape[0], 3, input.shape[1]))

  if randomize:
    rng = np.random.RandomState(42)
    colored_input = dequantize(colored_input, rng)

  for i in range(len(input))[:]:
    r = np.random.permutation([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # print(r[0])
    colors.append(r[0])
    # print(np.argmax(r[0]))
    colored_input[i][np.argmax(r[0])][:] = input[i]
  
  colored_input = colored_input.reshape(colored_input.shape[0], 3*28*28)

  return (colored_input, np.array(colors))


def get_data_loader(data, batch_size, shuffle=True):
  x = torch.tensor(data[0], dtype=torch.float32)
  y = torch.tensor(data[1], dtype=torch.float32)
  z = torch.tensor(data[2], dtype=torch.float32)

  if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    z = z.cuda()

  dataset = TensorDataset(x, y, z)
  
  return DataLoader(dataset, batch_size, shuffle, num_workers=0)