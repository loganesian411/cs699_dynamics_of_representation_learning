"""Custom transforms loading data."""

import numpy as np
import torch

class RandomNoise(torch.nn.Module):
  """Apply random Gaussian noise to each image.

  Args:
    std_dev (float): std_dev
  """
  def __init__(self, std_dev):
    super().__init__()
    self.std_dev = std_dev

  def forward(self, img):
    img += torch.randn_like(img) * self.std_dev
    return img

  def __repr__(self):
    format_string = self.__class__.__name__ + "("
    format_string += f"\n    std_dev={self.std_dev}"
    format_string += "\n)"
    return format_string

class RandomDrop(torch.nn.Module):
  """Randomly drop pixels from image (i.e., set to 0).

  Args:
    drop_percent (float): percentage of pixels to drop
  """
  def __init__(self, drop_percent):
    super().__init__()
    self.drop_percent = drop_percent

  def forward(self, img):
    # This won't work if the channels aren't the first dimension.....
    mask = torch.rand_like(img[0, :, :]) <= self.drop_percent
    img[:, mask] = 0
    return img

  def __repr__(self):
    format_string = self.__class__.__name__ + "("
    format_string += f"\n    drop percent={self.drop_percent}"
    format_string += "\n)"
    return format_string

class ShufflePixels(torch.nn.Module):
  """Randomly shuffles pixels.

  Args:
    num_to_shuffle (int): number of pixels that'll be swapped. needs to be even
      to change their places
  """
  def __init__(self, num_to_shuffle):
    super().__init__()
    if num_to_shuffle % 2 == 1: # Enforce even number.
      num_to_shuffle += 1
    self.half_num_to_shuffle = int(num_to_shuffle/2)

  def forward(self, img):
    inds = torch.randperm(torch.numel(img[0, :, :]))[:self.half_num_to_shuffle * 2]

    def unflatten_inds(ind, num_rows, num_cols):
      return int(ind // num_rows), int(ind % num_rows)

    vec_unflatten = np.vectorize(unflatten_inds)
    row_inds, col_inds = vec_unflatten(inds, img.shape[1], img.shape[2])

    first_half_row = row_inds[:self.half_num_to_shuffle].astype('int')
    second_half_row = row_inds[self.half_num_to_shuffle:self.half_num_to_shuffle*2].astype('int')
    first_half_col = col_inds[:self.half_num_to_shuffle].astype('int')
    second_half_col = col_inds[self.half_num_to_shuffle:self.half_num_to_shuffle*2].astype('int')

    tmp = torch.clone(img[:, first_half_row, first_half_col])
    img[:, first_half_row, first_half_col] = img[:, second_half_row, second_half_col]
    img[:, second_half_row, second_half_col] = tmp
    return img

  def __repr__(self):
    format_string = self.__class__.__name__ + "("
    format_string += f"\n    num to shuffle={self.half_num_to_shuffle}"
    format_string += "\n)"
    return format_string