"""Custom transforms loading data."""

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

  # def forward(self, img):
  #   img += torch.randn_like(img) * self.std_dev
  #   return img

  # def __repr__(self):
  #   format_string = self.__class__.__name__ + "("
  #   format_string += f"\n    std_dev={self.std_dev}"
  #   format_string += "\n)"
  #   return format_string