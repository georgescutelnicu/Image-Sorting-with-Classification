
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

import torch
from torchvision import datasets

def plot_random_images(image_directory: str):

  """Plot random images from the dataset."""

  dir = Path(image_directory)
  train_data = datasets.ImageFolder(dir)
  class_names = train_data.classes
  ROWS = 3
  COLS = 3
  fig = plt.figure(figsize=(9,9))
  for i in range(ROWS*COLS):
    img_idx = torch.randint(1, len(train_data), size=[1]).item()
    img, img_label = train_data[img_idx]
    fig.add_subplot(ROWS, COLS, i+1)
    plt.imshow(img)
    plt.title(class_names[img_label])
    plt.axis(False)


def plot_loss_curve(model_results: dict):

  """Plot the loss curve of a model using a dictionary with the results."""

  plt.figure(figsize=(12,5))

  model_df = pd.DataFrame(model_results)
  epochs = range(len(model_results['train_loss']))

  plt.subplot(1,2,1)
  plt.plot(epochs, model_df['train_loss'], label='Train')
  plt.plot(epochs, model_df['test_loss'], label='Test')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  plt.subplot(1,2,2)
  plt.plot(epochs, model_df['train_acc'], label='Train')
  plt.plot(epochs, model_df['test_acc'], label='Test')
  plt.title('Acc')
  plt.xlabel('Epochs')
  plt.legend()

  fig = plt.gcf()

  return fig


def save_model(model: torch.nn.Module,
               model_name: str):
  
  """Save PyTorch model."""

  # Create dir
  target_dir_path = Path('models')
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  model_save_path = target_dir_path / model_name

  # Save the model
  print(f"[INFO] Model saved to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
