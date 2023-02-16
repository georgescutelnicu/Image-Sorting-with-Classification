
from timeit import default_timer as timer 
from tqdm.auto import tqdm

import torch

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
  
  """Train for a single epoch."""

  train_loss = 0
  train_acc = 0

  model.train()
    
  for batch, (X, y) in enumerate(dataloader):

    X = X.to(device)
    y = y.to(device)

    y_prediction = model(X)

    loss = loss_fn(y_prediction, y)
    train_loss += loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    y_prediction_class = torch.argmax(torch.softmax(y_prediction, dim=1), dim=1)
    train_acc += (y_prediction_class == y).sum().item()/len(y_prediction)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)

  return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):

  """Test for a single epoch."""

  test_loss = 0
  test_acc = 0

  model.eval()
  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X = X.to(device)
      y = y.to(device)

      y_prediction = model(X)

      loss = loss_fn(y_prediction, y)
      test_loss += loss.item()

      y_prediction_class = y_prediction.argmax(dim=1)
      test_acc += ((y_prediction_class == y).sum().item()/len(y_prediction))

  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)

  return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
  
  """Train and test a model. It returns a dictionary with the results."""

  start_time = timer()
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}

  for epoch in tqdm(range(epochs)):

    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)
    
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)
    print(f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}")
    
     
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  end_time = timer()
  total_time = end_time - start_time
  print(f'Total time: {total_time:.2f} seconds')

  return results
