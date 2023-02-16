
import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import data_setup, engine, model_builder, utils

NUM_EPOCHS = 1
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

# Directories
train_dir = Path('/content/train') 
test_dir = Path('/content/test')

NUM_CLASSES = len(next(os.walk(train_dir))[1])

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
model, transform = model_builder.create_effnetb2_model(num_classes=NUM_CLASSES)
model.to(device)

# Dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, 
                                                                               test_dir=test_dir, 
                                                                               transform=transform, 
                                                                               batch_size=BATCH_SIZE, 
                                                                               num_workers=NUM_WORKERS)

# Train model
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       epochs=NUM_EPOCHS,
                       device=device)
                       
# Save model
utils.save_model(model=model,
                 model_name="model_0.pth")
