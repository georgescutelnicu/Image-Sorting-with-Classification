import os
import torch
from pathlib import Path
import argparse
import data_setup
import engine
import model_builder
import utils


def main(parsed_args):
    num_epochs = parsed_args.epochs
    num_workers = os.cpu_count()
    batch_size = parsed_args.batch_size
    learning_rate = parsed_args.learning_rate

    # Directories
    train_dir = Path(parsed_args.train_dir)
    test_dir = Path(parsed_args.test_dir)

    num_classes = len(next(os.walk(train_dir))[1])

    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model, transform = model_builder.create_effnetb2_model(num_classes=num_classes)
    model.to(device)

    # Dataloaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Train model
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=num_epochs,
        device=device
    )

    # Save model and loss curve plot
    loss_fig = utils.plot_loss_curve(results)
    loss_fig.savefig("loss_curve.png")
    utils.save_model(model=model, model_name=parsed_args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Training Script")

    parser.add_argument('--train-dir', '-tr', type=str, required=True, help='Path to the training directory')
    parser.add_argument('--test-dir', '-te', type=str, required=True, help='Path to the testing directory')
    parser.add_argument('--epochs', '-e', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, required=True, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--model_name', '-m', type=str, required=True, help='Name for the saved model file')

    args = parser.parse_args()
    main(args)
