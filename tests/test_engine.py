from engine import train_step, train
from engine import test_step as tst_step


def test_train_step(model, dataloaders, loss_fn, optimizer, device):
    train_loader, _, _ = dataloaders
    model, _ = model

    train_loss, train_acc = train_step(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )

    assert isinstance(train_loss, float)
    assert isinstance(train_acc, float)


def test_test_step(model, dataloaders, loss_fn, device):
    _, test_loader, _ = dataloaders
    model, _ = model

    test_loss, test_acc = tst_step(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device
    )

    assert isinstance(test_loss, float)
    assert isinstance(test_acc, float)


def test_train(dataloaders, model, loss_fn, optimizer, device):
    train_loader, test_loader, _ = dataloaders
    model, _ = model

    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=1,
        device=device
    )

    assert len(results["train_loss"]) == 1
    assert len(results["train_acc"]) == 1
    assert len(results["test_loss"]) == 1
    assert len(results["test_acc"]) == 1
