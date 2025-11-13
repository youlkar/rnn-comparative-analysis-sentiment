from torch._tensor import Tensor


import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


BATCH_SIZE = 32
EPOCHS = 10

# choose  the optimizer based on the params
def get_optimizer(config, params):
    learning_rate = 0.001

    if config["optimizer"] == "Adam":
        return optim.Adam(params, lr=learning_rate)
    elif config["optimizer"] == "SGD":
        return optim.SGD(params, lr=learning_rate)
    elif config["optimizer"] == "RMSprop":
        return optim.RMSprop(params, lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer selected")

def autocast_context(device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    elif device.type == "mps":
        return torch.amp.autocast(device_type="mps", dtype=torch.float16)
    else:
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)

def train_model(model, dataloaders, config, device):
    best_accuracy = 0.0
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(config, model.parameters())
    # set device type
    device_type = "cpu"
    if device.type == "cuda":
        device_type = "cuda"
    elif device.type == "mps":
        device_type = "mps"

    gradient_accumulation = 1 if device_type == "cuda" else 2

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True

    # move model to selected device
    model.to(device)

    # dataloaders for training and validation
    train_loader_dataset = dataloaders[config["sequence_length"]]["train"]
    val_loader_dataset = dataloaders[config["sequence_length"]]["val"]

    # gradient scalar for precision training
    use_amp = (device_type == "cuda")
    non_blocking = device_type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    print(f"\nTraining on {device.type.upper()} for {EPOCHS} epochs")
    training_start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for step, (X_train, y_train) in enumerate(train_loader_dataset):
            X_train = X_train.to(device, non_blocking=non_blocking)
            y_train = y_train.to(device, non_blocking=non_blocking)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                predictions = model(X_train)
                computed_loss = loss_function(predictions, y_train.float())
                computed_loss = computed_loss / gradient_accumulation 

            # backpropograte
            if use_amp:
                scaler.scale(computed_loss).backward()
            else:
                computed_loss.backward()

            # update after gradient accumulation
            if (step + 1) % gradient_accumulation == 0 or (step + 1) == len(train_loader_dataset):
                if config["stability_strategy"] == "gradient_clipping":
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            train_loss += computed_loss.item() * gradient_accumulation 

        avg_train_loss = train_loss / len(train_loader_dataset)

        # Validation
        model.eval()
        correct_predictions, total, validation_loss = 0, 0, 0.0
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
            for X_validation, y_validation in val_loader_dataset:
                X_validation = X_validation.to(device, non_blocking=non_blocking)
                y_validation = y_validation.to(device, non_blocking=non_blocking)

                validation_predictions = model(X_validation)
                val_loss = loss_function(validation_predictions, y_validation.float())
                validation_loss += val_loss.item()

                probabilities = torch.sigmoid(validation_predictions)
                correct_predictions += ((probabilities >= 0.5).float() == y_validation).sum().item()
                total += y_validation.size(0)

        avg_val_loss = validation_loss / len(val_loader_dataset)
        acc = correct_predictions / total

        if acc > best_accuracy:
            best_accuracy = acc

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")

    elapsed = time.time() - training_start_time
    return {"eval_acc": best_accuracy, "eval_loss": avg_val_loss, "train_time": elapsed}
