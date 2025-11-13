import os

import math
import pandas as pd
from itertools import product
import torch
import torch.nn as nn
from torchmetrics.functional.classification import f1_score, precision, recall

from models import Network
from preprocess import preprocess_dataset
from train import EPOCHS, train_model
from utils import get_variation_combos, create_dataloaders


# set the device type for training on torch
def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device selected for training: ", device)

    return device
 
# calls the training loop in the train.py file
def run_training(model, dataloaders, config, device):
    training_results = train_model(model, dataloaders, config, device)

    print("Training results for sequence length: ", 25, " are: ", training_results)

    return training_results

# tests trained model against test set and returns metrics
def test_model(model, dataloaders, config, device):
    model.eval()
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    test_dataset_loader = dataloaders[config["sequence_length"]]["test"]

    total, correct_predictions, total_loss = 0, 0, 0.0

    all_predictions, all_labels = [], []

    # gradient not needed for testing
    with torch.no_grad():
        for X_test, y_test in test_dataset_loader:
            if X_test.device.type != device.type:
                X_test, y_test = X_test.to(device, non_blocking=True), y_test.to(device, non_blocking=True)

            y_test = y_test.float()

            # predict on test set
            prediction = model(X_test)
            # compute loss
            computed_loss = loss_fn(prediction, y_test)
            total_loss += computed_loss.item()

            # number of samples
            total += y_test.size(0)

            # apply sigmoid in output layer
            probs = torch.sigmoid(prediction)
            prediction = (probs >= 0.5).float()

            # compute accuracy
            correct_predictions += (prediction == y_test).sum().item()

            all_predictions.append(prediction)
            all_labels.append(y_test)

    # accuracy of overall test set and average test loss
    test_accuracy = correct_predictions / total
    avg_test_loss = total_loss / len(test_dataset_loader)

    y_true = torch.cat(all_labels).to(device).int()
    y_pred = torch.cat(all_predictions).to(device).int()

    # get the f1, precision and recall scores
    precision_score = precision(y_pred, y_true, task="binary").item()
    recall_score = recall(y_pred, y_true, task="binary").item()
    f1 = f1_score(y_pred, y_true, task="binary").item()

    print(f"\nTest Loss: {avg_test_loss:.4f} \nTest Accuracy: {test_accuracy:.4f}\n")
    print(f"Precision: {precision_score:.4f} \n Recall: {recall_score:.4f} \n F1: {f1:.4f}")

    return {"test_loss": avg_test_loss, "test_accuracy": test_accuracy, "precision": precision_score, "recall": recall_score, "f1": f1}
        
def main():
    # set parameters
    max_words = 10000
    max_lengths = [25, 50, 100]
    dataset_path = os.getcwd()+"/data/IMDB dataset.csv"
    torch.manual_seed(42)

    # preprocess the data
    preprocessed_data = preprocess_dataset(dataset_path, max_words, max_lengths)

    print("Lenghth of Cleaned train texts:", len(preprocessed_data["cleaned_train_texts"]))
    print("Lenghth of Tokenized train samples:", len(preprocessed_data["tokenized_train"]))

    vocab_size = len(preprocessed_data["vocab"])
    print("Vocabulary size for the model: ", vocab_size)

    # get device that is available to train the model on
    device = check_device()

    # dataloaders for train, val and test sets
    dataloaders = create_dataloaders(preprocessed_data, batch_size=32, val_ratio=0.1, device=device)

    validation_results = []

    for i, config in enumerate(get_variation_combos(vocab_size)):
        # create the model
        model = Network(config)

        print(f"\nTraining with configuration number: {i+1} \n", config)

        # run training and get metrics
        training_metrics = run_training(model, dataloaders, config, device)

        # run evaluation on test set and get metrics
        test_metrics = test_model(model, dataloaders, config, device)

        # create a dict for train and test results
        training_result = {
            "model": config["model"],
            "activation": config["activation"],
            "optimizer": config["optimizer"],
            "sequence_length": config["sequence_length"],
            "gradient_clipping": "yes" if config["stability_strategy"] == "gradient_clipping" else "no",
            "accuracy": (math.floor(test_metrics["test_accuracy"] * 1000.0) / 1000.0)*100,
            "f1_score": math.floor(test_metrics["f1"] * 1000.0) / 1000.0,
            "time_per_epoch": math.floor((training_metrics["train_time"] / EPOCHS) * 1000.0) / 1000.0,
            "evaluation_accuracy": (math.floor(training_metrics["eval_acc"] * 1000.0) / 1000.0)*100,
            "training_total_time": math.floor(training_metrics["train_time"] * 1000.0) / 1000.0,
        }

        validation_results.append(training_result)

        print(test_metrics)

    # save the training summary as a CSV file after training is complete
    if validation_results:
        validation_results_df = pd.DataFrame(validation_results)
        csv_path = os.getcwd() + "/validation_results.csv"
        validation_results_df.to_csv(csv_path, index=False)
        print(f"\nSaved validation results to {csv_path}")


if __name__ == "__main__":
    main()
