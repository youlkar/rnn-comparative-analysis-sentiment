import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

np.random.seed(42)

# creates dataloaders for each of the sequence lengths
def create_dataloaders(preprocessed_data, batch_size, val_ratio = 0.1, generator_seed = 42, device=None):

    max_lengths = preprocessed_data["max_lengths"]

    # create tensors out of train and test labels
    train_labels = torch.tensor(preprocessed_data["train_labels"], dtype=torch.float32)
    test_labels = torch.tensor(preprocessed_data["test_labels"], dtype=torch.float32)

    train_padded_sequences = preprocessed_data["train_padded_sequences"]
    test_padded_sequences = preprocessed_data["test_padded_sequences"]

    # use cuda if available
    use_cuda = False
    if device is not None and getattr(device, "type", "") == "cuda":
        use_cuda = True

    # change the number of workes for dataloaders as cuda is available
    pin_memory = use_cuda
    if os.name == "nt":
        num_workers = 0
    else:
        if use_cuda:
            num_workers = 2
        else:
            num_workers = 4

    dataloaders = {}
    generator = torch.Generator().manual_seed(generator_seed)

    print("Amount of padded sequences for each sequence length: ", len(train_padded_sequences))
    print("Length of padded sequence: ", len(train_padded_sequences[0]))


    # create dataloaders for each of the sequence lengths
    for seq_len, train_array, test_array in zip(max_lengths, train_padded_sequences, test_padded_sequences):
        train_inputs = torch.tensor(train_array, dtype=torch.long)
        test_inputs = torch.tensor(test_array, dtype=torch.long)

        full_train_dataset = TensorDataset(train_inputs, train_labels)

        if int(len(full_train_dataset) * val_ratio) == 0:
            val_size = 1
        else:
            val_size = int(len(full_train_dataset) * val_ratio)

        train_dataset, val_dataset = random_split(full_train_dataset, [len(full_train_dataset) - val_size, val_size], generator=generator)


        # workers to keep dataloaders open for the entire training process
        persistent_workers = False
        if num_workers > 0:
            common_loader_kwargs["persistent_workers"] = True

        common_loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory, "persistent_workers":persistent_workers}


        # create dataloaders for training, validation and testing sets
        train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
        test_loader = DataLoader(TensorDataset(test_inputs, test_labels), shuffle=False, **common_loader_kwargs)

        # store dataloaders for each of the sequence lengths
        dataloaders[seq_len] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

    return dataloaders 



# variation combo configs - used for hyperparameters for the model
def get_variation_combos(vocab_size):
    return [
    {"model": "RNN", "activation": "sigmoid", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "RNN", "activation": "relu", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "RNN", "activation": "tanh", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "LSTM", "activation": "sigmoid", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "LSTM", "activation": "relu", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "LSTM", "activation": "tanh", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "BiLSTM", "activation": "sigmoid", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "BiLSTM", "activation": "relu", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "BiLSTM", "activation": "tanh", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},

    {"model": "RNN", "activation": "tanh", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "RNN", "activation": "tanh", "optimizer": "SGD", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "RNN", "activation": "tanh", "optimizer": "RMSprop", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "LSTM", "activation": "tanh", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "LSTM", "activation": "tanh", "optimizer": "SGD", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "LSTM", "activation": "tanh", "optimizer": "RMSprop", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "BiLSTM", "activation": "tanh", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "BiLSTM", "activation": "tanh", "optimizer": "SGD", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "BiLSTM", "activation": "tanh", "optimizer": "RMSprop", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},

    {"model": "RNN", "activation": "relu", "optimizer": "Adam", "sequence_length": 25, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "RNN", "activation": "relu", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "RNN", "activation": "relu", "optimizer": "Adam", "sequence_length": 100, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "LSTM", "activation": "relu", "optimizer": "Adam", "sequence_length": 25, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "LSTM", "activation": "relu", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "LSTM", "activation": "relu", "optimizer": "Adam", "sequence_length": 100, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "BiLSTM", "activation": "relu", "optimizer": "Adam", "sequence_length": 25, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "BiLSTM", "activation": "relu", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "BiLSTM", "activation": "relu", "optimizer": "Adam", "sequence_length": 100, "stability_strategy": "none", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},

    {"model": "RNN", "activation": "relu", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "gradient_clipping", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "LSTM", "activation": "relu", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "gradient_clipping", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
    {"model": "BiLSTM", "activation": "relu", "optimizer": "Adam", "sequence_length": 50, "stability_strategy": "gradient_clipping", "hidden_size": 64, "hidden_layers": 2, "dropout": 0.4, "vocab_size": vocab_size, "embedding_dim": 100},
]





