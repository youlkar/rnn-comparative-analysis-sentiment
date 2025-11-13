


import os
import re
from collections import Counter
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import time

np.random.seed(42)


CLEAN_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")

def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    lowercase_columns = []

    # lowercase column values
    for col in df.columns:
        lowercase_columns.append(col.lower())

    df.columns = lowercase_columns

    # extract review texts and sentiment labels
    return df["review"].astype(str).tolist(), df["sentiment"].tolist()


def train_test_split_data(review_texts, sentiment_labels):
    review_array = np.asarray(review_texts)
    label_array = np.asarray(sentiment_labels)

    indices = np.arange(review_array.shape[0])
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    midpoint = indices.shape[0] // 2
    train_indices = indices[:midpoint]
    test_indices = indices[midpoint:]

    train_texts = review_array[train_indices].tolist()
    test_texts = review_array[test_indices].tolist()
    train_labels = label_array[train_indices].tolist()
    test_labels = label_array[test_indices].tolist()

    return train_texts, test_texts, train_labels, test_labels


def clean_text(sentence):
    sentence = sentence.lower()
    sentence = CLEAN_RE.sub(" ", sentence)
    sentence = WHITESPACE_RE.sub(" ", sentence).strip()
    return sentence


def clean_dataset_texts(texts):
    series = pd.Series(texts, dtype="string")
    cleaned = (
        series.str.lower()
        .str.replace(CLEAN_RE, " ", regex=True)
        .str.replace(WHITESPACE_RE, " ", regex=True)
        .str.strip()
    )
    return cleaned.fillna("").tolist()

def extract_top_words(texts, max_words):
    freq = Counter()
    for text in texts:
        if isinstance(text, str):
            freq.update(text.split())
        else:
            freq.update(text)

    most_common_words = freq.most_common(max_words)
    vocab_words = [word for word, _ in most_common_words]

    return vocab_words, most_common_words


def tokenize_texts(cleaned_texts):
    return [text.split() for text in cleaned_texts]


def pad_truncate(sequence, max_length, pad_value=0):
    if len(sequence) >= max_length:
        return sequence[:max_length]
    return sequence + [pad_value] * (max_length - len(sequence))

# apply padding to sequences
def pad_sequences(sequences, max_lengths, pad_value=0):
    padded = []
    num_sequences = len(sequences)

    for max_length in max_lengths:
        arr = np.full((num_sequences, max_length), pad_value, dtype=np.int32)
        for index, sequence in enumerate(sequences):
            truncated_sequence = sequence[:max_length]
            arr[index, : len(truncated_sequence)] = truncated_sequence
        padded.append(arr)
    return padded

# convert tokens to ids
def token_to_ids(tokens, word_to_ids, oov=1):
    return [word_to_ids.get(token, oov) for token in tokens]

# get vocabulary with padding
def build_padded_vocab(vocab_words):
    padded_vocab = {"<PAD>": 0, "<OOV>": 1}
    for i, word in enumerate(vocab_words, start=2):
        padded_vocab[word] = i
        
    return padded_vocab


def preprocess_dataset(dataset_path, max_words, max_lengths):

    # read the data
    review_texts, sentiment_labels = read_dataset(dataset_path)

    # convert labels to binary
    label_map = {"negative": 0, "positive": 1}
    sentiment_labels = [1 if label == "positive" else 0 for label in sentiment_labels]


    # split into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split_data(review_texts, sentiment_labels)

    # clean annd tokenize the text
    cleaned_train_texts = clean_dataset_texts(train_texts)
    tokenized_train = tokenize_texts(cleaned_train_texts)

    cleaned_test_texts = clean_dataset_texts(test_texts)
    tokenized_test = tokenize_texts(cleaned_test_texts)

    # get the top words and pad the words (build vocab on train only)
    vocab_words, most_common_words = extract_top_words(tokenized_train, max_words)
    padded_vocab = build_padded_vocab(vocab_words)

    # build the sequences and pad them
    train_sequences = [token_to_ids(tokens, padded_vocab) for tokens in tokenized_train]
    test_sequences = [token_to_ids(tokens, padded_vocab) for tokens in tokenized_test]

    train_padded_sequences = pad_sequences(train_sequences, max_lengths=max_lengths)
    test_padded_sequences = pad_sequences(test_sequences, max_lengths=max_lengths)

    return {
        "train_texts": train_texts,
        "test_texts": test_texts,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "cleaned_train_texts": cleaned_train_texts,
        "cleaned_test_texts": cleaned_test_texts,
        "tokenized_train": tokenized_train,
        "tokenized_test": tokenized_test,
        "vocab": padded_vocab,
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "train_padded_sequences": train_padded_sequences,
        "test_padded_sequences": test_padded_sequences,
        "max_lengths": list(max_lengths),
    }
