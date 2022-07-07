import re
import unicodedata

import numpy as np
import torch.nn as nn


def convert_to_ascii(sentence):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", sentence)
        if unicodedata.category(c) != "Mn"
    )


def preprocessing(sentence):

    out = convert_to_ascii(sentence.lower().strip())
    out = re.sub(r'[" "]+', " ", out)
    out = re.sub(r"[^a-zA-ZäöüÄÖÜß]+", " ", out)
    out = out.rstrip().strip()
    out = "<start> " + out + " <end>"

    return out


def max_length(tensor):
    return max(len(t) for t in tensor)


def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len:
        padded[:] = x[:max_len]
    else:
        padded[: len(x)] = x
    return padded


def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    # transpose (batch x seq) to (seq x batch)
    return X.transpose(0, 1), y, lengths


def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
