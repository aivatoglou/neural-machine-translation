import math
import pickle
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import MyData
from decoder import Decoder
from encoder import Encoder
from helpers import epoch_time, init_weights, max_length, pad_sequences, preprocessing
from seq import Seq2Seq
from train_val import evaluate, train
from vocabulary import LanguageIndex

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# !---------- hyperparameters ----------!
N_EPOCHS = 250
CLIP = 1
BATCH_SIZE = 128

# !---------- read dataset ----------!
dataset = pd.read_csv(
    "seq2seq/deu-eng/deu.txt",
    header=None,
    usecols=[0, 1],
    sep="\t",
    names=["English", "Deutch"],
)
dataset = dataset.sample(25000, random_state=0)

# !---------- data preprocessing ----------!
dataset["English"] = dataset["English"].apply(preprocessing)
dataset["Deutch"] = dataset["Deutch"].apply(preprocessing)

# !---------- words to idx ----------!
inp_lang = LanguageIndex(dataset["English"].values.tolist())
targ_lang = LanguageIndex(dataset["Deutch"].values.tolist())

# !---------- save vocabularies for inference ----------!
with open("seq2seq/vocab/en.pkl", "wb") as f:
    pickle.dump(inp_lang, f)
with open("seq2seq/vocab/de.pkl", "wb") as f:
    pickle.dump(targ_lang, f)

# !---------- idx to tensors ----------!
input_tensor = [
    [inp_lang.word2idx[s] for s in es.split(" ")[::-1]]  # src input reversed
    for es in dataset["English"].values.tolist()
]

# Keep <start> & <end> to default order
for sublist in input_tensor:
    temp = sublist[0]
    sublist[0] = sublist[-1]
    sublist[-1] = temp

target_tensor = [
    [targ_lang.word2idx[s] for s in eng.split(" ")]
    for eng in dataset["Deutch"].values.tolist()
]

# !---------- calculate max tensor ----------!
max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

# !---------- padding according to max ----------!
input_tensor = [pad_sequences(x, max_length_inp) for x in input_tensor]
target_tensor = [pad_sequences(x, max_length_tar) for x in target_tensor]

# !---------- train-val split ----------!
(
    input_tensor_train,
    input_tensor_val,
    target_tensor_train,
    target_tensor_val,
) = train_test_split(input_tensor, target_tensor, test_size=0.2)

# !---------- dataloder preperation ----------!
train_dataset = MyData(input_tensor_train, target_tensor_train)
valid_dataset = MyData(input_tensor_val, target_tensor_val)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True
)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True
)

INPUT_DIM = len(inp_lang.word2idx)
OUTPUT_DIM = len(targ_lang.word2idx)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

SRC_PAD_IDX = inp_lang.word2idx["<pad>"]
TRG_PAD_IDX = targ_lang.word2idx["<pad>"]

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
model.apply(init_weights)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

best_valid_loss = float("inf")
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP, device)
    valid_loss, output = evaluate(model, valid_dataloader, criterion, device)

    evaluate_sentence = [
        targ_lang.idx2word[w.item()] for w in output if w != 0 and w != 1 and w != 2
    ]
    evaluate_sentence = " ".join(evaluate_sentence)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "seq2seq/model/seq2seq.pt")

    print(
        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | {evaluate_sentence}"
    )
