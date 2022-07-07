import pickle
import sys

import numpy as np
import pandas as pd
import torch

from seq2seq.training import vocabulary
from seq2seq.training.decoder import Decoder
from seq2seq.training.encoder import Encoder
from seq2seq.training.seq import Seq2Seq

sys.modules["vocabulary"] = vocabulary

from seq2seq.training.helpers import pad_sequences, preprocessing


class SeqTranslator:
    def __init__(self):

        f = open("seq2seq/vocab/en.pkl", "rb")
        self.inp_lang = pickle.load(f)

        f = open("seq2seq/vocab/de.pkl", "rb")
        self.targ_lang = pickle.load(f)

        INPUT_DIM = len(self.inp_lang.word2idx)
        OUTPUT_DIM = len(self.targ_lang.word2idx)
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        N_LAYERS = 2
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5
        SRC_PAD_IDX = self.inp_lang.word2idx["<pad>"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

        self.sent = "None"
        self.max_length_inp = 41  # taken from training process
        self.model = Seq2Seq(enc, dec, self.device).to(self.device)
        self.model.load_state_dict(
            torch.load("seq2seq/model/seq2seq.pt", map_location=torch.device("cpu"))
        )
        self.model.eval()

    def translate_sentence(self, input, model, device):

        src_tensor = torch.LongTensor(np.array(input)).to(device)
        src_tensor = src_tensor.reshape(src_tensor.size(1), 1)

        with torch.no_grad():
            hidden, cell = model.encoder(src_tensor)

        trg_indexes = [self.targ_lang.word2idx["<start>"]]

        for i in range(self.max_length_inp):

            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

            with torch.no_grad():
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            if pred_token == self.targ_lang.word2idx["<end>"]:
                break

        trg_tokens = [self.targ_lang.idx2word[w] for w in trg_indexes]

        return trg_tokens

    def translate(self, input_text):

        df = pd.DataFrame({"English": [input_text]})
        df = df["English"].apply(preprocessing)
        print(df)
        try:
            input_tensor = [
                [self.inp_lang.word2idx[s] for s in es.split(" ")]
                for es in df.values.tolist()
            ]
            input_tensor = [pad_sequences(x, self.max_length_inp) for x in input_tensor]
            print(input_tensor)
            translation = self.translate_sentence(input_tensor, self.model, self.device)
            # translation.pop(0)   # <start> token
            # translation.pop(-1)  # <end> token
            self.sent = " ".join(translation)
        except KeyError as e:
            self.sent = f"Key {e} not found in vocabulary."
