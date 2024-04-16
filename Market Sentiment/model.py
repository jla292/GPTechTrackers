import torch.nn as nn
import torch
from tqdm import tqdm
import os
import pandas as pd
import torchtext
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
import numpy as np

# done?
class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hid_dim, final_dim, batch_size):
        super(LSTM, self).__init__()

        self.batch_size = batch_size
        self.hid_dim = hid_dim
        self.embedding_layer = nn.Embedding(input_dim, embedding_dim)
        # self.lstm_layer = nn.LSTM(input_size=embedding_dim, hidden_size=hid_dim, batch_first=True, bidirectional=False, num_layers=1)
        self.lstm_cell = nn.LSTMCell(input_size=embedding_dim, hidden_size=hid_dim)
        self.linear_proj = nn.Linear(2 * hid_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, final_dim)
        self.softmax = nn.Softmax(dim=1)
        self.act = nn.ReLU()

    def forward(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sent_len = []
        for sent in input:
          i = 0
          while i < len(sent) and sent[i] != 0:
            i += 1
          sent_len.append(i-1)

        embeds = self.embedding_layer(input)
        batch_size = input.size(0)
        h_0, c_0 = torch.randn(batch_size, self.hid_dim).to(device), torch.randn(batch_size, self.hid_dim).to(device)
        hid_states = []
        cell_states = []
        for i in range(input.shape[1]):
            h_0, c_0 = self.lstm_cell(embeds[:, i, :], (h_0, c_0))
            hid_states.append(h_0)
            cell_states.append(c_0)

        final_hid_states = []
        for s in range(len(sent_len)):
          last_hid = hid_states[sent_len[s]][s]
          last_cell = cell_states[sent_len[s]][s]
          final_hid_states.append(torch.cat((last_hid, last_cell)))
        final_hid_states = torch.stack(final_hid_states).to(device)

        x = self.linear_proj(final_hid_states)
        x = self.output_layer(x)
        # x = self.softmax(x)
        return x

# plan: read from csv to df, split into train/test/val, tokenize & pad, return all with vocab size
def format_dataloader(file_path_train, file_path_test):
    # load data
    df1 = pd.read_csv(file_path_train, engine='python')
    df2 = pd.read_csv(file_path_test, engine='python')

    X_train = df1["Title"] + " " + df1["Description"] # include both title and then description
    y_train = df1["Label"] # label encoding: 0 as bullish, 1 as bearish, 2 as neutral
    X_test = df2["Title"] + " " + df2["Description"] # include both title and then description
    y_test = df2["Label"] # label encoding: 0 as bullish, 1 as bearish, 2 as neutral

    # split into test and val data
    third = int(len(X_test) / 3)
    X_val = X_test.head(third)
    y_val = y_test.head(third)
    X_test = X_test.tail(third * 2).reset_index(drop=True)
    y_test = y_test.tail(third *2).reset_index(drop=True)

    # tokenize & pad continue
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    tokenized_data_train = [tokenizer(sentence) for sentence in X_train]
    unique_tokens = set(token for seq in tokenized_data_train for token in seq)
    vocab = {token: idx for idx, token in enumerate(unique_tokens)}

    X_train = [torch.tensor([vocab[token] for token in tokenized_sentence]) for tokenized_sentence in tokenized_data_train]
    # X_train_final = [len(tokenized_sentence) - 1 for tokenized_sentence in tokenized_data_train]
    vocab['<unk>'] = len(vocab)
    X_train = [torch.tensor([vocab.get(token, vocab['<unk>']) for token in tokenized_sentence]) for tokenized_sentence in tokenized_data_train]
    X_train_len = torch.tensor([len(tokenized_sentence) - 1 for tokenized_sentence in tokenized_data_train])
    X_train = rnn_utils.pad_sequence(X_train, batch_first=True)

    max_seq_length_train = max(len(seq) for seq in X_train)
    padded_data_train = rnn_utils.pad_sequence([torch.tensor(seq) for seq in X_train], batch_first=True)

    # input size
    vocab_size = len(vocab)

    # tokenize & pad continue
    tokenized_data_test = [tokenizer(sentence) for sentence in X_test]
    X_test = [torch.tensor([vocab.get(token, vocab['<unk>']) for token in tokenized_sentence]) for tokenized_sentence in tokenized_data_test]
    X_test_len = torch.tensor([len(tokenized_sentence) - 1 for tokenized_sentence in tokenized_data_test])
    X_test = rnn_utils.pad_sequence(X_test, batch_first=True)

    tokenized_data_val = [tokenizer(sentence) for sentence in X_val]
    X_val = [torch.tensor([vocab.get(token, vocab['<unk>']) for token in tokenized_sentence]) for tokenized_sentence in tokenized_data_val]
    X_val_len = torch.tensor([len(tokenized_sentence) - 1 for tokenized_sentence in tokenized_data_val])
    X_val = rnn_utils.pad_sequence(X_val, batch_first=True)

    return X_train, y_train, X_test, y_test, X_val, y_val, vocab_size, X_train_len, X_test_len, X_val_len

def train_model(
    model, train, val, num_epochs, optimizer, loss_fn, batch_size=32, file=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        total_loss = 0
        running_loss = 0.0
        model.train()
        total_batchs = 0
        for batch in tqdm(train, desc="Training Batches"):
            optimizer.zero_grad()
            input = batch[0].to(device)
            label = batch[1].to(device)

            output = model(input)
            # print(output)
            # print(label)
            # assert False
            loss = loss_fn(output, label)
            total_loss += loss.item()
            total_batchs += 1

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / total_batchs
        train_losses.append(epoch_loss)
        print(f"Training Loss: {total_loss / total_batchs}")

        val_loss = 0
        running_loss = 0.0
        total_batchs = 0
        model.eval()
        for batch in tqdm(val, desc="validation batches"):
            input = batch[0].to(device)
            label = batch[1].to(device)

            output = model(input)
            loss = loss_fn(output, label)
            val_loss += loss.item()
            running_loss += loss.item()
            total_batchs += 1
        epoch_loss = running_loss / len(train)
        val_losses.append(epoch_loss)

        print(f"Validation Loss: {val_loss / total_batchs}")
    if file:
        path = os.path.join(os.getcwd(), "models", file)
        model.save_pretrained(path)

    return num_epochs, train_losses, val_losses

def eval_model(model, test, loss_fn):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.eval()
  test_loss = 0
  total_batchs = 0
  outputs = []
  labels = []
  for batch in tqdm(test, desc="test batches"):
      input = batch[0].to(device)
      label = batch[1].to(device)

      output = model(input)

      loss = loss_fn(output, label)
      test_loss += loss.item()
      total_batchs += 1
      outputs.append(output)
      labels.append(label)
  print(f"Test Loss: {test_loss / total_batchs}")
  return outputs, labels, test_loss / total_batchs

def pred_model(model, input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    input = input.to(device)

    output = model(input)
    return output

def save_model(model, path):
  torch.save(model, path)

def load_model(path):
  return torch.load(path)