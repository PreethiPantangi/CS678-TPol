import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from typing import Any, List, Dict, NewType
import csv
from evaluation import evaluate  # Importing the evaluate function
import argparse
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

InputDataClass = NewType("InputDataClass", Any)

EPSILON_LABEL = 0

class GeoQuery(Dataset):
    def __init__(self, dataframe, idx, tokenizer):
        self.dataframe = dataframe
        self.idx = idx
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        index = self.idx[i]
        item = {}
        item["x"] = self.tokenizer(self.dataframe.iloc[index]['NL'])
        item["label"] = self.dataframe.iloc[index]['MR']
        item["gold"] = self.dataframe.iloc[index]['GOLD']
        return item

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (ht, ct) = self.lstm(x_packed)
        output, _ = pad_packed_sequence(lstm_out, batch_first=True)
        final_output = self.fc(output)
        return final_output

def collate_fn(batch):
    batch_x = [item['x'] for item in batch]
    batch_labels = [item['label'] for item in batch]
    lengths = [len(x) for x in batch_x]
    padded_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=EPSILON_LABEL)
    return padded_x, padded_labels, lengths
def insert_deterministic_epsilon(data, idx):
    dic = create_dic_deterministic_epsilon(data, idx)
    nl, mr = insert_deterministic_epsilon_seq(data.NL.tolist(), data.MR.tolist(), data.ALIGNMENT.tolist(), dic)
    data.NL = nl
    data.MR = mr

def remove_epsilons(x):
    s = make_tuple(x)
    s = [j[1] for j in s if j[1]!='ε']
    return s

def preprocess_data(data, test_idx, val_idx, train_idx):
    data.MR = data["ALIGNMENT"].apply(preprocess_MR)
    data.NL = data["NL"].apply(preprocess_NL)
    labeltoid, idtolabel = create_label_vocabulary(data, train_idx+val_idx)
    insert_deterministic_epsilon(data, train_idx+val_idx)
    data["GOLD"] = data["ALIGNMENT"].apply(remove_epsilons)
    convert_MR_to_id(data, labeltoid, EPSILON_LABEL)
    return labeltoid, idtolabel
def evaluate_model(model, dataset, device):
    model.eval()
    preds = []
    golds = []
    mn_labels = []
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=32, collate_fn=collate_fn):
            inputs, labels, lengths = data
            inputs = inputs.to(device)
            outputs = model(inputs, lengths)
            outputs = outputs.argmax(dim=2).cpu().tolist()
            for idx, length in enumerate(lengths):
                pred = outputs[idx][:length]
                gold = labels[idx][:length]
                preds.append(' '.join([str(p) for p in pred]))
                golds.append(' '.join([str(g) for g in gold]))
                mn_labels.append(dataset.dataframe.iloc[dataset.idx[idx]]['MONOTONIC'])  # Assuming monotonic labels are stored here
    return evaluate(preds, golds, mn_labels, verbose=True)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = pd.read_csv(args.dataset)
    preprocess_data(data)  # Ensure this adjusts data as needed for your context

    tokenizer = lambda x: torch.tensor([int(token) for token in x.split()], dtype=torch.long)  # Adjust tokenizer as needed

    train_idx = [i for i in range(len(data)) if i not in args.test_ids + args.val_ids]
    train_dataset = GeoQuery(data, train_idx, tokenizer)
    val_dataset = GeoQuery(data, args.val_ids, tokenizer)
    test_dataset = GeoQuery(data, args.test_ids, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab_size = 1000  # Modify based on actual vocabulary size
    embedding_dim = 128
    hidden_dim = 256
    num_labels = len(set(data['MR'].explode()))

    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_labels).to(device)

    # Train model
    train_model(model, train_loader, val_loader, device)

    # Evaluate model on test dataset
    evaluate_model(model, test_dataset, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, help='input language (en, it, de)', required=True)
    parser.add_argument('--dataset', type=str, help='dataset path', required=True)
    parser.add_argument('--test-ids', type=str, help='test ids dataset path', required=True)
    parser.add_argument('--val-ids', type=str, help='val ids dataset path', required=True)
    parser.add_argument('--out-file', type=str, help='out file path', required=True)
    parser.add_argument('--results-file', type=str, help='file path with results')
    parser.add_argument('--all-predictions-file', type=str, help='out file path of predictions for all sequences of the dataset')
    args = parser.parse_args()

    # Implement main function handling the parsed args
    main(args)

