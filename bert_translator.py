import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, BertForTokenClassification, Trainer, TrainingArguments
from typing import Any, Dict, List, NewType
import csv
from evaluation import evaluate
from ast import literal_eval as make_tuple
import argparse
InputDataClass = NewType("InputDataClass", Any)
from bert_utils import convert_MR_to_id, create_label_vocabulary

EPSILON_LABEL = 0

class GeoQuery(Dataset):
    def __init__(self, dataframe, idx, tokenizer):
        all_nls = dataframe.NL.tolist()
        all_mrs = dataframe.MR.tolist()
        all_golds = dataframe.GOLD.tolist()
        self.nls = [all_nls[i] for i in range(len(all_nls)) if i in idx]
        self.mrs = [all_mrs[i] for i in range(len(all_mrs)) if i in idx]
        self.golds = [all_golds[i] for i in range(len(all_golds)) if i in idx]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.nls)

    def __getitem__(self, i):
        item = {}
        item["x"] = self.nls[i]
        item["label"] = self.mrs[i]
        item["tokenizer"] = self.tokenizer
        item["gold"] = self.golds[i]
        return item

"""
This function _align_labels_with_tokenization checks the tokenized part of the text with the offset in the original text
and adjusts the label to match the tokenization so that each label is associated with the correct token.
"""
def _align_labels_with_tokenization(offset_mapping, labels):
    new_labels = []
    for o, l in zip(offset_mapping, labels):
        lab = []
        count = -1
        for i in o:
            if i[0]==0 and i[1]==0:
                lab.append(EPSILON_LABEL)
            elif i[0] == 0:
                count += 1
                lab.append(l[count])
            else:
                lab.append(EPSILON_LABEL)
        new_labels.append(lab)
    return new_labels

"""
The function bert_classifier_data_collator takes a list of features such as the text and labels and aligns the labels with the 
tokenization and arranges the data in a format that is suitable to be passed to the BERT model for training. 
"""
def bert_classifier_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    tokenizer = features[0]['tokenizer']
    input_texts = [i["x"] for i in features]
    processed_data = tokenizer(input_texts, return_offsets_mapping=True, is_split_into_words=True, padding=True, return_tensors = 'pt')
    offset_mapping = processed_data['offset_mapping']
    if 'label' in features[0]:
        labels = [i["label"] for i in features]
        labels = _align_labels_with_tokenization(offset_mapping, labels)
        labels = torch.tensor(labels)
        processed_data['labels'] = labels
    del processed_data['offset_mapping']
    return processed_data # The return type of a here is a dictionary that contains all the information and requirements of the BERT model.

"""
The function preprocess_MR prepares the MR text in such a way that it can be easily processed by the BERT model.
"""
def preprocess_MR(text):
    s = make_tuple(text)
    s = [j[1] for j in s]
    return s

def preprocess_NL(text):
    return text.split(" ")

"""
The function find_epsilon_length calculates the length of the sequence of the epsilon labels from a given index. 
The epsilon values represent the empty or null values and this step is important to eleminate these irrelavent values from the 
data for training.
"""
def find_epsilon_length(x, y):
    count = 0
    for i in range(y, len(x)):
        if x[i] == 'ε':
            count += 1
        else:
            break
    return count

"""
The function create_dic_deterministic_epsilon generates a dictionary mapping non-empty labels to the length of the sequences 
of epsilon labels. It identifies and records the length of these sequences to manage deterministic occurrences of epsilon 
labels in the dataset. This ensures that the model can correctly recognize and deal with the presence of empty 
or optional elements in the dataset.
"""
def create_dic_deterministic_epsilon(data, idx):
    seq = data.ALIGNMENT.tolist()
    seq = [[j[0] for j in make_tuple(i)] for i in seq]
    seq = [seq[i] for i in range(len(seq)) if i in idx]

    dic = {}
    idaa= 0
    for i in seq:
        idaa += 1
        for j in range(len(i)-1):
            if i[j+1] == 'ε':
                if i[j] != 'ε':
                    dic[i[j]] = find_epsilon_length(i, j+1)
                    
    return dic   

"""
The function insert_deterministic_epsilon_seq modifies the sequences so that the deterministic patterns of epsilon labels 
identified in the alignment data can maintain alignment integrity for subsequent processing.
"""
def insert_deterministic_epsilon_seq(x, y, alignments, dic ): 
    new_x = []
    new_y = []
    al = [[j[0] for j in make_tuple(i)] for i in alignments]
    for a, b, c in zip(x,y,al):
        nx = []
        ny = []
        a_count = 0
        for i in range(len(c)):
            if c[i] == 'ε':
                continue
            elif c[i] in dic:
                nx.append(a[a_count])
                a_count += 1
                ny.append(b[i])

                for j in range(dic[c[i]]): 
                    if i+1+j < len(c):
                        if c[i+1+j] == 'ε':
                            nx.append('ε')
                            ny.append(b[i+1+j])    
                        else:
                            for _ in range(j, dic[c[i]]):
                                nx.append('ε')
                                ny.append('ε')
                            break
                    else:
                        nx.append('ε')
                        ny.append('ε')
            else:
                nx.append(a[a_count])
                a_count += 1
                ny.append(b[i])

        new_x.append(nx)
        new_y.append(ny)

    return new_x, new_y

"""The function insert_deterministic_epsilon helps in handling empty elements within the data, improving the dataset's usability for training and 
evaluation."""
def insert_deterministic_epsilon(data, idx):
    dic = create_dic_deterministic_epsilon(data, idx)
    nl, mr = insert_deterministic_epsilon_seq(data.NL.tolist(), data.MR.tolist(), data.ALIGNMENT.tolist(), dic)
    data.NL = nl
    data.MR = mr

"""The function remove_epsilons removes empty or null values so that the model can focus on the relevant parts of the data"""
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

def run(args):

    data = pd.read_csv(args.dataset)

    test_idx = [int(line.strip()) for line in open(args.test_ids)]
    val_idx = [int(line.strip()) for line in open(args.val_ids)]
    train_idx = [i for i in data.ID if i not in test_idx + val_idx]

    test_idx.sort()
    val_idx.sort()

    labeltoid, idtolabel = preprocess_data(data, test_idx, val_idx, train_idx)

    if args.language == 'it':
        MODEL_NAME = "dbmdz/bert-base-italian-uncased"
    elif args.language == 'de':
        MODEL_NAME = "dbmdz/bert-base-german-uncased"
    elif args.language == 'en':
        MODEL_NAME = 'bert-base-uncased'

    config=AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = len(labeltoid)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, config=config)

    test_data = GeoQuery(data, test_idx, tokenizer)
    
    args_t = TrainingArguments(
        output_dir = 'models/',
        num_train_epochs = 25,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        evaluation_strategy = 'epoch',
        learning_rate = 1e-4,
        weight_decay=0.01,
        save_total_limit=3,
        push_to_hub=False,
    )

    trainer = Trainer(
        data_collator=bert_classifier_data_collator,
        model=model,                         
        args=args_t,                 
        train_dataset=GeoQuery(data,train_idx,tokenizer),     
        eval_dataset=GeoQuery(data,val_idx,tokenizer),
    )    

    trainer.train()    

    preds = []
    golds = []
    for i in range(len(test_data)):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      with torch.no_grad():
        logits = model(tokenizer(test_data[i]["x"], is_split_into_words=True, return_tensors = 'pt').to(device)["input_ids"]).logits
        pred = logits.argmax(-1)

      gold = test_data[i]["gold"]
      pred = [idtolabel[i] for i in pred.cpu()[0].tolist()]
      pred = [j for j in pred if j != 'ε']

      preds.append(' '.join(pred))
      golds.append(' '.join(gold))
    

    monotonic = data.MONOTONIC.tolist()
    monotonic = [monotonic[i] for i in range(len(monotonic)) if i in test_idx]

    nls = data.NL.tolist()
    test_nls = [nls[i] for i in test_idx]

    new_df = pd.DataFrame({"ID":test_idx, "NL":test_nls, "MR": golds, "PRED": preds, "MONOTONIC": monotonic})
    new_df.to_csv(args.out_file)

    stats = evaluate(preds, golds, monotonic, verbose=True)

    if args.results_file is not None:
        with open(args.results_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['exact', 'exact_mn', 'exact_nmn', 'tokens', 'tokens_mn', 'tokens_nmn', 'span', 'span_mn', 'span_nmn'])
            writer.writerow([stats['exact_match']['acc'], stats['exact_match']['mn_acc'], stats['exact_match']['nmn_acc'],
                            stats['no_correct_tokens']['acc'], stats['no_correct_tokens']['mn_acc'], stats['no_correct_tokens']['nmn_acc'],
                            stats['max_correct_span']['acc'], stats['max_correct_span']['mn_acc'], stats['max_correct_span']['nmn_acc'],])

    
    if args.all_predictions_file is not None:
        all_idx = [i for i in data.ID]
        all_data = GeoQuery(data, all_idx, tokenizer)
        preds = []
        gold = []
        for i in range(len(all_data)):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                logits = model(tokenizer(all_data[i]["x"], is_split_into_words=True, return_tensors = 'pt').to(device)["input_ids"]).logits
                pred = logits.argmax(-1)

            gold = all_data[i]["gold"]
            pred = [idtolabel[i] for i in pred.cpu()[0].tolist()]
            pred = [j for j in pred if j != 'ε']

            preds.append(' '.join(pred))
            golds.append(' '.join(gold))
        
        new_df = pd.DataFrame({"ID":data.ID, "PRED_ALIGNMENT": preds})
        new_df.to_csv(args.all_predictions_file)

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

    run(args)
