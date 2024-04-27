import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from typing import Any, Dict, List, NewType
import csv
from ast import literal_eval as make_tuple
import argparse
InputDataClass = NewType("InputDataClass", Any)
from evaluation import evaluate
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

def roberta_classifier_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    tokenizer = features[0]['tokenizer']
    input_texts = [i["x"] for i in features]
    processed_data = tokenizer(input_texts, return_offsets_mapping=True, is_split_into_words=True, padding=True, return_tensors='pt')
    offset_mapping = processed_data['offset_mapping']
    
    if 'label' in features[0]:
        labels = [i["label"] for i in features]
        # Adjust labels to match tokenization
        new_labels = []
        for label, offsets in zip(labels, offset_mapping):
            new_label = []
            idx = 0
            for offset in offsets:
                if offset[0] == 0 and idx < len(label):
                    new_label.append(label[idx])
                    idx += 1
                else:  # Padding token
                    new_label.append(EPSILON_LABEL)
            new_labels.append(new_label)
        labels = torch.tensor(new_labels)
        processed_data['labels'] = labels

    del processed_data['offset_mapping']
    return processed_data

def preprocess_MR(text):
    s = make_tuple(text)
    s = [j[1] for j in s]
    return s

def preprocess_NL(text):
    return text.split(" ")

def remove_epsilons(x):
    s = make_tuple(x)
    s = [j[1] for j in s if j[1]!='ε']
    return s

def preprocess_data(data, test_idx, val_idx, train_idx):
    data.MR = data["ALIGNMENT"].apply(preprocess_MR)
    data.NL = data["NL"].apply(preprocess_NL)
    labeltoid, idtolabel = create_label_vocabulary(data, train_idx+val_idx)
    data["GOLD"] = data["ALIGNMENT"].apply(remove_epsilons)
    convert_MR_to_id(data, labeltoid)
    return labeltoid, idtolabel

def run(args):
    data = pd.read_csv(args.dataset)
    test_idx = [int(line.strip()) for line in open(args.test_ids)]
    val_idx = [int(line.strip()) for line in open(args.val_ids)]
    train_idx = [i for i in data.ID if i not in test_idx + val_idx]
    test_idx.sort()
    val_idx.sort()

    labeltoid, idtolabel = preprocess_data(data, test_idx, val_idx, train_idx)

    MODEL_NAME = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(labeltoid))

    test_data = GeoQuery(data, test_idx, tokenizer)
    
    args_t = TrainingArguments(
        output_dir='models/',
        num_train_epochs=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=3,
        push_to_hub=False,
    )

    trainer = Trainer(
        data_collator=roberta_classifier_data_collator,
        model=model,
        args=args_t,
        train_dataset=GeoQuery(data, train_idx, tokenizer),
        eval_dataset=GeoQuery(data, val_idx, tokenizer),
    )

    trainer.train()

    preds = []
    golds = []
    for i in range(len(test_data)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            logits = model(tokenizer(test_data[i]["x"], is_split_into_words=True, return_tensors='pt').to(device)["input_ids"]).logits
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
    parser.add_argument('--dataset', type=str, help='dataset path', required=True)
    parser.add_argument('--test-ids', type=str, help='test ids dataset path', required=True)
    parser.add_argument('--val-ids', type=str, help='val ids dataset path', required=True)
    parser.add_argument('--out-file', type=str, help='out file path', required=True)
    parser.add_argument('--results-file', type=str, help='file path with results')
    parser.add_argument('--all-predictions-file', type=str, help='out file path of predictions for all sequences of the dataset')
    args = parser.parse_args()

    run(args)

