import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from typing import Any, Dict, List, NewType
import csv
from ast import literal_eval as make_tuple
import argparse
from evaluation import evaluate
from utils import convert_MR_to_id, create_label_vocabulary

InputDataClass = NewType("InputDataClass", Any)

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
    processed_data = tokenizer(input_texts, padding=True, return_tensors = 'pt')
    if 'label' in features[0]:
        labels = [i["label"] for i in features]
        labels = torch.tensor(labels)
        processed_data['labels'] = labels
    return processed_data

def preprocess_data(data, test_idx, val_idx, train_idx):
    labeltoid, idtolabel = create_label_vocabulary(data, train_idx+val_idx)
    convert_MR_to_id(data, labeltoid)
    return labeltoid, idtolabel

def run(args):
    data = pd.read_csv(args.dataset)

    test_idx = [int(line.strip()) for line in open(args.test_ids)]
    val_idx = [int(line.strip()) for line in open(args.val_ids)]
    train_idx = [i for i in data.ID if i not in test_idx + val_idx]

    labeltoid, idtolabel = preprocess_data(data, test_idx, val_idx, train_idx)

    if args.language == 'it':
        MODEL_NAME = "roberta-base"
    elif args.language == 'de':
        MODEL_NAME = "roberta-base"
    elif args.language == 'en':
        MODEL_NAME = 'roberta-base'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    test_data = GeoQuery(data, test_idx, tokenizer)

    args_t = TrainingArguments(
        output_dir='models/',
        num_train_epochs=25,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        evaluation_strategy='epoch',
        learning_rate=1e-4,
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
        with torch.no_grad():
            logits = model(tokenizer(test_data[i]["x"], return_tensors='pt').input_ids).logits
            pred = logits.argmax(-1)

        gold = test_data[i]["gold"]
        pred = [idtolabel[i] for i in pred.cpu()[0].tolist()]
        preds.append(' '.join(pred))
        golds.append(' '.join(gold))

    stats = evaluate(preds, golds)

    if args.results_file is not None:
        with open(args.results_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Accuracy'])
            writer.writerow([stats['accuracy']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, help='input language (en, it, de)', required=True)
    parser.add_argument('--dataset', type=str, help='dataset path', required=True)
    parser.add_argument('--test-ids', type=str, help='test ids dataset path', required=True)
    parser.add_argument('--val-ids', type=str, help='val ids dataset path', required=True)
    parser.add_argument('--results-file', type=str, help='file path with results')
    args = parser.parse_args()

    run(args)

