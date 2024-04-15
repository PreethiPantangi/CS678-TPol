from typing import Any, Dict, List, NewType
InputDataClass = NewType("InputDataClass", Any)
import torch

"""
This function _align_labels_with_tokenization checks the tokenized part of the text with the offset in the original text
and adjusts the label to match the tokenization so that each label is associated with the correct token.
"""
def _align_labels_with_tokenization(offset_mapping, labels, EPSILON_LABEL):
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
    x = [i["x"] for i in features]
    a = tokenizer(x, return_offsets_mapping=True, is_split_into_words=True, padding=True, return_tensors = 'pt')
    offset_mapping = a['offset_mapping']
    if 'label' in features[0]:
        labels = [i["label"] for i in features]
        labels = _align_labels_with_tokenization(offset_mapping, labels)
        labels = torch.tensor(labels)
        a['labels'] = labels
    del a['offset_mapping']
    return a

"""
The function create_label_vocabulary creates a mapping of the label and the id so that the during the training process the 
model understands and processes them. labeltoid and idtolabel are of type dictionary.
"""
def create_label_vocabulary(data, idx):
    seq = data.MR.tolist()
    seq = [seq[i] for i in range(len(seq)) if i in idx]

    labeltoid = {}
    idtolabel = {}
    count = 0

    labeltoid['ε'] = 0
    idtolabel[0] = 'ε'
    count = 1

    for i in seq:
        for j in i:
            if j not in labeltoid:
                labeltoid[j] = count
                idtolabel[count] = j
                count += 1
    return labeltoid, idtolabel

"""
The function convert_MR_to_id maps the labels in the data into numerical IDs so that the model can process the data efficiently
using numbers. 
"""
def convert_MR_to_id(data, labeltoid, EPSILON_LABEL):
    mr = data.MR.tolist()
    mr = [[labeltoid[j] if j in labeltoid else EPSILON_LABEL for j in i] for i in mr]
    data.MR = mr