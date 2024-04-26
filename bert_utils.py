from typing import Any, NewType
InputDataClass = NewType("InputDataClass", Any)

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
def convert_MR_to_id(data, labeltoid, EPSILON_LABEL=0):
    mr = data.MR.tolist()
    mr = [[labeltoid[j] if j in labeltoid else EPSILON_LABEL for j in i] for i in mr]
    data.MR = mr
