# CS 678 Final Project - TPol - Translate First Reorder Later
The paper we picked for this project is from Findings of the Association for Computational Linguistics: EMNLP 2023 
Paper title - [Translate First Reorder Later: Leveraging Monotonicity in Semantic Parsing](https://arxiv.org/pdf/2210.04878.pdf).


## Installation
We first create a Conda environment using the tpol-environment.yml file and activate it using the following commands:
```
conda env create -f tpol-environment.yml
conda activate tpol-private
```

## Instructions

The heart of TPol model uses either bert or mbart model as its architecture. The order of execution is translator and then reorderer. 

1. Running TPol with bert model and english language.
2. To run the TPol with bert based model for translator we run the command 
```
python bert_translator.py --language en --dataset "path/to/data/GEO-Aligned/data/EN.csv" --test-ids "path/to/data/GEO-Aligned/splits/length/test.txt" --val-ids "path/to/data/GEO-Aligned/splits/length/dev1.txt" --out-file "path/to/bert-translator-output.txt" --results-file "path/to/bert-translator-results.txt"
```

3. To run the TPol with bert based model for reorderer we run the command 
```
python bert_reorderer.py --language en --dataset "path/to/data/GEO-Aligned/data/EN.csv" --test-ids "path/to/data/GEO-Aligned/splits/length/test.txt" --val-ids "path/to/data/GEO-Aligned/splits/length/dev1.txt" --out-file "path/to/mbart-reorderer-output.txt" --results-file "path/to/mbart-reorderer-results.txt" --lexical-predictions "path/to/bert-translator-output.txt"
```

## Citation
```
@inproceedings{locatelli-quattoni-2022-measuring,
    title = "Translate First Reorder Later: Leveraging Monotonicity in Semantic Parsing",
    author = "Cazzaro, Francesco   and Locatelli, Davide" and Quattoni, Ariadna"
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = Feb,
    year = "2023",
    pages = "227--238"
}
```
