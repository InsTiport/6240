#!/bin/bash

# set environment 

conda env create -f environment.yml
conda activate hgru4rec

# fetch dataset
cd data
wget http://snap.stanford.edu/jodie/reddit.csv
cd ..
python gen_embedding_file.py

python build_dataset.py data/train_interactions_embed.tsv full_train_embed
python build_dataset.py data/test_interactions_embed.tsv test_embed
