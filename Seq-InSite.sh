#!/bin/bash
# Usage: bash Seq-InSite.sh [DATASET_FILE directory]


export User_Dir=$1
# Create appropriate directories:
DATASET_FILE=$User_Dir/dataset.txt
mkdir $User_Dir/testFeatures
mkdir $User_Dir/testFeatures/fasta/
mkdir $User_Dir/testFeatures/a3m/
mkdir $User_Dir/testFeatures/embd/
mkdir $User_Dir/testFeatures/t5Embd/
mkdir $User_Dir/output
fastaDir=$User_Dir/testFeatures/fasta/
a3mDir=$User_Dir//testFeatures/a3m/
embd=$User_Dir/testFeatures/embd/
t5Embd=$User_Dir/testFeatures/t5Embd/
output=$User_Dir/output/
# Create a file for each protein in fasta file
python Utils/fastaToFiles.py $DATASET_FILE $fastaDir

# Compute MSA file for each protein
bash Utils/gra3m.sh $fastaDir $a3mDir

# Calculate msa-transformer and T5 embeddings for each sequence
python Utils/T5Embd.py $DATASET_FILE $t5Embd
python Utils/msaEmbd.py $DATASET_FILE $a3mDir $embd

# Predict Seq-InSite 
python Architectures/predict_ENS.py $DATASET_FILE $embd $t5Embd $output
