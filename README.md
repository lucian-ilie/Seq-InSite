# The source code of Seq-InSite paper
The source code of Seq-InSite is optimized for high-throughput predictions and does not share the website's restriction of 10 sequences per run.
# Citation
S. Hosseini,  G.B. Golding, L. Ilie, Seq-InSite: sequence supersedes structure for protein interaction site prediction, submitted, 2023.

Contact: 

SeyedMohsen Hosseini (shosse59@uwo.ca)

Brian Golding (Golding@mcmaster.ca)

Lucian Ilie (ilie@uwo.ca)

# System requirement
Seq-InSite is developed under Linux environment with python 3.8.
Recommended RAM: > 24GB. The RAM requirement mainly depends on the length of the input sequence. 

# Installation
1. clone the source code of Seq-InSite
```
mkdir -p Src && cd Src
git clone [Seq-InSite git link]
```
2. Install msa-transformer in order to calculate msa-transformer embbedings

3. Install bio_embeddings packege in order to calculate T5 embeddings

4. install dependencies

 
    - install [hh-suite](https://github.com/soedinglab/hh-suite). The [database](http://gwdu111.gwdg.de/~compbiol/uniclust/2020_06/) used in Seq-InSite is uniref30_2020_06.
 
# Running Seq-InSite
There are 6 different model weights have been released, each corresponding to a different model, as described below: 

1. LSTM_T5_MSA_without*.h5 -LSTM architecture was trained using both T5 and msa-transformer's embeddings as input. This particular model was trained on data that does not share similarity with Dset_* where * is 60, 70, 315.
2. MLP_T5_MSA_without*.h5 -MLP architecture was trained using both T5 and msa-transformer's embeddings as input. This particular model was trained on data that does not share similarity with Dset_* where * is 60, 70, 315.


In order to run Seq-InSite use the following command 
```
bash Seq-InSite.sh [Fasta file directory]
```
Assuming that a file named "dataset.txt" is present in the given directory, this script will create the required files and directories for calculating alignments, embeddings, and the appropriate output directory. Finally, it will execute the "predict_ENS.py" file to predict the interactions. 
By default, the code will execute the ensemble version of Seq-InSite on the dataset that dissimilar from dset_60. If you desire to run a specific architecture of Seq-InSite, you must alter the predict file and provide the corresponding weights for that model.

If you already possess the appropriate embeddings, you may utilize the following command:

```
python predict_ENS.py /path/to/dataset /path/to/msa-embeddings /path/to/t5-embeddings /path/to/output
```
Please note that the accepted naming convention for embedding files is "PDBID.embd".
Each line of the embedding file must begin with the one-letter code for the corresponding amino acid, followed by a colon (:) symbol. The embedding representation of the amino acid should then be divided by spaces e.g.:
```
M:0.30833972 -0.17879489 -0.019303203 ...
A:0.32114908 -0.01173505 -0.1363031 ...
L:0.23623097 -0.295787 0.056586854 ...
```

# Training
In order to retrain the model you should use 'train_T5_MSA_LSTM.py' and 'train_T5_MSA_MLP.py'. If you have the embedding stored in the desired directory, you can employ the subsequent command to train each branch of the model.

```
python train_T5_MSA_MLP.py
python train_T5_MSA_LSTM.py
```


# Predictions

The Results directory encompass the predictions utilized in this paper. This directory contains the outputs of the methods employed in the research study, which were then utilized to draw conclusions. By putting the predictions in the Results directory, the researchers can easily access and analyze them for further investigation and evaluation.
