# The source code of Seq-InSite paper
The source code of Seq-InSite is optimized for high-throughput predictions and does not share the website's restriction of 10 sequences per run.
# Citation
S. Hosseini,  G. Brian Golding, L. Ilie, Seq-InSite: sequence supersedes structure for protein
interaction site prediction

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
Seq-InSite consists of 7 distinct architectures, each designed for predicting interactions. These are:

1. predict_MSA_MLP.py - MLP architecture utilizing msa-transformer's embeddings as input.
2. predict_T5_MLP.py - MLP architecture utilizing T5's embeddings as input.
3. predict_MSA_LSTM.py - LSTM architecture utilizing msa-transformer's embeddings as input.
4. predict_T5_LSTM.py - LSTM architecture utilizing T5's embeddings as input.
5. predict_T5_MSA_MLP.py -MLP architecture utilizing both T5 and msa-transformer's embeddings as input.
6. predict_T5_MSA_LSTM.py -LSTM architecture utilizing both T5 and msa-transformer's embeddings as input.
7. predict_ENS.py - The ensemble model.

Additionally, 12 different model weights have been released, each corresponding to a different model, as described below: 

1. MLP_MSA_L9.h5 -MLP architecture was trained on TR dataset using msa-transformer's embeddings as input. This particular model was trained on data that does not share similarity with Dset_448, Dset_164, Dset_186, and Dset_72.
2. MLP_T5U50_L9.h5 -MLP architecture was trained on TR dataset using T5's embeddings as input. This particular model was trained on data that does not share similarity with Dset_448, Dset_164, Dset_186, and Dset_72.
3. MLP_T5_MSA_L9.h5 -MLP architecture was trained on TR dataset using both T5 and msa-transformer's embeddings as input. This particular model was trained on data that does not share similarity with Dset_448, Dset_164, Dset_186, and Dset_72.
4. LSTM_T5_MSA_L9.h5 -LSTM architecture was trained on TR dataset using both T5 and msa-transformer's embeddings as input. This particular model was trained on data that does not share similarity with Dset_448, Dset_164, Dset_186, and Dset_72.
5. LSTM_MSA_L9.h5 -LSTM architecture was trained on TR dataset using msa-transformer's embeddings as input. This particular model was trained on data that does not share similarity with Dset_448, Dset_164, Dset_186, and Dset_72.
6. LSTM_T5U50_L9.h5 -LSTM architecture was trained on TR dataset using T5's embeddings as input. This particular model was trained on data that does not share similarity with Dset_448, Dset_164, Dset_186, and Dset_72.
7. LSTM_T5_MSA_without*.h5 -LSTM architecture was trained using both T5 and msa-transformer's embeddings as input. This particular model was trained on data that does not share similarity with Dset_* where * is 60, 70, 315.
8. MLP_T5_MSA_without*.h5 -MLP architecture was trained using both T5 and msa-transformer's embeddings as input. This particular model was trained on data that does not share similarity with Dset_* where * is 60, 70, 315.


In order to run Seq-InSite use the following command 
```
bash Seq-InSite.sh [Fasta file directory]
```

The default behavior of the code is to run the ensemble version of Seq-InSite on the TR dataset. If you want to run a specific architecture of Seq-InSite, you will need to modify the predict file and provide the appropriate weights for that model.

For example to run the LSTM branch that uses both msa-transformer and T5 embeddings, which was trained on data dissimilar to Dset_60, you should use the 'predict_T5_MSA_LSTM.py' file. Additionally, you will need to modify the code to utilize the 'LSTM_T5_MSA_without60.h5' weights.

# Predictions

The Results directory encompass the predictions utilized in this paper. This directory contains the outputs of the methods employed in the research study, which were then utilized to draw conclusions. By putting the predictions in the Results directory, the researchers can easily access and analyze them for further investigation and evaluation.
