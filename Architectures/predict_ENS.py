import sys
from tensorflow import keras
from keras.models import *
from keras.layers import *
import tensorflow as tf
import keras.backend as K

from keras.layers import LSTM, Dense, Flatten, Reshape, TimeDistributed, Bidirectional, CuDNNLSTM, CuDNNGRU, GRU, \
    Dropout, Input, Conv2D, MaxPool2D, ConvLSTM2D, SpatialDropout2D, Conv1D, MaxPool1D, Concatenate, BatchNormalization, \
    Activation, AveragePooling2D, Embedding, MultiHeadAttention, Lambda, GlobalMaxPooling1D


import numpy as np
from keras.models import load_model
from collections import defaultdict


LOCAL_NEGHBOR_SIZE = 4 
WINDOW_SIZE = 1 + 2*LOCAL_NEGHBOR_SIZE


def protToDict(datasetAddress, embdAddress):
    protDict =  defaultdict(dict)
    dataset_file = open(datasetAddress)
    while True:
        line_PID = dataset_file.readline().strip()[1:]
        line_Pseq = dataset_file.readline().strip()
        #line_feature = dataset_file.readline().strip()
        if not line_Pseq:
            break
        #if len(line_Pseq) < 1024:
        prot_file = open('{}/{}.embd'.format(embdAddress, line_PID))
        for index, prot_line in enumerate(prot_file):
            #print(line_PID)
            prot_line = prot_line.strip().split(':')[1]
            embd_value = [float(x) for x in prot_line.split()]
            protDict[line_PID][index] = embd_value
    return protDict




def readFeatures2D(neighborList,proteinName, protDict, protDictMSA):
    selectedFeature = []
    selectedFeatureMSA = []
    for neighbor in neighborList:
        if neighbor != 'Zpad':
            try:
                selectedFeature.append(np.array(protDict[proteinName][neighbor]))
                selectedFeatureMSA.append(np.array(protDictMSA[proteinName][neighbor]))
            except:
                print(proteinName, neighbor)
                print(protDict[proteinName])
                exit(1)
        else:
                selectedFeature.append(np.zeros(1024).astype('float32'))
                selectedFeatureMSA.append(np.zeros(768).astype('float32'))

    return np.array(selectedFeature), np.array(selectedFeatureMSA)
    

def build2DWindows(aaIndex, protLen, proteinName, protDict, protDictMSA, windowSize=WINDOW_SIZE):
    proteinLenght = protLen
    addToLeft = 0
    addToRight = 0
    if aaIndex <= LOCAL_NEGHBOR_SIZE:
        addToLeft = LOCAL_NEGHBOR_SIZE - aaIndex
    if aaIndex+LOCAL_NEGHBOR_SIZE + 1 > proteinLenght:
        addToRight = aaIndex+LOCAL_NEGHBOR_SIZE + 1 - proteinLenght

    neighborList = [i for i in range(aaIndex-LOCAL_NEGHBOR_SIZE + addToLeft, aaIndex+LOCAL_NEGHBOR_SIZE+1 - addToRight)]
    #print(neighborList)
    cnt = 0
    lrFlag = 0 + addToRight - addToLeft
    for i in range(100):
        if cnt < addToRight + addToLeft:
            if lrFlag <= 0:
                neighborList = ['Zpad'] + neighborList
                lrFlag +=1
                cnt +=1
            else:
                neighborList = neighborList + ['Zpad']
                lrFlag -= 1
                cnt +=1
        else:
            break
    
    Features2D, Features2DMSA = readFeatures2D(neighborList, proteinName, protDict, protDictMSA)
    #print(Features2D.shape)
    return Features2D, np.array(Features2DMSA)
    




def readSort(datasetAddress, embd_files, t5Embd_files):
    features3D = []
    features3DMSA = []
    labels = []
    protDict = protToDict(datasetAddress, t5Embd_files)
    protDictMSA = protToDict(datasetAddress, embd_files)
    dataset_file = open(datasetAddress, 'r')
    while True:
        line_PID = dataset_file.readline().strip()
        line_Pseq = dataset_file.readline().strip()
        #line_feature = dataset_file.readline().strip()
        if not line_Pseq:
            break
        protName = line_PID[1:]
        for aaIndex in range(len(line_Pseq)):
            feTmp, feMsa = build2DWindows(aaIndex, len(line_Pseq), protName, protDict, protDictMSA)
            features3D.append(feTmp)
            features3DMSA.append(feMsa)
    print(np.array(features3D).shape)
    return np.array(features3D), np.array(features3DMSA)




def Predict(test_all_features_np3D, input_file, output_dir):

    #MLP    
    input_features = Input(shape=((int)(WINDOW_SIZE), 1024), name="input_ens_1")
    input_features2 = Input(shape=((int)(WINDOW_SIZE), 768), name="input_ens_2")
    out2 = Dense(1024, activation='relu', name="dense_msa_0")(input_features)
    out2 = Dropout(rate=0.3)(out2)
    out2 = Flatten()(out2)
    out3 = Dense(768, activation='relu', name="dense_t5_0")(input_features2)
    out3 = Dropout(rate=0.3)(out3)
    out3 = Flatten()(out3)
    concatenated = Concatenate()([out3, out2])
    out3 = Flatten()(concatenated)
    out3 = Dense(256, activation='relu', name="dense_com_1")(out3)
    out3 = Dropout(rate=0.3)(out3)
    out3 = Dense(128, activation='relu', name="dense_com_2")(out3)
    out3 = Dropout(rate=0.3)(out3)
    out3 = Dense(16, activation='relu', name="dense_com_3")(out3)
    out3 = Dropout(rate=0.3)(out3)
    out3 = Dense(1, activation='sigmoid', name="dense_com_4")(out3)
    
        
    model = keras.models.Model(inputs=[input_features,input_features2], outputs=out3)
    model.load_weights("Models/MLP_T5_MSA_without60.h5") 
    y_pred_testing = model.predict(test_all_features_np3D, batch_size=1024).ravel()


    input_features = Input(shape=((int)(WINDOW_SIZE), 1024), name="input_ens_1")
    input_features2 = Input(shape=((int)(WINDOW_SIZE), 768), name="input_ens_2")
    out2 = Bidirectional(
            LSTM(name="lstm_T5", activation="tanh", recurrent_activation="sigmoid", units=64,
                return_sequences=True, unroll=False, use_bias=True, recurrent_dropout=0.0),
            name="bidirectional_T5")(input_features)
    out3 = Bidirectional(
            LSTM(name="lstm_MSA", activation="tanh", recurrent_activation="sigmoid", units=64,
                return_sequences=True, unroll=False, use_bias=True, recurrent_dropout=0.0),
            name="bidirectional_MSA")(input_features2)
    out2 = Flatten()(out2)
    out2 = Dense(1024, activation='relu', name="dense_LSTM_msa_0")(out2)
    out2 = Dropout(rate=0.3)(out2)
    out3 = Flatten()(out3)
    out3 = Dense(768, activation='relu', name="dense_LSTM_t5_0")(out3)
    out3 = Dropout(rate=0.3)(out3)
    concatenated = Concatenate(name='LSTM_concat')([out3, out2])
    out3 = Flatten()(concatenated)
    out3 = Dense(256, activation='relu', name="dense_LSTM_com_1")(out3)
    out3 = Dropout(rate=0.3)(out3)
    out3 = Dense(128, activation='relu', name="dense_LSTM_com_2")(out3)
    out3 = Dropout(rate=0.3)(out3)
    out3 = Dense(16, activation='relu', name="dense_LSTM_com_3")(out3)
    out3 = Dropout(rate=0.3)(out3)
    out3 = Dense(1, activation='sigmoid', name="dense_LSTM_com_4")(out3)

    model = keras.models.Model(inputs=[input_features,input_features2], outputs=out3)
    model.load_weights("Models/LSTM_T5_MSA_without60.h5") 
    y_pred_testingRNN = model.predict(test_all_features_np3D, batch_size=1024).ravel()
    

    

    # load input proteins again and output the predict values 
    start_index = 0
    fin = open(input_file, "r")
    while True:
        line_PID = fin.readline()[1:].rstrip('\n').rstrip(' ')
        line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
        #line_feature = fin.readline().rstrip('\n').rstrip(' ')
        if not line_Pseq:
            break
        fout = open("{}/{}.txt".format(output_dir, line_PID.upper()), "w")
        
        for i in range(len(line_Pseq)):
            fout.write(str((y_pred_testing[start_index + i] + y_pred_testingRNN[start_index + i])/2) + "\n")
            #fout.write(str((y_pred_testingCNN[start_index + i] + y_pred_testingTF[start_index + i])/2) + "\n")
        fout.close()
        start_index += len(line_Pseq)
    fin.close()







def main():
    #input_file = 'dataset/Dset_315_Pid_Pseq.txt'
    
    input_file = sys.argv[1] #dataset
    embd_files = sys.argv[2] #dataset msa-tf embd dir
    t5Embd_files = sys.argv[3] #dataset t5 embd dir
    output = sys.argv[4] #out dir

    test_all_features_np3D = readSort(input_file, embd_files, t5Embd_files)
    Predict(test_all_features_np3D, input_file, output)
    print('Done!')
    

if __name__ == '__main__':
    main()
