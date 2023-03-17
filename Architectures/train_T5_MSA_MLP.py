import numpy as np
import math
import random
from collections import defaultdict
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM, Dense, Flatten, Reshape, Bidirectional, GRU, Dropout, Input, Conv2D, MaxPool2D, Conv1D, MaxPool1D, Concatenate, BatchNormalization, Activation, AveragePooling2D, Embedding, MultiHeadAttention, Lambda, GlobalMaxPooling1D

LOCAL_NEGHBOR_SIZE = 4
WINDOW_SIZE = 1 + 2*LOCAL_NEGHBOR_SIZE
KEY_SIZE = WINDOW_SIZE
HEAD_NUM = 1


class DataGenerator(keras.utils.Sequence):
    def __init__(self, trainPartition, lenSize, shuffleSet=True, batch_size=1024):
        self.batch_size = batch_size
        self.trainPartition = trainPartition
        self.lenSize = lenSize 
        self.shuffleSet = shuffleSet
        self.on_epoch_end()

    def protToDict(self, proteinName, embdAddress):
        protDict =  defaultdict(dict)
        prot_file = open('{}/{}.embd'.format(embdAddress, proteinName))
        for index, prot_line in enumerate(prot_file):
            prot_line = prot_line.strip().split(':')[1]
            embd_value = [float(x) for x in prot_line.split()]
            protDict[proteinName][index] = embd_value
        return protDict

    def readFeatures2D(self, neighborList,proteinName, protDict, protDictMSA):
        selectedFeature = []
        selectedFeatureMSA = []
        for neighbor in neighborList:
            if neighbor != 'Zpad':
                try:
                    selectedFeature.append(np.array(protDict[neighbor]))
                    selectedFeatureMSA.append(np.array(protDictMSA[neighbor]))
                except:
                    print(proteinName, neighbor)
                    print(protDict[proteinName])
                    exit(1)
            else:
                    selectedFeature.append(np.zeros(1024))
                    selectedFeatureMSA.append(np.zeros(768))

        return np.array(selectedFeature), np.array(selectedFeatureMSA)


    def build2DWindows(self, aaIndex, pLength, proteinName, protDict, protDictMSA, windowSize=WINDOW_SIZE):
        Features2D = []
        proteinLenght = pLength 
        addToLeft = 0
        addToRight = 0
        if aaIndex <= LOCAL_NEGHBOR_SIZE:
            addToLeft = LOCAL_NEGHBOR_SIZE - aaIndex
        if aaIndex+LOCAL_NEGHBOR_SIZE + 1 > proteinLenght:
            addToRight = aaIndex+LOCAL_NEGHBOR_SIZE + 1 - proteinLenght

        neighborList = [i for i in range(aaIndex-LOCAL_NEGHBOR_SIZE + addToLeft, aaIndex+LOCAL_NEGHBOR_SIZE+1 - addToRight)]
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
        Features2D, Features2DMSA = self.readFeatures2D(neighborList, proteinName,protDict, protDictMSA)   

        return np.array(Features2D), np.array(Features2DMSA)
    


    def __len__(self):
        return math.floor(self.lenSize / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.trainPartition[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.lenSize)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = []
        XMSA = []
        y = []
        
        protDictChunk = defaultdict(dict)
        protDictChunkMSA = defaultdict(dict)
        for i,item in enumerate(list_IDs_temp):
            protName, protAAIndex, protAALabel, protLength, RegenerationID = item.split('_')
            if len(protDictChunk[protName]) == 0:
                protDict = self.protToDict(protName, "/path/to/t5/embedding")
                protDictMSA = self.protToDict(protName, "/path/to/msa/embedding")
                protDictChunk.update(protDict)
                protDictChunkMSA.update(protDictMSA)
            Features2D, Features2DMSA = self.build2DWindows(int(protAAIndex), int(protLength), protName, protDictChunk[protName], protDictChunkMSA[protName])
            
            X.append(np.array(Features2D)) 
            XMSA.append(np.array(Features2DMSA)) 
            y.append(int(protAALabel))
        if self.shuffleSet == True:    
            XIndex = np.arange(len(X))
            random.shuffle(XIndex)
            XShuffled = [X[k] for k in XIndex]
            XMSAShuffled = [XMSA[k] for k in XIndex]
            yShuffled = [y[k] for k in XIndex]
            return [np.array(XShuffled), np.array(XMSAShuffled)], np.array(yShuffled)
        else:
            return [np.array(X), np.array(XMSA)], np.array(y)

                    
"""
def rec(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def prc(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = prc(y_true, y_pred)
    recall = rec(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def aupr(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred) 
    auprv = auc(recall, precision)
    return auprv
"""
aupr_metric = AUC(curve='PR')

def partitioning(datasetAddress):
    dataset_file = open(datasetAddress, 'r')
    trainPartition = []
    while True:
        line_PID = dataset_file.readline().strip()
        line_Pseq = dataset_file.readline().strip()
        line_feature = dataset_file.readline().strip()
        if not line_Pseq:
            break
        if len(line_Pseq) < 1024:
            for aaIndex in range(len(line_Pseq)):
                trainPartition.append("{}_{}_{}_{}_0".format(line_PID[1:], aaIndex,line_feature[aaIndex],str(len(line_Pseq))))
    
    return trainPartition
        

trainPartition = partitioning("/path/to/training/set")
valPartition = partitioning("/path/to/validation/set")

# More Train Less val
training_generator = DataGenerator(trainPartition, len(trainPartition))
validation_generator = DataGenerator(valPartition, len(valPartition), shuffleSet=False)



model = Sequential() 
input_features = Input(shape=((int)(WINDOW_SIZE), 1024), name="input_ens_1")
input_features2 = Input(shape=((int)(WINDOW_SIZE), 768), name="input_ens_2")
out2 = Dense(1024, activation='relu', name="dense_T5_0")(input_features)
out2 = Dropout(rate=0.3)(out2)
out2 = Flatten()(out2)
out3 = Dense(768, activation='relu', name="dense_MSA_0")(input_features2)
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



optimizer_adam = keras.optimizers.Adam(learning_rate=(float)(0.001), beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=optimizer_adam, metrics=['acc',aupr_metric]) #metrics=['acc',f1,prc,rec,aupr_metric])
model.summary()


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint("/path/to/model/weights.t5",
                        save_weights_only=True, monitor='val_loss',
                        mode='min', verbose=1, save_best_only=True)
model.fit(x=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    callbacks=[es, mc],
                    workers=16, verbose=2, epochs=100)














