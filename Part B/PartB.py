import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

NUM_EPOCHS = 50
VAL_SPLIT = 0.1
DROPOUT_RATE = 0.2
CPU, GPU = '/CPU:0', '/GPU:0'
DEVICE = CPU

def loadData():
    pre = os.path.dirname(os.path.realpath(__file__))
    fname = 'data.h5'
    path = os.path.join(pre, fname)
    with h5py.File(path,'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        allTrain = hf.get('trainData')
        allTest = hf.get('testData')
        npTrain = np.array(allTrain)
        npTest = np.array(allTest)
        print('Shape of the array dataset_1: \n', npTrain.shape)
        print('Shape of the array dataset_2: \n', npTest.shape)    

        return npTrain[:,:-1], npTrain[:, -1], npTest[:,:-1], npTest[:, -1]

with tf.device(DEVICE):
    trainX, trainY, testX, testY = loadData()

    print("Enter model type to run [0 layer, 1 layer, 2 layer, 3 layer, 2 layer dropout, 3 layer dropout]: ")
    layer = input()

    flag = True
    while (flag):
        if layer == "0 layer":
            model = tf.keras.models.Sequential([
                layers.Dense(10, activation=tf.nn.softmax, input_shape=(784,))
            ])
            flag = False
        elif layer == "1 layer":
            model = tf.keras.models.Sequential([
                layers.Dense(200, activation=tf.nn.relu, input_shape=(784,)),
                layers.Dense(10, activation=tf.nn.softmax)
            ])
            flag = False
        elif layer == "2 layer":
            model = tf.keras.models.Sequential([
                layers.Dense(400, activation=tf.nn.relu, input_shape=(784,)),
                layers.Dense(200, activation=tf.nn.relu),
                layers.Dense(10, activation=tf.nn.softmax)
            ])
            flag = False
        elif layer == "3 layer":
            model = tf.keras.models.Sequential([
                layers.Dense(600, activation=tf.nn.relu, input_shape=(784,)),
                layers.Dense(400, activation=tf.nn.relu),
                layers.Dense(200, activation=tf.nn.relu),
                layers.Dense(10, activation=tf.nn.softmax)
            ])
            flag = False
        elif layer == "2 layer dropout":
            model = tf.keras.models.Sequential([
                layers.Dense(400, activation=tf.nn.relu, input_shape=(784,)),
                layers.Dropout(DROPOUT_RATE, seed=0),
                layers.Dense(200, activation=tf.nn.relu),
                layers.Dropout(DROPOUT_RATE, seed=0),
                layers.Dense(10, activation=tf.nn.softmax)
            ])
            flag = False
        elif layer == "3 layer dropout":
            model = tf.keras.models.Sequential([
                layers.Dense(600, activation=tf.nn.relu, input_shape=(784,)),
                layers.Dropout(DROPOUT_RATE, seed=0),
                layers.Dense(400, activation=tf.nn.relu),
                layers.Dropout(DROPOUT_RATE, seed=0),
                layers.Dense(200, activation=tf.nn.relu),
                layers.Dropout(DROPOUT_RATE, seed=0),
                layers.Dense(10, activation=tf.nn.softmax)
            ])
            flag = False
        else:
            print("invalid input, try again")
            print("input type for model [1 layer, 2 layer, 3 layer, 2 layer dropout, 3 layer dropout]: ")
            layer = input()

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    history = model.fit(trainX, trainY, batch_size=256, epochs=NUM_EPOCHS, validation_split=VAL_SPLIT)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, NUM_EPOCHS), history.history["loss"], label="train_loss")
    if VAL_SPLIT > 0:
        plt.plot(np.arange(0, NUM_EPOCHS), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), history.history["accuracy"], label="accuracy")
    if VAL_SPLIT > 0:
        plt.plot(np.arange(0, NUM_EPOCHS), history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training loss and accuracy")
    plt.xlabel("Epoch#")
    plt.ylabel("Loss/Accuracy")
    plt.legend(['train_loss','val_loss','train_acc','val_acc'], loc="best")
    plt.show()

    results = model.evaluate(testX, testY)
    print ("Test dataset results: ", results)