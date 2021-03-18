import tensorflow as tf
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

CPU, GPU = '/CPU:0', '/GPU:0'
DEVICE = GPU

def loadData():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # load the training and test data    
    (tr_x, tr_y), (te_x, te_y) = fashion_mnist.load_data()

    # reshape the feature data
    tr_x = tr_x.reshape(tr_x.shape[0], 784)
    te_x = te_x.reshape(te_x.shape[0], 784)

    # noramlise feature data
    tr_x = tr_x / 255.0
    te_x = te_x / 255.0

    print( "Shape of training features ", tr_x.shape)
    print( "Shape of test features ", te_x.shape)

    # one hot encode the training labels and get the transpose
    tr_y = np_utils.to_categorical(tr_y,10)
    print ("Shape of training labels ", tr_y.shape)

    # one hot encode the test labels and get the transpose
    te_y = np_utils.to_categorical(te_y,10)
    print ("Shape of testing labels ", te_y.shape)

    return tr_x, tr_y, te_x, te_y

def forward_pass(x, weights1, bias1, weights2, bias2):
    # first layer (ReLu)
    A1 = tf.matmul(x, weights1) + bias1
    H1 = tf.maximum(A1, 0)

    # second layer (softmax)
    A2 = tf.matmul(H1, weights2) + bias2
    t = tf.math.exp(A2)
    sumOfT = tf.reduce_sum(t, axis=1)
    sumOfT = tf.reshape(sumOfT, (H1.shape[0], 1))
    H2 = t / sumOfT
    return H2

def cross_entropy(y, pred):
    entropy = -tf.reduce_sum(y * tf.math.log(pred + 1e-10), axis=1)
    loss = tf.reduce_mean(entropy)
    return loss

def calculate_accuracy(x, y, weights1, bias1, weights2, bias2):
    pred = forward_pass(x, weights1, bias1, weights2, bias2)
    predictions_correct = tf.cast(tf.equal(tf.math.argmax(pred, axis=1), tf.math.argmax(y, axis=1)), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)

    return accuracy

def main():
    with tf.device(DEVICE):
        learning_rate = 0.01
        iterations = 500
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

        TRAIN, TRAIN_LABEL, TEST, TEST_LABEL = loadData()

        TRAIN = tf.cast(TRAIN, tf.float32)
        TEST = tf.cast(TEST, tf.float32)
        TRAIN_LABEL = tf.cast(TRAIN_LABEL, tf.float32)
        TEST_LABEL = tf.cast(TEST_LABEL, tf.float32)

        weights1, weights2 = tf.Variable(tf.random.normal([784, 300], mean=0.0, stddev=0.05, seed=0)), tf.Variable(tf.random.normal([300, 10], mean=0.0, stddev=0.05, seed=0))
        bias1, bias2 = tf.Variable(tf.zeros([300])), tf.Variable(tf.zeros([10]))

        accuracy_arr, val_acc_arr, loss_arr, val_loss_arr = [], [], [], []

        for i in range(iterations):
            with tf.GradientTape() as tape:
                pred = forward_pass(TRAIN, weights1, bias1, weights2, bias2)
                current_loss = cross_entropy(TRAIN_LABEL, pred)

                val_pred = forward_pass(TEST, weights1, bias1, weights2, bias2)
                val_loss = cross_entropy(TEST_LABEL, val_pred)

            gradients = tape.gradient(current_loss, [weights1, bias1, weights2, bias2])
            accuracy = calculate_accuracy(TRAIN, TRAIN_LABEL, weights1, bias1, weights2, bias2)

            val_accuracy = calculate_accuracy(TEST, TEST_LABEL, weights1, bias1, weights2, bias2)

            accuracy_arr.append(accuracy)
            loss_arr.append(current_loss)

            val_acc_arr.append(val_accuracy)
            val_loss_arr.append(val_loss)

            print ("Iteration", i, ": Loss = ", current_loss.numpy(), " Acc:", accuracy.numpy(), " Val loss = ", val_loss.numpy(), " Val Acc = ", val_accuracy.numpy())

            adam_optimizer.apply_gradients(zip(gradients, [weights1, bias1, weights2, bias2]))

        test_accuracy = calculate_accuracy(TEST, TEST_LABEL, weights1, bias1, weights2, bias2)
        print ("\nTest accuracy: ", test_accuracy.numpy())

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, iterations), accuracy_arr, label="train_accuracy")
        plt.plot(np.arange(0, iterations), loss_arr, label="train_loss")
        plt.plot(np.arange(0, iterations), val_acc_arr, label="val_accuracy")
        plt.plot(np.arange(0, iterations), val_loss_arr, label="val_loss")
        plt.title("Training loss and accuracy")
        plt.xlabel("Epoch#")
        plt.ylabel("Loss/Accuracy")
        plt.legend(['train_acc','train_loss', 'val_acc', 'val_loss'], loc="best")
        plt.show()

main()