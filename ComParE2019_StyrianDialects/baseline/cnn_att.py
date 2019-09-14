import numpy as np
import tensorflow as tf
import math
import os
import joblib
from sklearn.utils import class_weight
from sklearn.metrics import recall_score, confusion_matrix
from data import *
from argparse import ArgumentParser

# arguments
parser = ArgumentParser()
parser.add_argument('-lr', dest='lr', default=0.001, type=float)
parser.add_argument('-keep_proba', dest='keep_proba', default=0.5, type=float)
parser.add_argument('-batch_size', dest='batch_size', default=512, type=int)
parser.add_argument('-save_path', dest='save_path', default='./model/brnn', type=str)
parser.add_argument('-hidden_dim', dest='hidden_dim', default=128, type=int)
parser.add_argument('-grad_clip', dest='grad_clip', default=10, type=float)
parser.add_argument('-input_dim', dest='input_dim', default=23, type=int)
parser.add_argument('-df_dim', dest='df_dim', default=64, type=int)
parser.add_argument('-mode', dest='mode', default='train', type=str)
args = parser.parse_args()



def bn(X, eps=1e-8, offset = 0, scale = 1):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        var = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        output = tf.nn.batch_normalization(X, mean, var, offset, scale, eps)
    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        var = tf.reduce_mean(tf.square(X-mean), 0)
        output = tf.nn.batch_normalization(X, mean, var, offset, scale, eps)
    else:
        raise NotImplementedError
    return output

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        return conv

class CNN:
    def __init__(self, mode, hparams):
        self.mode = mode
        self.hps = hparams

        with tf.variable_scope('cnn_input'):
            # use None for batch size and dynamic sequence length
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.hps.input_dim, 1])
            self.groundtruths = tf.placeholder(tf.float32, shape=[None, 3])

        with tf.variable_scope('net'):
            df_dim = self.hps.df_dim
            net_0 = tf.nn.relu(conv2d(self.inputs, df_dim, name="net_0/cnn"))#32
            d, l = math.ceil(self.hps.input_dim/2), math.ceil(self.hps.seq_length/2) 
            net_1 = tf.nn.relu(bn(conv2d(net_0, df_dim*2, name="net_1/cnn")))#16
            d, l = math.ceil(d/2), math.ceil(l/2) 
            net_2 = tf.nn.relu(bn(conv2d(net_1, df_dim*4, name="net_2/cnn")))#8
            d, l = math.ceil(d/2), math.ceil(l/2) 
            net_3 = tf.nn.relu(bn(conv2d(net_2, df_dim*8, name="net_3/cnn")))#4
            d, l = math.ceil(d/2), math.ceil(l/2) 
            # reshape to: [batch_size, seq_length/8, input_dim/8, df_dim*8]
            net_4 = tf.transpose(net_3, [0, 3, 1, 2])
            net_4 = tf.reshape(net_4, [-1, d*l])
            full_weight0 = tf.get_variable('full_weight0', shape=[d*l, self.hps.hidden_dim], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
            full_bias0 = tf.get_variable('full_bias0', shape=[self.hps.hidden_dim], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            net_4 = tf.matmul(net_4, full_weight0) + full_bias0
            net_4 = tf.reshape(net_4, [-1, df_dim*8 , self.hps.hidden_dim])
            if self.mode == 'train':
                  net_4 = tf.nn.dropout(net_4, self.hps.keep_proba)
            
        with tf.variable_scope('attention_layer'):
            # hidden size of the RNN layer
            hidden_size = net_4.shape[2].value
            attention_size = 16
            # Trainable parameters
            W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
            b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            v = tf.tanh(tf.tensordot(net_4, W, axes=1) + b) 
            vu = tf.tensordot(v, u, axes=1)   # (Batch size,T)
            alphas = tf.nn.softmax(vu)        # (Batch size,T)
            self.outputs = tf.reduce_sum(net_4 * tf.expand_dims(alphas, -1), 1)

        with tf.variable_scope('out_layer'):
            # fully layer
            full_weight = tf.get_variable('full_weight', shape=[self.hps.hidden_dim, 3], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
            full_bias = tf.get_variable('full_bias', shape=[3], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            dense = tf.matmul(self.outputs, full_weight) + full_bias
            #dense = tf.nn.relu(dense)
            '''
            # when training, add dropout to regularize.
            if self.mode == 'train':
                dense = tf.nn.dropout(dense, keep_prob=self.hps.keep_proba)
            '''
            self.logits = tf.nn.softmax(dense)

        with tf.variable_scope('rnn_loss'):
            # use cross_entropy as class loss
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.groundtruths, logits=dense)
            self.loss = tf.reduce_mean(loss)
            # apply gradient clipping
            self.optimizer = tf.train.AdamOptimizer(self.hps.lr).minimize(self.loss)
            
        with tf.variable_scope('rnn_accuracy'):
            self.accuracy = tf.contrib.metrics.accuracy(
                labels=tf.argmax(self.groundtruths, axis=1),
                predictions=tf.argmax(self.logits, axis=1))
            
        with tf.variable_scope('rnn_uar'):  
            lab_argmax = tf.argmax(self.groundtruths, axis=1)
            pred_argmax = tf.argmax(self.logits, axis=1)
            self.lab_argmax = lab_argmax
            self.pred_argmax = pred_argmax

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  # don't forget to initial all variables
        self.sess.run(tf.local_variables_initializer())   # don't forget to initialise the local variables hidden in the tf.metrics.recall method.
        self.saver = tf.train.Saver(max_to_keep=200)       # a saver is for saving or restoring your trained weight

    def train(self, batch_x, batch_y):
        #feed dict
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        # feed in input and groundtruth to get loss and update the weight via Adam optimizer
        loss, accuracy, _ = self.sess.run(
            [self.loss, self.accuracy, self.optimizer], fd)
        lab_argmax= self.sess.run(self.lab_argmax, {self.groundtruths: batch_y})
        pred_argmax= self.sess.run(self.pred_argmax, fd)

        return loss, accuracy

    def test(self, batch_x, batch_y):
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        prediction, accuracy, pred_argmax = self.sess.run([self.logits, self.accuracy, self.pred_argmax], fd)
        lab_argmax= self.sess.run(self.lab_argmax, {self.groundtruths: batch_y})
        loss = self.sess.run(self.loss, fd)
        logits = self.sess.run(self.logits, fd)

        return loss, accuracy, logits

    def get_pred(self, batch_x, batch_y):
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        pred_argmax = self.sess.run(self.pred_argmax, fd)

        return pred_argmax

    def save(self, e):
        if not os.path.exists(self.hps.save_path):
            os.makedirs(self.hps.save_path)
        self.saver.save(self.sess, self.hps.save_path+'/model_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, self.hps.save_path+'/model_%d.ckpt' % (e))


def get_max_len(X, Y, Z):
    length1 = [len(i) for i in X]
    length2 = [len(i) for i in Y]
    length3 = [len(i) for i in Z]

    return max(max(length1), max(length2), max(length3))

# hyperparameter of our network
def get_hparams():
    hparams = tf.contrib.training.HParams(
        EPOCHS=35,
        grad_clip = args.grad_clip,
        BATCH_SIZE=args.batch_size,
        input_dim=args.input_dim,
        df_dim = args.df_dim,
        seq_length=100,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        keep_proba=args.keep_proba,
        save_path=args.save_path)
    return hparams

if __name__ == '__main__':
    # hyperparameters
    hparams = get_hparams()
    # get max sequence length
    hparams.seq_length = get_max_len(X_train, X_devel, X_test)
    print('max sequence length:', hparams.seq_length)
    
    n_train = len(X_train) // hparams.BATCH_SIZE + 1
    n_devel = len(X_devel) // hparams.BATCH_SIZE + 1
    n_test = len(X_test) // hparams.BATCH_SIZE + 1
    if args.mode == 'train':
        # model
        model = CNN(mode='train', hparams=hparams)
        # training
        EPOCHS = hparams.EPOCHS
        uar_devel = []
        conf = []
        for _epoch in range(EPOCHS):  # train for several epochs
            loss_train = 0
            accuracy_train = 0
            UAR_train = 0

            # initialize data generator
            train_batch = BatchGenerator(X_train, y_train, hparams)
            devel_batch = BatchGenerator(X_devel, y_devel, hparams)

            model.mode = 'train'
            pred = []
            gt = []
            for b in range(n_train):  # feed batches one by one
                batch_x, batch_y = train_batch.get(b)
                batch_x = batch_x.reshape(-1, hparams.seq_length, hparams.input_dim, 1)
                loss_batch, accuracy_batch = model.train(batch_x, batch_y)    
                pred_ = model.get_pred(batch_x, batch_y)
                pred += list(pred_)
                gt += list(batch_y)
                loss_train += loss_batch
                accuracy_train += accuracy_batch

            loss_train /= n_train
            accuracy_train /= n_train
            UAR_train = recall_score(np.argmax(gt, 1), pred, average='macro')

            model.save(_epoch)  # save your model after each epoch
            # validation
            if (_epoch + 1) % 1 == 0:
                accuracy_devel = 0
                model.mode = 'test'
                model.hps.keep_proba = 1
                pred = []
                gt = []
                for b in range(n_devel):
                    batch_x, batch_y = devel_batch.get(b)
                    batch_x = batch_x.reshape(-1, hparams.seq_length, hparams.input_dim, 1)
                    _, accuracy_batch,  _ = model.test(batch_x, batch_y)
                    pred_ = model.get_pred(batch_x, batch_y)
                    pred += list(pred_)
                    gt += list(batch_y)
                    accuracy_devel += accuracy_batch
        
                accuracy_devel /= n_devel
                UAR_devel = recall_score(np.argmax(gt, 1), pred, average='macro')
                uar_devel.append(UAR_devel)
                conf.append(confusion_matrix(np.argmax(gt, 1), pred))
                print("Epoch: [%2d/%2d], loss: %.3f, Accuracy train: %.3f, Accuracy devel: %.3f, UAR_train: %.3f, UAR_devel: %.3f"  % (_epoch+1, 
                                                                                                                                           EPOCHS, 
                                                                                                                                           loss_train, 
                                                                                                                                           accuracy_train,
                                                                                                                                           accuracy_devel,
                                                                                                                                           UAR_train,
                                                                                                                                           UAR_devel)) 
                print(conf[-1])
        optimum_epoch = [i+1 for i in range(EPOCHS)][np.argmax(uar_devel)]
        print('\nOptimum epoch: {}, maximum UAR on Devel {}\n'.format(optimum_epoch, np.max(uar_devel)))
        print(conf[np.argmax(uar_devel)])

    else:
        uar = []
        conf = []
        logits_devel = []
        logits_test = []
        devel_batch = BatchGenerator(X_devel, y_devel, hparams, False)
        test_batch = BatchGenerator(X_test, y_devel[:len(X_test)], hparams, False)
        # model
        model = CNN(mode='test', hparams=hparams)
        for e in range(hparams.EPOCHS):
            model.restore(e+1)
            pred = []
            l_devel = []
            l_test = []
            for b in range(n_devel):
                batch_x, batch_y = devel_batch.get(b)
                batch_x = batch_x.reshape(-1, hparams.seq_length, hparams.input_dim, 1)
                pred_ = model.get_pred(batch_x, batch_y)
                _, _, logits_ = model.test(batch_x, batch_y) 
                pred += list(pred_)
                l_devel += list(logits_)
            for b in range(n_test):
                batch_x, batch_y = test_batch.get(b)
                batch_x = batch_x.reshape(-1, hparams.seq_length, hparams.input_dim, 1)
                _, _, logits_ = model.test(batch_x, batch_y)
                l_test += list(logits_)
            #print(pred)
            #y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values
            uar.append(recall_score(np.argmax(y_devel, 1), pred, average='macro'))
            conf.append(confusion_matrix(np.argmax(y_devel, 1), pred))
            print("epoch: {}, uar: {}".format(e+1, recall_score(np.argmax(y_devel, 1), pred, average='macro')))
            logits_devel.append(l_devel) 
            logits_test.append(l_test) 
            #print(confusion_matrix(np.argmax(y_devel, 1), pred))
            tf.reset_default_graph()
        optimum_epoch = [i+1 for i in range(hparams.EPOCHS)][np.argmax(uar)]
        print('\nOptimum epoch: {}, maximum UAR on Devel {}\n'.format(optimum_epoch, np.max(uar)))
        print(conf[np.argmax(uar)])
        np.save('./predictions/SD_devel_cnn_att{}_{}.npy'.format(hparams.save_path.split('_')[-1] ,optimum_epoch), logits_devel[np.argmax(uar)])
        np.save('./predictions/SD_test_cnn_att{}_{}.npy'.format(hparams.save_path.split('_')[-1] ,optimum_epoch), logits_test[np.argmax(uar)])
        #print(logits[optimum_epoch])
