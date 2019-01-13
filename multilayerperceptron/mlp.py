# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:31:55 2018

@author: jseiffer
"""
#####################################################################
# Edited by: 		Dex Vo
# Date:				09/25/2018
# Class:			CSC 371 - Assignment 2
#####################################################################
import os
import struct
import numpy as np
from neuralnet import NeuralNetMLP
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels

X_train, y_train = load_mnist('', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

#####################################################################
# Step 2
#####################################################################
#fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
#ax = ax.flatten()
#for i in range(10):
#    img = X_train[y_train == i][0].reshape(28, 28)
#    ax[i].imshow(img, cmap='Greys')
#ax[0].set_xticks([])
#ax[0].set_yticks([])
#plt.tight_layout()
#plt.show()
#####################################################################
# Step 3
#####################################################################
#fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
#ax = ax.flatten()
#for i in range(25):
# img = X_train[y_train == 7][i].reshape(28, 28)
# ax[i].imshow(img, cmap='Greys')
#ax[0].set_xticks([])
#ax[0].set_yticks([])
#plt.tight_layout()
#plt.show()
#####################################################################
# Step 4
#####################################################################
#nn = NeuralNetMLP(n_hidden=100,
#                 l2=0.01, epochs=100, eta=0.0005,
#                 shuffle=True, minibatch_size=100, seed=1);
#nn.fit(X_train[:55000], y_train[:55000], X_train[55000:], y_train[55000:])
#plt.plot(range(nn.epochs), nn.eval_['cost'])
#plt.ylabel('Cost')
#plt.xlabel('Epochs')
#plt.show()
#####################################################################
# Step 5
#####################################################################
#nn = NeuralNetMLP(n_hidden=100,
#                 l2=0.01, epochs=250, eta=0.0005,
#                 shuffle=True, minibatch_size=100, seed=1);
#nn.fit(X_train[:55000], y_train[:55000], X_train[55000:], y_train[55000:])
#plt.plot(range(nn.epochs), nn.eval_['train_acc'],
#label='training')
#plt.plot(range(nn.epochs), nn.eval_['valid_acc'],
#label='validation', linestyle='--')
#plt.ylabel('Accuracy')
#plt.xlabel('Epochs')
#plt.legend()
#plt.show()
#
#y_test_pred = nn.predict(X_test)
#acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])
#print('Test accuracy: %.2f%%' % (acc * 100))
#####################################################################
# Step 6
#####################################################################
#nn = NeuralNetMLP(n_hidden=100,
#                 l2=0.1, epochs=200, eta=0.0005,
#                 shuffle=True, minibatch_size=100, seed=1);
#nn.fit(X_train[:55000], y_train[:55000], X_train[55000:], y_train[55000:])
#plt.plot(range(nn.epochs), nn.eval_['train_acc'],
#label='training')
#plt.plot(range(nn.epochs), nn.eval_['valid_acc'],
#label='validation', linestyle='--')
#plt.ylabel('Accuracy')
#plt.xlabel('Epochs')
#plt.legend()
#plt.show()
#
#y_test_pred = nn.predict(X_test)
#acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])
#print('Test accuracy: %.2f%%' % (acc * 100))
#####################################################################
# Step 7
#####################################################################
nn = NeuralNetMLP(n_hidden=100,
                 l2=0.01, epochs=200, eta=0.0005,
                 shuffle=True, minibatch_size=100, seed=1);
nn.fit(X_train[:55000], y_train[:55000], X_train[55000:], y_train[55000:])
plt.plot(range(nn.epochs), nn.eval_['train_acc'],
label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'],
label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])
print('Test accuracy: %.2f%%' % (acc * 100))

miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
 img = miscl_img[i].reshape(28, 28)
 ax[i].imshow(img, cmap='Greys', interpolation='nearest')
 ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i],
miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


