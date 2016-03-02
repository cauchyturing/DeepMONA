# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 15:19:35 2016

@author: Stephen
"""
#python trainmnist.py -s mnist.npy

import VariationalAutoencoder
import numpy as np
import argparse
import time
import gzip, cPickle
import PIL.Image
from utils import tile_raster_images
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-d","--double", help="Train on hidden layer of previously trained AE - specify params", default = False)

args = parser.parse_args()

print "Loading Gen data"
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

f = file('../../TIMIT_1.pkl', 'rb')
#f = file('../../Gen_NTTF_1024_res_4000.pkl', 'rb')
data  = cPickle.load(f)
f.close()

dimZ = 20
HU_decoder = 225
HU_encoder = HU_decoder

batch_size = 20
L = 1
learning_rate = 0.05

if args.double:
    print 'computing hidden layer to train new AE on'
    prev_params = np.load(args.double)
    data = (np.tanh(data.dot(prev_params[0].T) + prev_params[5].T) + 1) /2
#    x_test = (np.tanh(x_test.dot(prev_params[0].T) + prev_params[5].T) +1) /2

[N,dimX] = data.shape
encoder = VariationalAutoencoder.VA(HU_decoder,HU_encoder,dimX,dimZ,batch_size,L,learning_rate)


if args.double:
    encoder.continuous = True

print "Creating Theano functions"
encoder.createGradientFunctions()

print "Initializing weights and biases"
encoder.initParams()
lowerbound = np.array([])
testlowerbound = np.array([])

begin = time.time()
epoch = 25
for j in xrange(epoch):
    encoder.lowerbound = 0
    print 'Iteration:', j
    encoder.iterate(data)
    end = time.time()
    print("Iteration %d, lower bound = %.2f,"
          " time = %.2fs"
          % (j, encoder.lowerbound/N, end - begin))
    begin = end

#    if j % 5 == 0:
#        print "Calculating test lowerbound"
#        testlowerbound = np.append(testlowerbound,encoder.getLowerBound(x_test))
#%%
imagew = PIL.Image.fromarray(tile_raster_images( X= encoder.params[0][:], img_shape=(40, 20), 
                                                tile_shape=(15, 15), tile_spacing=(1, 1)))
plt.figure()                                                
plt.imshow(imagew)
#imagew.save('MNIST_VAE_'+str(epoch)+'.png')                                                
#%%