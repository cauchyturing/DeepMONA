# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:51:40 2016

@author: Stephen-Lu
"""

from __future__ import print_function

import os
import time
import PIL.Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import theano
import theano.tensor as T

import lasagne
from sklearn import svm
from utils import tile_raster_images
import matplotlib.pyplot as plt
import update_func

def build_dA(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    da = lasagne.layers.DenseLayer(
            Input, num_units=hid,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform())
            #W=W)
    
    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l2 = lasagne.layers.DenseLayer(
            da, num_units=length*height,
            nonlinearity=lasagne.nonlinearities.sigmoid, 
            #W=l1.W.T)
            W=da.W.T)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l2


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
toneorchord = 'tone'
fltorder = 'C' #C or F
saverst = 1
printmd = 1

num_epochs=10000
noise_factor = 0.0
contraction_level=10 #0.001, 1
length = 98 
height = 39 #39, 193, 2048
lr = 0.01 #0.001
hid = 7
plot_extent = 100

savename = toneorchord + '_STFT_cA_node'+str(hid)
output_folder = 'D:/Dropbox/DL/DeepMSS/Report/STFT_'+toneorchord+'_3s/'

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

f = 'D:/Research/nUS_7-dES_3000ms-nOES_1-fBW_2000mHz-tBW_31ms-amp_94dB-pOl_94-noNoise_'+toneorchord
p_ff = np.loadtxt(f+'/realPartSTFT.csv', delimiter=',')
p_f = []

for i in xrange(7):
    p_f.append(np.asarray(p_ff[:,i*98:(i+1)*98].flatten(order = fltorder), dtype = np.float32))
p_f = np.asarray(p_f, dtype = np.float32) 
pfmax = np.max(p_f)
pfmin = np.min(p_f)
p_f = (p_f - pfmin)/(pfmax - pfmin)
#p_f = (p_f-np.mean(p_f))/np.std(p_f)
#Read mat file
#p_f = np.log(np.loadtxt(f+'/PSD.csv', delimiter=','))
#print p_f.shape
#p_f = (p_f - (np.mean(p_f.flatten()))) / np.var(p_f.flatten())
#p_f = (p_f - np.min(p_f))/(np.max(p_f) - np.min(p_f))

# Record labels    
flabel = open(f+'/signal_labels.csv','rb')
label = [each.strip() for each in flabel.readlines()[1:]]
dic = dict(zip(set(label), range(len(set(label)))))
label_int = [dic[i] for i in label]
#label = map(float, label)
franges = open(f+'/FrequencyBins.csv', 'rb')
ranges = [each.strip() for each in franges.readlines()]
ranges = map(float, ranges)
maxtick = max(ranges)
mintick = min(ranges)
train_set_x_full, train_set_y_full  = p_f, label_int

# Load the dataset

(X_train, y_train), (X_val, y_val), (X_test, y_test) = (p_f, label_int),(p_f, label_int),(p_f, label_int)
X_train = X_train.reshape(len(X_train), -1, height, length)
X_val = X_val.reshape(len(X_val), -1, height, length)
X_test = X_test.reshape(len(X_test), -1, height, length)
batch_size = int(np.ceil(train_set_x_full.shape[0]/10.))
#X_train = X_train.astype('float32')/255.
#X_test = X_test.astype('float32')/255.
#X_test = X_test.reshape(-1, 1, 28, 28)
#X_val = X_train[50000:].reshape(-1, 1, 28, 28)
#y_val = y_train[50000:]
#X_train = X_train[:60000].reshape(-1, 1, 28, 28)
#y_train = y_train[:60000]

X_train_noisy = X_train * np.random.binomial(n=1, p = 1-noise_factor, size=X_train.shape).astype(np.float32) 
X_val_noisy = X_val * np.random.binomial(n=1, p = 1-noise_factor, size=X_val.shape).astype(np.float32) 
X_test_noisy = X_test * np.random.binomial(n=1, p = 1-noise_factor, size=X_test.shape).astype(np.float32) 
# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.tensor4('targets')


if printmd: print("Building model and compiling functions...")
Input = lasagne.layers.InputLayer(shape=(None, 1, height, length),
                                 input_var=input_var)

#W = theano.shared(value=lasagne.init.GlorotNormal(), name='W', borrow=True)
# Add a fully-connected layer of 800 units, using the linear rectifier, and
# initializing weights with Glorot's scheme (which is the default anyway):
da = lasagne.layers.DenseLayer(
        Input, num_units=hid,
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform())
        #W=W)

# Finally, we'll add the fully-connected output layer, of 10 softmax units:
l2 = lasagne.layers.DenseLayer(
        da, num_units=length*height,
        nonlinearity=lasagne.nonlinearities.sigmoid, 
        W=da.W.T)

hidden = lasagne.layers.get_output(da)
hidden_function = theano.function([input_var], hidden)

J = T.reshape(hidden * (1 - hidden),
                         (batch_size, 1, hid)) * T.reshape(
                             da.W, (1, length*height, hid))
        
# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(l2, deterministic=True)
hid_output = lasagne.layers.get_output(da)
predict_function = theano.function([input_var], [prediction, hid_output])
#x = target_var.flatten(2)
#z = prediction

#loss = - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var.flatten(2))
loss = loss.sum(axis= 1).mean() + contraction_level * T.mean(T.sum(J ** 2) / batch_size)
# We could add some weight decay as well here, see lasagne.regularization.

params = lasagne.layers.get_all_params(l2, trainable=True)
#optim = update_func.rmsprop(params)
#updates = optim.updates(loss, params, learning_rate = lr/batch_size)
updates = lasagne.updates.adadelta(
        loss, params, learning_rate=lr/batch_size)
#updates = lasagne.updates.apply_momentum(updates, params=None, momentum=0.9)

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(l2, deterministic=True)
test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                        target_var.flatten(2))
test_loss = test_loss.sum(axis= 1).mean()
# As a bonus, also create an expression for the classification accuracy:
#test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var.flatten(2)),
#                  dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_loss])

# Finally, launch the training loop.
if printmd: print("Starting training...")
# We iterate over epochs:
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train_noisy, X_train, batch_size, shuffle=False):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val_noisy, X_val, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    if printmd:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.6f} ".format(
            val_acc / val_batches))
# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test_noisy, X_test, batch_size, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
if printmd:
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.6f} ".format(
        test_acc / test_batches))

image = PIL.Image.fromarray(tile_raster_images(
    X=da.W.get_value(borrow=True).T,
    img_shape=(height, length), tile_shape=(1, hid),
    tile_spacing=(0, 0)))
cbarmin = np.min(da.W.get_value(borrow=True))
cbarmax = np.max(da.W.get_value(borrow=True))
#image.save('cae_filters.png')
fig, ax = plt.subplots(figsize=(23,16));im = ax.imshow(image, extent=[0,hid*10,mintick,maxtick]);plt.title(savename)
divider = make_axes_locatable(ax);cax = divider.append_axes("right", size="2%", pad=0.1);
cbar = fig.colorbar(im, cax = cax);cbarlen = len(cbar.ax.get_yticklabels());
cbarrange = np.arange(cbarmin, cbarmax + (cbarmax-cbarmin)/cbarlen, (cbarmax-cbarmin)/cbarlen);cbar.ax.set_yticklabels(np.round(cbarrange,2))
# Optionally, you could now dump the network weights to a file like this:
# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
#
# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)

if saverst: plt.savefig(output_folder+savename+'_filter.png', dpi = 100)
savecsv = tile_raster_images( X=np.flipud(da.W.get_value(borrow=True)).T, img_shape=(height, length), tile_shape=(1, hid), tile_spacing=(0, 0))/256.;
savecsv = savecsv * (pfmax - pfmin) + pfmin                
if saverst: np.savetxt(output_folder + savename +'_filter.csv',savecsv, delimiter=',')                
#os.chdir('../')
#            plt.savefig(output_folder+savename+'.png', dpi = 100)
#os.chdir('../')

#%%
#                print 'Saving the weights'
if not os.path.isdir('dA_weights_chord/'):
    os.makedirs('dA_weights_chord/')
#model_file = file('cA_weights_chord/'+savename+'_L1_weight.npz', 'wb')
#numpy.savez(model_file, W_l1 = ca.W.get_value(borrow=True).T)
#model_file.close()
rec_z, rec_y = predict_function(X_train_noisy) 

clf = svm.SVC(kernel='linear', probability=True)
clf.fit(rec_y[:], y_train)
trainacc = clf.score(rec_y[:],y_train)
testacc = clf.score(rec_y[:],y_train)
print ('training score:', trainacc)
print ('test score:', testacc)
  
savematy = tile_raster_images( X=rec_y, img_shape=(hid, 1), tile_shape=(1, 7), tile_spacing=(0, 0))
savemat1 = tile_raster_images( X=p_f, img_shape=(height, length), tile_shape=(1, 7), tile_spacing=(0, 0))
savemat2= tile_raster_images( X=rec_z, img_shape=(height, length), tile_shape=(1, 7), tile_spacing=(0, 0)) 
image1 = PIL.Image.fromarray(savemat1)
image2 = PIL.Image.fromarray(savemat2)
imagey = PIL.Image.fromarray(savematy)
if saverst: np.savetxt(output_folder + savename +'_raw.csv',savemat1/256., delimiter=',')                
if saverst: np.savetxt(output_folder + savename +'_rec.csv',savemat2/256., delimiter=',')                
plt.figure(figsize=(23,16));plt.subplot(211);plt.title('raw signal');plt.imshow(image1, extent=[0,7 * plot_extent,pfmin,pfmax]);plt.subplot(212);plt.title('reconstructed signal');plt.imshow(image2, extent=[0,7 * plot_extent,pfmin,pfmax])
if saverst: plt.savefig(output_folder+savename+'_raw_rec.png', dpi = 100)
#        writeacc.write(str(hid)+'\t'+str(trainacc)+'\t'+str(testacc)+'\n')
#log_prob_mat = clf.predict_log_proba(feature_set[:])
#log_prob = []
#for i in xrange(len(y_test)):
#    log_prob.append(log_prob_mat[i,y_train[i]]) 
#log_prob = np.asarray(log_prob)

#x = T.matrix('x')
#train_da_test = theano.function([], da.L,
#                           givens={x: X_test})
#ce_test = np.mean(train_da_test())
#plt.figure(figsize=(20,14));plt.scatter(ce_test, log_prob);plt.xlabel('CE');plt.ylabel('SVM_prob')
#plt.savefig(output_folder+savename+'_ce_prob.png', dpi = 100)

#Generate .arff file
#if not os.path.isdir('dA_arff_chord/'):
#    os.makedirs('dA_arff_chord/')
#                wkfeatures = numpy.hstack((feature_set[:],numpy.reshape(train_set_y, (len(train_set_y), 1))))
#                write_to_weka('cA_arff/'+savename+'_train.arff', 'spectrogram',map(str, range(hid)),wkfeatures)
#                wkfeatures = numpy.hstack((feature_set[:],numpy.reshape(test_set_y, (len(test_set_y), 1))))
#                write_to_weka('cA_arff/'+savename+'_test.arff', 'spectrogram',map(str, range(hid)),wkfeatures)     
#Rebuild from tone source
#%%
###Input a single column to a well trained model
xx = X_train.copy()
xx[:4,:]=0
xx[5:,:]=0
zz, yy = predict_function(xx)
savematxx = tile_raster_images( X=xx, img_shape=(height, length), tile_shape=(1, 7), tile_spacing=(0, 0)); savematzz= tile_raster_images( X=zz, img_shape=(height, length), tile_shape=(1, 7), tile_spacing=(0, 0))
savematyy = tile_raster_images( X=yy, img_shape=(hid, 1), tile_shape=(1, 7), tile_spacing=(0, 0));
imagexx = PIL.Image.fromarray(savematxx)
imagezz = PIL.Image.fromarray(savematzz)
imageyy = PIL.Image.fromarray(savematyy)
if saverst: np.savetxt(output_folder + savename +'_raw_1source.csv',savematxx/256., delimiter=',')
if saverst: np.savetxt(output_folder + savename +'_rec_1source.csv',savematzz/256., delimiter=',')
plt.figure(figsize=(23,16));plt.subplot(211);plt.title('new input');plt.imshow(imagexx, extent=[0,7 * plot_extent,pfmin,pfmax]);plt.subplot(212);plt.title('reconstruction without retraining');plt.imshow(imagezz, extent=[0,7 * plot_extent,pfmin,pfmax])
if saverst: plt.savefig(output_folder+savename+'_raw_rec_1source.png', dpi = 100)  
#%%
###'dreaming' from white noise
xx = np.random.normal(scale = 0.1, size = X_train.shape).astype(np.float32)
zz, yy = predict_function(xx)
savematxx = tile_raster_images( X=xx, img_shape=(height, length), tile_shape=(1, 7), tile_spacing=(0, 0))
savematzz= tile_raster_images( X=zz, img_shape=(height, length), tile_shape=(1, 7), tile_spacing=(0, 0))
savematyy = tile_raster_images( X=yy, img_shape=(hid, 1), tile_shape=(1, 7), tile_spacing=(0, 0));
imagexx = PIL.Image.fromarray(savematxx)
imagezz = PIL.Image.fromarray(savematzz)
imageyy = PIL.Image.fromarray(savematyy)
if saverst: np.savetxt(output_folder + savename +'_raw_whitenoise.csv',savematxx/256., delimiter=',')
if saverst: np.savetxt(output_folder + savename +'_rec_whitenoise.csv',savematzz/256., delimiter=',')
plt.figure(figsize=(23,16));plt.subplot(211);plt.title('white noise input');plt.imshow(imagexx, extent=[0,7 * plot_extent,pfmin,pfmax]);plt.subplot(212);plt.title('dreaming of AE');plt.imshow(imagezz, extent=[0,7 * plot_extent,pfmin,pfmax])
if saverst: plt.savefig(output_folder+savename+'_raw_rec_whitenoise.png', dpi = 100)  
#%% 
###reconstruction decomposition w.r.t hidden nodes
plt.figure(figsize=(23,16));plt.subplot(4,2,1);plt.title('new input');plt.imshow(image1, extent=[0,7 * plot_extent,pfmin,pfmax]);

da_single = lasagne.layers.DenseLayer(
        Input, num_units=1,
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform())
        #W=W)

# Finally, we'll add the fully-connected output layer, of 10 softmax units:
l2_single = lasagne.layers.DenseLayer(
        da_single, num_units=length*height,
        nonlinearity=lasagne.nonlinearities.sigmoid, 
        #W=l1.W.T)
        W=da_single.W.T)

prediction_single = lasagne.layers.get_output(l2_single, deterministic=True)
hid_output_single = lasagne.layers.get_output(da_single)
predict_function_single = theano.function([input_var], [prediction_single, hid_output_single])
for h_i in xrange(hid):
#    filters = da.W.get_value()[:,h_i]
#    filters = filters.reshape(filters.shape[0],1)
#    yy = spe.expit(np.dot(xx, filters) +da.b.get_value()[h_i])
#    zz= spe.expit(np.dot(yy, filters.T) +l2.b.get_value()[h_i])
    filters, bias1, bias2 = lasagne.layers.get_all_param_values(l2)
    #filters_new = np.zeros(filters.shape)
    filters_new = filters[:, h_i].reshape(-1,1)
    #bias1_new = np.zeros(bias1.shape)
    bias1_new = np.asarray(bias1[h_i], dtype = np.float32).reshape(1,)
    bias2 /= hid
    lasagne.layers.set_all_param_values(l2_single, [filters_new.astype(np.float32), bias1_new.astype(np.float32), bias2.astype(np.float32)])
    zz, yy = predict_function_single(X_train_noisy)
    savematzz= tile_raster_images( X=zz, img_shape=(height, length), tile_shape=(1, 7), tile_spacing=(0, 0))
    #savematyy = tile_raster_images( X=yy, img_shape=(hid, 1), tile_shape=(1, 7), tile_spacing=(0, 0));
    imagezz = PIL.Image.fromarray(savematzz)
    #imageyy = PIL.Image.fromarray(savematyy)
    if saverst: np.savetxt(output_folder + savename +'_rec_1node.csv',savematzz/256., delimiter=',')
    plt.subplot(4,2,h_i+2);plt.title('reconstruction from node '+str(h_i));plt.imshow(imagezz, extent=[0,7 * plot_extent,pfmin,pfmax])
    if saverst: plt.savefig(output_folder+savename+'_raw_rec_1node.png', dpi = 100)