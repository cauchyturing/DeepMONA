# -*- coding: utf-8 -*-
"""
Created on Sat May 14 23:17:56 2016

@author: Stephen-Lu
"""
import cPickle
import gzip
import os
import sys
import time


import numpy
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from matplotlib.ticker import FuncFormatter
from utils import tile_raster_images, write_to_weka
from sklearn import svm
import PIL.Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.special as spe
import update_func
import scipy.io as sio

plt.ioff()
def ReLU(x):
    return (x + numpy.abs(x)) * 0.5

        
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

class dA(object):
    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500, n_batchsize=20,
                 W=None, bhid=None, bvis=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_batchsize = n_batchsize
        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
#            initial_W = numpy.asarray(numpy_rng.uniform(
#                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
#                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
#                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            initial_W = numpy.asarray(0.02 * numpy_rng.randn(n_visible, n_hidden),dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                         dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix('input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
        self.optim = update_func.rmsprop(self.params)
    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        #z = T.clip(z, 0.00001, 0.99999)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        self.L = L
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        # generate the list of updates

        updates = self.optim.updates(cost, self.params, learning_rate = learning_rate/self.n_batchsize)
        return (cost, updates)

learning_rate=0.005
training_epochs=1000
rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))
#hid = 400
#nt = 7
#dur = 4000
#fr = 250

#nts = [21, 7, 2]
#durs = [250, 1000, 4000]
#frs = [4000, 1000, 250]
#hids = [400]
nts = [2, 7, 21]
durs = [250, 1000, 4000]
frs = [4000, 1000, 250]
hids = [100]
chords = [2,3,8]

nts = [7]
durs = [1000]
frs = [1000]
hids = [7,6,5,4,3,2,1]
chords = [3]
toneorchord = 'envir'
fltorder = 'C' #C or F
saverst = 1
printmd = 0
#writeacc = open('acchid21cae.txt', 'w')
for chord in chords:
##    direc = 'D:/Research/MSS/datasetsGenerated_chords_v3_reduced/numberOfTonesPerChord'+str(chord)
##    direcs = os.listdir(direc)
    for nt in nts:
        for dur, fr in zip(durs, frs):
            fname = 'nUnqSig_'+str(nt)+'-durEachSig_'+str(dur)+'ms-nOccurEachSig_3-fBW_'+str(fr)
##            dir_ = [each for each in direcs if fname in each]
##            f = direc+'/'+dir_[0]
            f = 'D:/Research/nUnqSig_7-durEachSig_1000ms-nOccurEachSig_3-fBW_2000mHz-tBW_31ms-amp_94dB-prcOvrlp_94-noNoise_envir'
            #f = 'D:/Research/numberOfTonesPerChord3/nUnqSig_7-durEachSig_1000ms-nOccurEachSig_3-fBW_2000mHz-tBW_31ms-amp_94dB-prcOvrlp_94-noNoise'
            p_ff = np.loadtxt(f+'/nPSD.csv', delimiter=',')
            p_f = []
            tmp2 = list(p_ff[:,:24].flatten(order = fltorder)); tmp1 = tmp2[:p_ff.shape[0]*4];tmp3 = tmp2[-p_ff.shape[0]*4:]
            tmp = np.asarray(tmp1+tmp2+tmp3, dtype = np.float32);p_f.append(tmp)
            for i in xrange(19):
                p_f.append(np.asarray(p_ff[:,24+i*32:24+(i+1)*32].flatten(order = fltorder), dtype = np.float32))
            tmp2 = list(p_ff[:,-24:].flatten(order = fltorder));tmp1 = tmp2[:p_ff.shape[0]*4];tmp3 = tmp2[-p_ff.shape[0]*4:]
            tmp = np.asarray(tmp1+tmp2+tmp3, dtype = np.float32);p_f.append(tmp)
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
    
            
    #        writeacc.write(pivot+'\n')
            for hid in hids:
                hid = hid
                savename = 'dA_'+ fname + '_'+toneorchord+'_' + str(hid)
#                print 'hidden nodes,', hid
                #train_num = 3000
                output_folder = 'D:/Dropbox/DL/DeepMSS/Report/nPSD_envir/'
                 
                """
                This demo is tested on MNIST
                
                :type learning_rate: float
                :param learning_rate: learning rate used for training the contracting
                                      AutoEncoder
                
                :type training_epochs: int
                :param training_epochs: number of epochs used for training
                
                :type dataset: string
                :param dataset: path to the picked dataset
                
                """
                #datasets = load_data(dataset)
                #train_set_x, train_set_y = datasets[0]
                #
                #print 'load data', dataset
                #f = gzip.open(dataset, 'rb')
                #train_data, train_label, train_index = cPickle.load(f)
                #f.close()

                input_size = train_set_x_full.shape[1]
                train_num = int(train_set_x_full.shape[0] * 0.8)
                test_num = train_set_x_full.shape[0] - train_num
                batch_size = int(np.ceil(train_set_x_full.shape[0]/10.))
                
                length = 1
                height = input_size
                
                test_set_x = train_set_x_full[:]
                train_set_x = train_set_x_full[:]
                test_set_y = train_set_y_full[:]
                train_set_y = train_set_y_full[:]
                
                #train_set_x = train_data.T
                train_set_x = theano.shared(numpy.asarray(train_set_x,
                                                           dtype=theano.config.floatX),
                                             borrow=True)
                
                test_set_x = theano.shared(numpy.asarray(test_set_x,
                                                           dtype=theano.config.floatX),
                                             borrow=True)
                # compute number of minibatches for training, validation and testing
                n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
                
                # allocate symbolic variables for the data
                index = T.lscalar()    # index to a [mini]batch
                x = T.matrix('x')  # the data is presented as rasterized images
                
                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder)
                ####################################
                #        BUILDING THE MODEL        #
                ####################################
                
                rng = numpy.random.RandomState(123)
                
                da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
                        n_visible=length * height, n_hidden=hid, n_batchsize=batch_size)
                
                cost, updates = da.get_cost_updates(corruption_level=0.3,
                                                    learning_rate=learning_rate)
                
                train_da = theano.function([index], cost, updates=updates,
                                          givens={x: train_set_x[index * batch_size:
                                              (index + 1) * batch_size]})
                
                start_time = time.clock()
                
                ############
                # TRAINING #
                ############
                
                for epoch in xrange(training_epochs):
                    # go through trainng set
                    c = []
                    for batch_index in xrange(n_train_batches):
                        c.append(train_da(batch_index))
                
                    if printmd: print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
                
                end_time = time.clock()
                
                training_time = (end_time - start_time)
                
                training_time = (end_time - start_time)
                
#                print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
#                                      ' ran for %.2fm' % ((training_time) / 60.))
                #%%
                image = PIL.Image.fromarray(tile_raster_images(
                    X=da.W.get_value(borrow=True).T,
                    img_shape=(height/32, length*32), tile_shape=(1, hid),
                    tile_spacing=(1, 1)))
                cbarmin = np.min(da.W.get_value(borrow=True))
                cbarmax = np.max(da.W.get_value(borrow=True))
                #image.save('cae_filters.png')
                fig, ax = plt.subplots(figsize=(23,16));im = ax.imshow(image, extent=[0,hid*10,mintick,maxtick]);plt.title(savename)
                divider = make_axes_locatable(ax);cax = divider.append_axes("right", size="2%", pad=0.1);
                cbar = fig.colorbar(im, cax = cax);cbarlen = len(cbar.ax.get_yticklabels());
                cbarrange = np.arange(cbarmin, cbarmax + (cbarmax-cbarmin)/cbarlen, (cbarmax-cbarmin)/cbarlen);cbar.ax.set_yticklabels(np.round(cbarrange,2))
##savecsv = tile_raster_images( X=numpy.flipud(da.W.get_value(borrow=True)).T, img_shape=(height/31, length*31), tile_shape=(1, hid), tile_spacing=(1, 1))/256.;np.savetxt('D:\Dropbox\DL\DeepMSS\Report\dA-21 hidden units-tone dataset.csv',savecsv)                                
                if saverst: plt.savefig(output_folder+savename+'.png', dpi = 100)
                savecsv = tile_raster_images( X=numpy.flipud(da.W.get_value(borrow=True)).T, img_shape=(height/32, length*32), tile_shape=(1, hid), tile_spacing=(1, 1))/256.;
                savecsv = savecsv * (pfmax - pfmin) + pfmin                
                if saverst: np.savetxt(output_folder + savename +'.csv',savecsv, delimiter=',')                
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
                
                clf = svm.SVC(kernel='linear', probability=True)
                feature_set = spe.expit(numpy.dot(train_set_x_full, da.W.get_value()) +da.b.get_value())
        #            feature_set = numpy.dot(ca.W.get_value().T,train_set_x_full).T +ca.b.get_value()
                clf.fit(feature_set[:], train_set_y)
                trainacc = clf.score(feature_set[:],train_set_y)
                testacc = clf.score(feature_set[:],test_set_y)
                
#                print 'training score:', trainacc
#                print 'test score:', testacc
                print hid, '\t',numpy.mean(c), '\t', trainacc
                rec_z = spe.expit(numpy.dot(feature_set, da.W.get_value().T) +da.b_prime.get_value())
                savemat1 = tile_raster_images( X=p_f, img_shape=(height/32, length*32), tile_shape=(1, 21), tile_spacing=(1, 1))
                savemat2= tile_raster_images( X=rec_z, img_shape=(height/32, length*32), tile_shape=(1, 21), tile_spacing=(1, 1))             
                image1 = PIL.Image.fromarray(savemat1)
                image2 = PIL.Image.fromarray(savemat2)
                if saverst: np.savetxt(output_folder + savename +'_raw.csv',savemat1/256., delimiter=',')                
                if saverst: np.savetxt(output_folder + savename +'_rec.csv',savemat2/256., delimiter=',')                
                plt.figure(figsize=(23,16));plt.subplot(211);plt.title('raw signal');plt.imshow(image1, extent=[0,21,pfmin,pfmax]);plt.subplot(212);plt.title('reconstructed signal');plt.imshow(image2, extent=[0,21,pfmin,pfmax])
                if saverst: plt.savefig(output_folder+savename+'_raw_rec.png', dpi = 100)
        #        writeacc.write(str(hid)+'\t'+str(trainacc)+'\t'+str(testacc)+'\n')
                log_prob_mat = clf.predict_log_proba(feature_set[:])
                log_prob = []
                for i in xrange(len(test_set_y)):
                    log_prob.append(log_prob_mat[i,test_set_y[i]]) 
                log_prob = numpy.asarray(log_prob)
                
                train_da_test = theano.function([], da.L,
                                           givens={x: test_set_x})
                ce_test = numpy.mean(train_da_test())
                #plt.figure(figsize=(20,14));plt.scatter(ce_test, log_prob);plt.xlabel('CE');plt.ylabel('SVM_prob')
                #plt.savefig(output_folder+savename+'_ce_prob.png', dpi = 100)
                
                #Generate .arff file
                if not os.path.isdir('dA_arff_chord/'):
                    os.makedirs('dA_arff_chord/')
#                wkfeatures = numpy.hstack((feature_set[:],numpy.reshape(train_set_y, (len(train_set_y), 1))))
#                write_to_weka('cA_arff/'+savename+'_train.arff', 'spectrogram',map(str, range(hid)),wkfeatures)
#                wkfeatures = numpy.hstack((feature_set[:],numpy.reshape(test_set_y, (len(test_set_y), 1))))
#                write_to_weka('cA_arff/'+savename+'_test.arff', 'spectrogram',map(str, range(hid)),wkfeatures)
#writeacc.close()            
#%%
#W_ = numpy.flipud(ca.W.get_value(borrow=True))[:,].reshape(height, length)
#nt = 21
#dur = 250
#fr = 4000  
#hid = 400        
#pivot = 'numTones_'+str(nt)+'_Duration_ms_'+str(dur)+'_FreqResolution_mHz_'+str(fr)
#direc = 'D:\\Research\\MSS\\datasetsGenerated'
#direcs = os.listdir(direc)
#direcs = [each for each in direcs if pivot in each]
#flabel = open(direc+os.sep+direcs[0]+os.sep+'tone_labels.csv','rb')
#label = flabel.read().split()
#flabel.close()
#label = map(float, label)
#maxtick = max(label)
#mintick = min(label)
#savename = 'CAE_'+ fname + '_bsp_10000_hid_sigmoid_'+str(400)
#fname = 'Gen_'+pivot
#W_= numpy.load('cA_weights/'+savename+'_L1_weight.npz')
#W_ = W_['W_l1'].T
#length = 1
#height = W_.shape[0]
#W_ = numpy.flipud(W_)[:,:]#.reshape(height, length)
#cbarmin = np.min(W_)
#cbarmax = np.max(W_)
#W_ = PIL.Image.fromarray(tile_raster_images( X=W_.T, img_shape=(height, length), tile_shape=(1, hid), tile_spacing=(1, 1)))
##image.save('cae_filters.png')
#savename = 'CAE_'+ fname + '_bsp_10000_hid_sigmoid_'+str(hid)
#fig, ax = plt.subplots(figsize=(23,16));im = ax.imshow(W_, extent=[0,hid,mintick,maxtick]);
#divider = make_axes_locatable(ax);cax = divider.append_axes("right", size="20%", pad=0.1);
#cbar = fig.colorbar(im, cax = cax);cbarlen = len(cbar.ax.get_yticklabels());
#cbarrange = np.arange(cbarmin, cbarmax + (cbarmax-cbarmin)/cbarlen, (cbarmax-cbarmin)/cbarlen);cbar.ax.set_yticklabels(np.round(cbarrange,5))
#           
