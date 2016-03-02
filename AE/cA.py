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

from utils import tile_raster_images, write_to_weka
from sklearn import svm
import PIL.Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.special as spe
import update_func

plt.ion()
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
    

class cA(object):
    """ Contractive Auto-Encoder class (cA)

    The contractive autoencoder tries to reconstruct the input with an
    additional constraint on the latent space. With the objective of
    obtaining a robust representation of the input space, we
    regularize the L2 norm(Froebenius) of the jacobian of the hidden
    representation with respect to the input. Please refer to Rifai et
    al.,2011 for more details.

    If x is the input then equation (1) computes the projection of the
    input into the latent space h. Equation (2) computes the jacobian
    of h with respect to x.  Equation (3) computes the reconstruction
    of the input, while equation (4) computes the reconstruction
    error and the added regularization term from Eq.(2).

    .. math::

        h_i = s(W_i x + b_i)                                             (1)

        J_i = h_i (1 - h_i) * W_i                                        (2)

        x' = s(W' h  + b')                                               (3)

        L = -sum_{k=1}^d [x_k \log x'_k + (1-x_k) \log( 1-x'_k)]
             + lambda * sum_{i=1}^d sum_{j=1}^n J_{ij}^2                 (4)

    """

    def __init__(self, numpy_rng, input=None, n_visible=27, n_hidden=100,
                 n_batchsize=20, W=None, bhid=None, bvis=None):
        """Initialize the cA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the contraction level. The
        constructor also receives symbolic variables for the input, weights and
        bias.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given
                     one is generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone cA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type n_batchsize int
        :param n_batchsize: number of examples per batch

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_batchsize = n_batchsize
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
#                      size=(n_visible, n_hidden)),
#                                      dtype=theano.config.floatX)
#            initial_W = numpy.asarray(0.02 * numpy_rng.randn(n_visible, n_hidden),dtype=theano.config.floatX)
            initial_W = numpy.asarray(numpy.random.normal(scale = 1./n_visible, size = (n_visible, n_hidden)), 
                                      dtype=theano.config.floatX)
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
        
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
        self.optim = update_func.rmsprop(self.params)
    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        #return T.tanh(T.dot(input, self.W) + self.b)
        #return ReLU(T.dot(input, self.W) + self.b)

    def get_jacobian(self, hidden, W):
        """Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis

        """
        return T.reshape(hidden * (1 - hidden),
                         (self.n_batchsize, 1, self.n_hidden)) * T.reshape(
                             W, (1, self.n_visible, self.n_hidden))

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        #return  T.tanh(T.dot(hidden, self.W_prime) + self.b_prime)
        #return  ReLU(T.dot(hidden, self.W_prime) + self.b_prime)
        
    def get_cost_updates(self, contraction_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the cA """

        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
#        z = T.clip(z, 0.001, 0.999)
        J = self.get_jacobian(y, self.W)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        self.L = self.x * T.log(z) + (1 - self.x) * T.log(1 - z)
                             
        self.L_rec = - T.sum(self.x * T.log(z) +
                             (1 - self.x) * T.log(1 - z),
                             axis=1)

        # Compute the jacobian and average over the number of samples/minibatch
        self.L_jacob = T.sum(J ** 2) / self.n_batchsize

        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(self.L_rec) + contraction_level * T.mean(self.L_jacob)

        # compute the gradients of the cost of the `cA` with respect
        # to its parameters
#        gparams = T.grad(cost, self.params)
        # generate the list of updates
#        updates = []
#        for param, gparam in zip(self.params, gparams):
#            #gparam = T.clip(gparam, -1, 1)
#            updates.append((param, param - learning_rate * gparam))
#        updates = update_func.RMSprop(cost, self.params)
        #updates = AdaDelta(cost, self.params)
        #updates = adam(cost, self.params)
        updates = self.optim.updates(cost, self.params, learning_rate = learning_rate/self.n_batchsize)
#        updates = update_func.adagrad(cost, self.params, learning_rate = learning_rate/self.n_batchsize)
        return (cost, updates)


learning_rate=0.01
training_epochs=30
#hid = 400
#nt = 7
#dur = 4000
#fr = 250

#nts = [21, 7, 2]
#durs = [250, 1000, 4000]
#frs = [4000, 1000, 250]
#hids = [400]
nts = [2]
durs = [1000]
frs = [1000]
hids = [100]
#writeacc = open('acchid21cae.txt', 'w')
for nt in nts:
    for dur, fr in zip(durs, frs):
        direc = 'D:\\Research\\MSS\\datasetsGenerated'
        pivot = 'numTones_'+str(nt)+'_Duration_ms_'+str(dur)+'_FreqResolution_mHz_'+str(fr)
        print 'loading dataset:', pivot
        direcs = os.listdir(direc)
        direcs = [each for each in direcs if pivot in each]
        flabel = open(direc+os.sep+direcs[0]+os.sep+'tone_labels.csv','rb')
        label = flabel.read().split()
        flabel.close()
        label = map(float, label)
        maxtick = max(label)
        mintick = min(label)
        
        fname = 'Gen_'+pivot
        dataset='../'+fname+'_bsp_10000.pkl'
#        writeacc.write(pivot+'\n')
        for hid in hids:
            hid = hid
            print 'hidden nodes,', hid
            #train_num = 3000
            batch_size=50
            output_folder='cA_plots/'
            contraction_level=.1
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
            
            print 'load data', dataset
            f = file(dataset, 'rb')
            train_set_x_full, train_set_y_full  = cPickle.load(f)
            f.close()
            #
            #print 'load data', dataset
            #f = gzip.open(dataset, 'rb')
            #train_data, train_label, train_index = cPickle.load(f)
            #f.close()
            
            input_size = train_set_x_full.shape[0]
            train_num = int(train_set_x_full.shape[1] * 0.8)
            test_num = train_set_x_full.shape[1] - train_num
            
            length = 1
            height = input_size
            
            test_set_x = train_set_x_full.T[train_num:]
            train_set_x = train_set_x_full.T[:train_num]
            test_set_y = train_set_y_full.T[train_num:]
            train_set_y = train_set_y_full.T[:train_num]
            
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
            
            ca = cA(numpy_rng=rng, input=x,
                    n_visible=length * height, n_hidden=hid, n_batchsize=batch_size)
            
            cost, updates = ca.get_cost_updates(contraction_level=contraction_level,
                                                learning_rate=learning_rate)
            
            train_ca = theano.function([index], [T.mean(ca.L_rec), ca.L_jacob],
                                       updates=updates,
                                       givens={x: train_set_x[index * batch_size:
                                                            (index + 1) * batch_size]})
            
            start_time = time.clock()
            
            ############
            # TRAINING #
            ############
            
            # go through training epochs
            for epoch in xrange(training_epochs):
                # go through trainng set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(train_ca(batch_index))
            
                c_array = numpy.vstack(c)
                print 'Training epoch %d, reconstruction cost ' % epoch, numpy.mean(
                    c_array[0]), ' jacobian norm ', numpy.mean(numpy.sqrt(c_array[1]))
            
            end_time = time.clock()
            
            training_time = (end_time - start_time)
            
            print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                                  ' ran for %.2fm' % ((training_time) / 60.))
            #%%
            image = PIL.Image.fromarray(tile_raster_images(
                X=numpy.flipud(ca.W.get_value(borrow=True)).T,
            #    X=np.reshape(numpy.flipud(ca.W.get_value(borrow=True)).T[106], (height, length)),
                img_shape=(height, length), tile_shape=(1, hid),
                tile_spacing=(1, 1)))
            cbarmin = np.min(ca.W.get_value(borrow=True))
            cbarmax = np.max(ca.W.get_value(borrow=True))
            #image.save('cae_filters.png')
            savename = 'CAE_'+ fname + '_bsp_10000_hid_sigmoid_'+str(hid)
            fig, ax = plt.subplots(figsize=(23,16));im = ax.imshow(image, extent=[0,hid,mintick,maxtick]);
            divider = make_axes_locatable(ax);cax = divider.append_axes("right", size="20%", pad=0.1);plt.title(savename)
            cbar = fig.colorbar(im, cax = cax);cbarlen = len(cbar.ax.get_yticklabels());
            cbarrange = np.arange(cbarmin, cbarmax + (cbarmax-cbarmin)/cbarlen, (cbarmax-cbarmin)/cbarlen);cbar.ax.set_yticklabels(np.round(cbarrange,2))
            
            plt.figure(figsize=(23.0, 16.0));plt.imshow(image);plt.title(savename);plt.colorbar()
            plt.savefig(output_folder+savename+'.png', dpi = 100)
#os.chdir('../')
        
            #%%
            print 'Saving the weights'
            if not os.path.isdir('cA_weights/'):
                os.makedirs('cA_weights/')
            model_file = file('cA_weights/'+savename+'_L1_weight.npz', 'wb')
            numpy.savez(model_file, W_l1 = ca.W.get_value(borrow=True).T)
            model_file.close()
            
            clf = svm.SVC(kernel='linear', probability=True)
            feature_set = spe.expit(numpy.dot(ca.W.get_value().T,train_set_x_full).T +ca.b.get_value())
    #            feature_set = numpy.dot(ca.W.get_value().T,train_set_x_full).T +ca.b.get_value()
            clf.fit(feature_set[:train_num], train_set_y)
            trainacc = clf.score(feature_set[:train_num],train_set_y)
            testacc = clf.score(feature_set[train_num:],test_set_y)
            
            print 'training score:', trainacc
            print 'test score:', testacc
    #        writeacc.write(str(hid)+'\t'+str(trainacc)+'\t'+str(testacc)+'\n')
            log_prob_mat = clf.predict_log_proba(feature_set[train_num:])
            log_prob = []
            for i in xrange(len(test_set_y)):
                log_prob.append(log_prob_mat[i,test_set_y[i]]) 
            log_prob = numpy.asarray(log_prob)
            
            train_ca_test = theano.function([], ca.L,
                                       givens={x: test_set_x})
            ce_test = numpy.mean(train_ca_test(), axis = 1)
            plt.figure(figsize=(20,14));plt.scatter(ce_test, log_prob);plt.xlabel('CE');plt.ylabel('SVM_prob')
            plt.savefig(output_folder+savename+'_ce_prob.png', dpi = 100)
            
            #Generate .arff file
            if not os.path.isdir('cA_arff/'):
                os.makedirs('cA_arff/')
            wkfeatures = numpy.hstack((feature_set[:train_num],numpy.reshape(train_set_y, (len(train_set_y), 1))))
            write_to_weka('cA_arff/'+savename+'_train.arff', 'spectrogram',map(str, range(hid)),wkfeatures)
            wkfeatures = numpy.hstack((feature_set[train_num:],numpy.reshape(test_set_y, (len(test_set_y), 1))))
            write_to_weka('cA_arff/'+savename+'_test.arff', 'spectrogram',map(str, range(hid)),wkfeatures)
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
