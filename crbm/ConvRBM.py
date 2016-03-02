import morb, numpy
from morb import rbms, stats, updaters, trainers, monitors, units, parameters
import PIL.Image
import theano
import theano.tensor as T
from utils1 import tile_raster_images
import numpy as np

import gzip, cPickle, time, os

import matplotlib.pyplot as plt
plt.ioff()

from sklearn import svm
from utils import generate_data, get_context
from utils1 import tile_raster_images
import scipy.special as spe
# DEBUGGING
from mpl_toolkits.axes_grid1 import make_axes_locatable
from theano import ProfileMode
# mode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
# mode = theano.compile.DebugMode(check_py_code=False, require_matching_strides=False)
mode = None

def plot_data(d):
    plt.figure(5)
    plt.clf()
    plt.imshow(d.reshape((height,width)), interpolation='gaussian')
    plt.draw()


def sample_evolution(start, ns=100): # start = start data
    sample = t.compile_function(initial_vmap, mb_size=1, monitors=[m_model], name='evaluate', train=False, mode=mode)
    
    data = start
    plot_data(data)
    

    while True:
        for k in range(ns):
            for x in sample({ rbm.v: data }): # draw a new sample
                data = x[0]
            
        plot_data(data)
# load data


#f = gzip.open('../mnist.pkl.gz','rb')
#train_set, valid_set, test_set = cPickle.load(f)
#f.close()

#train_set_x, train_set_y = train_set
#valid_set_x, valid_set_y = valid_set
#test_set_x, test_set_y = test_set
#
#
## TODO DEBUG
#train_set_x = train_set_x[:10000]
#valid_set_x = valid_set_x[:1000]
#dataset='../TIMIT_1.pkl'
#train_num = 15000
#dataset='../Gen_NTTF_1024.pkl'
#train_num = 3000
nts = [21, 7, 2]
durs = [250, 1000, 4000]
frs = [4000, 1000, 250]
hids = [1, 2, 4, 8, 16]
#nts = [7,2]
#durs = [1000]
#frs = [1000]
epochs = 20
writeacc = open('acc1hidcrbm.txt', 'w')
for hid in hids:
    hidden_maps = hid # 100 # 50
    print 'hidden units:', hidden_maps
    for nt in nts:
        for dur, fr in zip(durs, frs):
            pivot = 'numTones_'+str(nt)+'_Duration_ms_'+str(dur)+'_FreqResolution_mHz_'+str(fr)
            print ">> Loading dataset...", pivot, hidden_maps       
            writeacc.write(pivot+'_hid_'+str(hidden_maps)+'\n')
            direc = 'D:\\Research\\MSS\\datasetsGenerated'
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
            
            f = file(dataset, 'rb')
            train_set_x_full, train_set_y_full  = cPickle.load(f)
            f.close()
            
            input_size = train_set_x_full.shape[0]
            train_num = int(train_set_x_full.shape[1] * 0.8)
            test_num = train_set_x_full.shape[1] - train_num
            
            width = 1
            height = input_size
            visible_maps = 1
            filter_height = height # 8
            filter_width = 1 # 8
            mb_size = 1 # 1
            test_set_x = train_set_x_full.T[train_num:]
            train_set_x = train_set_x_full.T[:train_num]
            test_set_y = train_set_y_full.T[train_num:]
            train_set_y = train_set_y_full.T[:train_num]
            valid_set_x = test_set_x
            
            # reshape data for convolutional RBM
            train_set_x = train_set_x.reshape((train_set_x.shape[0], 1, height, width))
            valid_set_x = valid_set_x.reshape((valid_set_x.shape[0], 1, height, width))
            #test_set_x = test_set_x.reshape((test_set_x.shape[0], 1, 28, 28))
            
            
            
            print ">> Constructing RBM..."
            fan_in = visible_maps * filter_height * filter_width
            
            """
            initial_W = numpy.asarray(
                        self.numpy_rng.uniform(
                            low = - numpy.sqrt(3./fan_in),
                            high = numpy.sqrt(3./fan_in),
                            size = self.filter_shape
                        ), dtype=theano.config.floatX)
            """
            numpy_rng = np.random.RandomState(123)
            initial_W = np.asarray(
                        numpy_rng.normal(
                            0, 0.5 / np.sqrt(fan_in),
                            size = (hidden_maps, visible_maps, filter_height, filter_width)
                        ), dtype=theano.config.floatX)
            initial_bv = np.zeros(visible_maps, dtype = theano.config.floatX)
            initial_bh = np.zeros(hidden_maps, dtype = theano.config.floatX)
            
            
            
            shape_info = {
              'hidden_maps': hidden_maps,
              'visible_maps': visible_maps,
              'filter_height': filter_height,
              'filter_width': filter_width,
              'visible_height': height,
              'visible_width': width,
              'mb_size': mb_size
            }
            
            # shape_info = None
            
            
            # rbms.SigmoidBinaryRBM(n_visible, n_hidden)
            rbm = morb.base.RBM()
            rbm.v = units.BinaryUnits(rbm, name='v') # visibles
            rbm.h = units.BinaryUnits(rbm, name='h') # hiddens
            rbm.W = parameters.Convolutional2DParameters(rbm, [rbm.v, rbm.h], theano.shared(value=initial_W, name='W'), name='W', shape_info=shape_info)
            # one bias per map (so shared across width and height):
            rbm.bv = parameters.SharedBiasParameters(rbm, rbm.v, 3, 2, theano.shared(value=initial_bv, name='bv'), name='bv')
            rbm.bh = parameters.SharedBiasParameters(rbm, rbm.h, 3, 2, theano.shared(value=initial_bh, name='bh'), name='bh')
            
            initial_vmap = { rbm.v: T.tensor4('v') }
            
            # try to calculate weight updates using CD-1 stats
            print ">> Constructing contrastive divergence updaters..."
            s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=1, mean_field_for_stats=[rbm.v], mean_field_for_gibbs=[rbm.v])
            
            umap = {}
            for var in rbm.variables:
                pu =  var + 0.001 * updaters.CDUpdater(rbm, var, s)
                umap[var] = pu
            
            print ">> Compiling functions..."
            t = trainers.MinibatchTrainer(rbm, umap)
            m = monitors.reconstruction_mse(s, rbm.v)
            m_data = s['data'][rbm.v]
            m_model = s['model'][rbm.v]
            e_data = rbm.energy(s['data']).mean()
            e_model = rbm.energy(s['model']).mean()
            
            
            # train = t.compile_function(initial_vmap, mb_size=32, monitors=[m], name='train', mode=mode)
            train = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[m, e_data, e_model], name='train', mode=mode)
            evaluate = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[m, m_data, m_model, e_data, e_model], name='evaluate', train=False, mode=mode)
            
            # TRAINING 
            
            
            print ">> Training for %d epochs..." % epochs
            
            mses_train_so_far = []
            mses_valid_so_far = []
            edata_train_so_far = []
            emodel_train_so_far = []
            edata_so_far = []
            emodel_so_far = []
            
            start_time = time.time()
            
            for epoch in range(epochs):
                monitoring_data_train = [(cost, energy_data, energy_model) for cost, energy_data, energy_model in train({ rbm.v: train_set_x })]
                mses_train, edata_train_list, emodel_train_list = zip(*monitoring_data_train)
                mse_train = np.mean(mses_train)
                edata_train = np.mean(edata_train_list)
                emodel_train = np.mean(emodel_train_list)
                
                monitoring_data = [(cost, data, model, energy_data, energy_model) for cost, data, model, energy_data, energy_model in evaluate({ rbm.v: valid_set_x })]
                mses_valid, vdata, vmodel, edata, emodel = zip(*monitoring_data)
                mse_valid = np.mean(mses_valid)
                edata_valid = np.mean(edata)
                emodel_valid = np.mean(emodel)
                
                # plotting
                mses_train_so_far.append(mse_train)
                mses_valid_so_far.append(mse_valid)
                edata_so_far.append(edata_valid)
                emodel_so_far.append(emodel_valid)
                edata_train_so_far.append(edata_train)
                emodel_train_so_far.append(emodel_train)
                
            #    plt.figure(1)
            #    plt.clf()
            #    plt.plot(mses_train_so_far, label='train')
            #    plt.plot(mses_valid_so_far, label='validation')
            #    plt.title("MSE")
            #    plt.legend()
            #    plt.draw()
            #    
            #    plt.figure(4)
            #    plt.clf()
            #    plt.plot(edata_so_far, label='validation / data')
            #    plt.plot(emodel_so_far, label='validation / model')
            #    plt.plot(edata_train_so_far, label='train / data')
            #    plt.plot(emodel_train_so_far, label='train / model')
            #    plt.title("energy")
            #    plt.legend()
            #    plt.draw()
            #    
            #    # plot some samples
            #    plt.figure(2)
            #    plt.clf()
            #    plt.imshow(vdata[0][0].reshape((28, 28)))
            #    plt.draw()
            #    plt.figure(3)
            #    plt.clf()
            #    plt.imshow(vmodel[0][0].reshape((28, 28)))
            #    plt.draw()
            
                
                print "Epoch %d" % epoch
                print "training set: MSE = %.6f, data energy = %.2f, model energy = %.2f" % (mse_train, edata_train, emodel_train)
                print "validation set: MSE = %.6f, data energy = %.2f, model energy = %.2f" % (mse_valid, edata_valid, emodel_valid)
                print "Time: %.2f s" % (time.time() - start_time)
            #%%
            base = rbm.W.var.get_value().reshape(hidden_maps, filter_height * filter_width)
            image = PIL.Image.fromarray(tile_raster_images(
                         X= base[:],
                         img_shape=(filter_height, filter_width), tile_shape=(1, hidden_maps),
                         tile_spacing=(1, 1)))
            
            savename = 'CRBM_'+ fname + '_bsp_10000_'+str(hidden_maps)
    #        plt.figure(figsize=(23.0, 16.0));plt.imshow(image);plt.title(savename)
            fig, ax = plt.subplots(figsize=(23,16));im = ax.imshow(image, extent=[0,hidden_maps,mintick,maxtick]);
            divider = make_axes_locatable(ax);cax = divider.append_axes("right", size="2%", pad=0.1);fig.colorbar(im, cax = cax)
    #        plt.title(savename);plt.savefig(savename+'.png', dpi = 100)
            
            #%%
    #        print '>> Saving the weights'
    #        model_file = file(savename+'_L1_weight.npz', 'wb')
    #        numpy.savez(model_file, W_l1 = base)
    #        model_file.close()
    #        
            clf = svm.SVC(kernel='linear', probability=True)
            #feature_set = spe.expit(numpy.dot(base,train_set_x_full).T)
            feature_set = numpy.dot(base,train_set_x_full).T
            clf.fit(feature_set[:train_num], train_set_y)
            
            trainacc = clf.score(feature_set[:train_num],train_set_y)
            testacc = clf.score(feature_set[train_num:],test_set_y)
            print 'training score:', trainacc
            print 'test score:', testacc
            writeacc.write(str(trainacc)+'\n')
            writeacc.write(str(testacc)+'\n')
#        log_prob_mat = clf.predict_log_proba(feature_set[train_num:])
#        log_prob = []
#        for i in xrange(len(test_set_y)):
#            log_prob.append(log_prob_mat[i,test_set_y[i]]) 
#        log_prob = numpy.asarray(log_prob)
#        
#        mses_valid = np.asarray(mses_valid)
#        plt.figure(figsize=(23,16));plt.scatter(mses_valid, log_prob);plt.xlabel('MSE');plt.ylabel('SVM_prob')
#        plt.savefig(savename+'_mse_prob.png', dpi = 100)
writeacc.close()