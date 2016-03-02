# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 17:00:54 2016

@author: Stephen
"""
import os, numpy
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import tile_raster_images
import PIL.Image
from sklearn import cluster
import matplotlib.pyplot as plt
plt.ion()
nt = 7
dur = 1000
fr = 1000
hid = 400     
   
pivot = 'numTones_'+str(nt)+'_Duration_ms_'+str(dur)+'_FreqResolution_mHz_'+str(fr)
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
savename = 'CAE_'+ fname + '_bsp_10000_hid_sigmoid_'+str(hid)
#cddt = [176, 391, 100, 8, 151, 144, 250, 91, 81, 21, 222, 0] #21 1000 1000
#cddt = [160, 289, 106, 176, 339, 25, 283, 60, 84, 2, 21, 0] #21 4000 250
#cddt = [31, 351, 68, 347, 24, 176, 1, 36, 0] #21 250 4000
cddt = [26, 73, 5, 0] #7 1000 1000
#cddt = [54, 71, 0] #7 4000 250
#cddt = [55, 377, 0] #7 250 4000
outfolder = 'cA_plots/'

W= numpy.load('cA_weights/'+savename+'_L1_weight.npz')
W = W['W_l1'].T
length = 1
height = W.shape[0]

clt = cluster.KMeans(n_clusters = len(cddt),  random_state = 1306)
#clt = cluster.AgglomerativeClustering(n_clusters = len(cddt), affinity='l1', linkage = 'complete')
clt.fit(W.T)
Wclt = clt.cluster_centers_.T
Wclt = numpy.flipud(Wclt)
cbarmin = np.min(Wclt)
cbarmax = np.max(Wclt)
image = PIL.Image.fromarray(tile_raster_images( X=Wclt.T, img_shape=(height, length), tile_shape=(1, len(cddt)), tile_spacing=(1, 1)))
fig, ax = plt.subplots(figsize=(23,16));im = ax.imshow(image, extent=[0,len(cddt)*10,mintick,maxtick], interpolation='none');
divider = make_axes_locatable(ax);cax = divider.append_axes("right", size="5%", pad=0.1);
cbar = fig.colorbar(im, cax = cax);cbarlen = len(cbar.ax.get_yticklabels());
cbarrange = np.arange(cbarmin, cbarmax + (cbarmax-cbarmin)/cbarlen, (cbarmax-cbarmin)/cbarlen);cbar.ax.set_yticklabels(np.round(cbarrange,5))
plt.suptitle('Cluster center by Kmeans, '+str(len(cddt))+'maps,'+pivot)
ax.set_xticklabels([])
ax.set_xlabel(range(len(cddt)))
#plt.savefig(outfolder+'Kmeans_'+pivot+'.png', dpi=200)

W_ = numpy.flipud(W)[:,cddt]#.reshape(height, length)
cbarmin = np.min(W_)
cbarmax = np.max(W_)
image = PIL.Image.fromarray(tile_raster_images( X=W_.T, img_shape=(height, length), tile_shape=(1, len(cddt)), tile_spacing=(1, 1)))
fig, ax = plt.subplots(figsize=(23,16));im = ax.imshow(image, extent=[0,len(cddt)*10,mintick,maxtick], interpolation='none');
divider = make_axes_locatable(ax);cax = divider.append_axes("right", size="5%", pad=0.1);
cbar = fig.colorbar(im, cax = cax);cbarlen = len(cbar.ax.get_yticklabels());
cbarrange = np.arange(cbarmin, cbarmax + (cbarmax-cbarmin)/cbarlen, (cbarmax-cbarmin)/cbarlen);cbar.ax.set_yticklabels(np.round(cbarrange,5))
plt.suptitle('Selected features by proned tree, '+str(len(cddt))+'maps, '+pivot)
ax.set_xticklabels([])
ax.set_xlabel(clt.labels_[cddt])
#plt.savefig(outfolder+'C45_'+pivot+'.png', dpi=200)
