# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:33:25 2016

@author: Stephen
"""
import theano
import theano.tensor as T
import numpy
from itertools import izip
import numpy as np
from collections import OrderedDict

def create_shared(x):
    return theano.shared(numpy.asarray(x,dtype=theano.config.floatX),
                                 borrow=True)
                                 
def RMSprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def AdaDelta(cost, parameters,rho = 0.95,eps = 1e-4):
	# create variables to store intermediate updates
    gradients = T.grad(cost, parameters)
    gradients_sq = [ create_shared(numpy.zeros(p.get_value().shape)) for p in parameters ]
    deltas_sq = [ create_shared(numpy.zeros(p.get_value().shape)) for p in parameters ]
     
    	# calculates the new "average" delta for the next iteration
    gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]
     
    	# calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]
     
    	# calculates the new "average" deltas for the next step.
    deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]
     
    	# Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
    parameters_updates = [(p,p - d) for p,d in izip(parameters,deltas)]
    updates = gradient_sq_updates + deltas_sq_updates + parameters_updates 
    return   updates
    
class rmsprop(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params):
        self.running_square_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]
        self.running_avg_ = [theano.shared(np.zeros_like(p.get_value()))
                             for p in params]
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, cost, params, learning_rate = 0.1, momentum= 0.95, rescale=5.):
        grads = T.grad(cost, params)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.9
        minimum_grad = 1e-4
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (scaling_num / scaling_den))
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(grad)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
            rms_grad = T.sqrt(new_square - new_avg ** 2)
            rms_grad = T.maximum(rms_grad, minimum_grad)
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad / rms_grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad / rms_grad
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates

def adagrad(cost, params, learning_rate=0.1, epsilon=1e-6):
    grads = T.grad(cost, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates
        
class sgd_nesterov(object):
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, cost, params, learning_rate = 0.1, momentum= 0.95):
        grads = T.grad(cost, params)
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class sgd(object):
    def __init__(self, params):
        pass

    def updates(self, cost, params, learning_rate = 0.1):
        grads = T.grad(cost, params)        
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            updates.append((param, param - learning_rate * grad))
        return updates
