from __future__ import unicode_literals, print_function, division
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from scipy import sparse

from io import open
import unicodedata
import string, time, random
from itertools import count

import json
import re



class RNN(object):
    """
    A Vanilla Recurrent neural network.
    """

    def __init__(self, vocab_size=80 ,hidden_dim=100, 
                 weight_scale=0.01, dtype=np.float32,):
        """
        Initialize a new network.

        Inputs:
        - vocab_size: the total letters of the input
        - hidden_dim: Number of units to use in hidden layer
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - dtype: numpy datatype to use for computation.
        """
        self.h0 = np.zeros((hidden_dim , 1))
        
        self.params = {}
        self.dtype = dtype
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.weight_scale = weight_scale
                      
        self.init_sets()
    
    
    def init_sets(self):
        """
        Initialize model parameters
        """
        # The biases
        self.params['b'] = np.zeros((self.hidden_dim,1))  # m x 1
        self.params['c'] = np.zeros((self.vocab_size,1))  # K x 1
           
        # The weights
        self.params['U'] = self.weight_scale*np.random.randn(self.hidden_dim,self.vocab_size)  # m x K
        self.params['W'] = self.weight_scale*np.random.randn(self.hidden_dim,self.hidden_dim)  # m x m
        self.params['V'] = self.weight_scale*np.random.randn(self.vocab_size,self.hidden_dim)  # K x m

        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)
        
    
    ## Exercise 3: Synthesize text from your randomly initialized RNN
    def sample(self, x0, n=140):
        """
        - h0: m x 1 vector of hidden state at time 0 
        - x0: K x 1 first input vector
        - n: the length of the sequence to generate
        """
        if n>140:
            return("Length of sequence is at most 140! Please choose another length!")
        
        U, W, V = self.params['U'], self.params['W'], self.params['V']
        b, c = self.params['b'], self.params['c']
        
        xnext = np.zeros((len(x0),n))
        h, x = self.h0.copy(), x0
        for i in range(n):
            h = np.tanh(np.dot(W,h) + np.dot(U,x) + b)  # m x 1
            s = np.dot(V,h) + c  # K x 1
            p = np.exp(s-np.max(s))/np.sum(np.exp(s-np.max(s)))  # K x 1
            # find next x
            au = np.random.rand()
            ix = np.argwhere(np.cumsum(p)>au)[0]
            xnext[ix,i] = 1
            x = xnext[:,[i]]
        
        return xnext

    def loss(self, X, Y, mode='train'):
        """
        Evaluate loss and gradient for recurrent neural network.
        - y=None: for testing
        - h0 is mx1 array of initial hidden state
        """
        X = X.astype(self.dtype)
        Y = Y.astype(self.dtype)

        U, W, V = self.params['U'], self.params['W'], self.params['V']
        b, c = self.params['b'], self.params['c']
        
        _,N = X.shape
        
        # Forward pass
        ht, st, pt = {}, {}, {}
        ht[-1] = self.h0.copy()
        
        loss = 0
        for t in range(N):
            ht[t] = np.tanh(np.dot(W,ht[t-1]) + np.dot(U,X[:,[t]]) + b)  # m x 1
            st[t] = np.dot(V,ht[t]) + c  # K x 1
            pt[t] = np.exp(st[t]-np.max(st[t]))/np.sum(np.exp(st[t]-np.max(st[t])))
            loss += -np.log(np.dot(Y[:,[t]].T,pt[t]))
            
        if mode=='test':
            return loss
        
        # Backward pass
        grads = {}
        dU, dW, dV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
        db, dc = np.zeros_like(b), np.zeros_like(c)
        dh_next = np.zeros_like(self.h0)
        for t in reversed(range(N)):
            dst = pt[t] - Y[:,[t]]
            dV += np.dot(dst,ht[t].T)
            dc = dc + dst
            dht = np.dot(V.T,dst) + dh_next
            dat = dht*(1-ht[t]*ht[t])  # m x 1
            db = db + dat
            dU += np.dot(dat,X[:,[t]].T)  # m x K
            dW += np.dot(dat,ht[t-1].T)  # m x m
            dh_next = np.dot(W.T,dat)
               
        grads['U'], grads['W'], grads['V'] = dU, dW, dV
        grads['b'], grads['c'] = db, dc
        
        for k,v in grads.items():
            np.clip(v,-5,5,out=v)
        
        self.h0 = ht[N-1].copy()
        return loss, grads


# In[88]:


class Model(object):
    
    def __init__(self, model, data, **kwargs):
        
        self.model = model
        self.data = data

        # Unpack keyword arguments
        self.optimizer = kwargs.pop('optimizer', 'adagrad')
        self.optim_config = kwargs.pop('optim_config', {})
        self.seq_length = kwargs.pop('seq_length', 25)
        self.num_updates = kwargs.pop('num_updates', 10000)

        self.print_every = kwargs.pop('print_every', 100)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        self._reset()
    
    
    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_loss = 0
        self.best_params = {}
        self.loss_history = []
        self.smooth_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
    
    
    def train(self):
        """
        Run optimization to train the model.
        """
        hidden_dim = self.model.hidden_dim
        vocab_size = self.model.vocab_size
        seq_length = self.seq_length
        num_iterations = self.num_updates
        
        smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
        self.best_loss = smooth_loss
        pos = 0
        for t in range(num_iterations): 
            # preparing data to train
            X_chars = self.data[pos:pos+seq_length]
            end_tweet = X_chars.find("@\n")  # end-of-stweet
            if end_tweet>-1:  
                self.model.h0 = np.zeros((hidden_dim,1)) # reset h0
                pos = pos+end_tweet+2
                X_chars = self.data[pos:pos+seq_length]
            Y_chars = self.data[pos+1:pos+seq_length+1]
            X = one_hot_matrix(X_chars, vocab_size)
            Y = one_hot_matrix(Y_chars, vocab_size)
            # Print sample from the model now and then
            if self.verbose and t % self.print_every == 0:
                sample_ix = self.model.sample(X[:,[0]], 140)
                txt = ''.join(int2char[ix] for ix in np.argmax(sample_ix, axis=0))
                print('----\n %s \n----' % (txt, ))
            
            # Compute loss and gradient
            loss, grads = self.model.loss(X, Y)
            self.loss_history.append(loss)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001 
            self.smooth_loss_history.append(smooth_loss)
            # Print training smooth_loss and sample from the model now and then
            if self.verbose and t % self.print_every == 0: 
                print('(Iteration %d / %d) loss: %f' % (
                       t + 1, num_iterations, smooth_loss))
            
            # Perform a parameter update
            for p, w in self.model.params.items():
                dw = grads[p]
                config = self.optim_configs[p]
                next_w, next_config = globals()[self.optimizer](w, dw, config)
                self.model.params[p] = next_w
                self.optim_configs[p] = next_config
                                          
            pos += seq_length
                           
            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            if pos+seq_length+1>=len(self.data) or t==0 or (t == num_iterations - 1):
                self.model.h0 = np.zeros((hidden_dim,1)) # reset h0
                pos = 0
                # Keep track of the best model
                if smooth_loss < self.best_loss:
                    self.best_loss = smooth_loss
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params



def adagrad(w, dw, config=None):
    """
    Performs a variant of stochastic gradient descent, AdaGrad.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 0.1)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
 
    config['m'] += dw**2
    next_w = w - config['learning_rate']*dw/(np.sqrt(config['m'])+config['epsilon'])

    return next_w, config


# Read in the data

data = []
for year in range(2009,2019):
    with open('Data/condensed_'+str(year)+'.json') as json_file:
        data = data + json.load(json_file)

all_letters = string.printable + "’"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
                   c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn'
                   and c in all_letters
                   )

url_str = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
tweets = '@\n'.join(re.sub(url_str, '', unicodeToAscii(d['text']), flags=re.MULTILINE) for d in data)

with open('Data/trumptweet.txt', 'w', encoding='utf-8') as file:
    file.write(tweets)

fname = 'Data/trumptweet.txt'
tweet_data = open(fname, encoding='utf-8').read()

tweet_chars = ''.join(sorted({l for word in tweet_data for l in word}))

# Dictionary with each key is a character and value is its position in book_chars (1,94)
char2int = dict(zip(tweet_chars, count(0)))
# Dictionary with each key is a number in (1,94) and value is corresponding character
int2char = {v: k for k, v in char2int.items()}


def one_hot_matrix(chars, vocab_size):
    n = len(chars)
    x_one_hot = np.zeros((vocab_size,n))    
    x_one_hot[[char2int[ch] for ch in chars], range(n)] = 1
    return x_one_hot

def plotResults(x,y,name='loss',save_name=None):
    fig = plt.figure()
    plt.plot(x,y)
    plt.xlabel('n_update')
    plt.ylabel(name)
    if save_name is not None:
        fig.savefig('Figures/'+name+'_'+save_name+'.pdf')
    plt.show()


# Train RNN

net = RNN(vocab_size=len(tweet_chars))
model16_1 = Model(net, tweet_data,
              num_updates=5000000,
              seq_length=16,
              optimizer='adagrad',
              optim_config={
                  'learning_rate': 0.1},
              verbose=True, print_every=10000)

tic = time.time()
model16_1.train()
toc = time.time()
print('Execution time: ',toc-tic)
print(model16_1.best_loss)


x0 = one_hot_matrix(tweet_data[0], model16_1.model.vocab_size)
xnext = model16_1.model.sample(x0,140)
text = ''.join(int2char[ix] for ix in np.argmax(xnext, axis=0))
print(text)


n_updates = 5000000
x = np.arange(0,n_updates,20000)
loss = np.squeeze(np.array(model16_1.loss_history))[x]
plotResults(x,loss,name='smooth_loss',save_name='bonus_5M_updates_20000')
