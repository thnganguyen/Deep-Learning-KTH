import numpy as np
import matplotlib.pyplot as plt
import time, random
import pickle


class Parameter:
    #mini-batch Gradient Descent parameters

    def __init__(self, n_batch, eta, n_epochs):
        self.n_batch = n_batch
        self.eta = eta
        self.n_epochs = n_epochs


# Exercise 1:
# Read in the data and initialize the parameters of the network

def unpickle(file):    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def loadBatch(filename):
    cifar = unpickle(filename)

    X = cifar[b'data'].T  # numpy array of size dxN     
    y = np.array(cifar[b'labels'])  # vector of lables of size 1xN
    Y = np.zeros((10,X.shape[1]))  # matrix of KxN
    Y[y,np.arange(len(y))] = 1

    return X,Y,y

def initializeParams(n_nodes,n_class,seed=None):
    W = []
    b = []
    n_layer = len(n_nodes)
    n_in = n_nodes[0]
    for i in range(n_layer):
        if i<n_layer-1:
            n_out = n_nodes[i+1]           
        else: 
            n_out = n_class  # last output        
        W.append(np.random.randn(n_out,n_in)/np.sqrt(n_in))
        b.append(np.zeros((n_out,1)))
        n_in = n_out
        
    return W,b


# Exercise 2:
# Compute the gradients for the network parameters

def computeScore(X,W,b):
    s = np.dot(W,X) + b
    return s

def ReLu(s):
    return np.maximum(0,s)

def SoftMax(s):
    p = np.exp(s)/np.expand_dims(np.exp(s).sum(axis=0),axis=0)  # matrix of size KxN
    return p

def evaluateClassifier(X,W,b):
    n_layer = len(W)
    h = X
    H = []
    for i in range(n_layer-1):
        s = computeScore(h,W[i],b[i])
        h = ReLu(s)
        H.append(h)
    s = computeScore(h,W[-1],b[-1])
    P = SoftMax(s)
    return H,P

def computeAccuracy(X,y,W,b):
    H,P = evaluateClassifier(X,W,b)
    y_pred = np.argmax(P, axis=0)
    acc = np.sum(y==y_pred)/(len(y)+0.0)
    return acc

def computeCost(X,Y,W,b,lamb):
    H,P = evaluateClassifier(X,W,b)
    loss = np.sum(-np.log(np.diag(np.dot(Y.T,P))))/(X.shape[1]+0.0)
    cost = loss + lamb*sum([np.einsum('ij,ij', wi, wi) for wi in W])
    return loss, cost
    
def computeGradients(X,Y,W,b,lamb):
    grad_W = []
    grad_b = []
    N = X.shape[1]
    H,P = evaluateClassifier(X,W,b)
    g = P-Y
    for i in range(len(W)-1):
        h = H[-1-i]
        grad_b.append(np.expand_dims(g.sum(axis=1),axis=1)/(N+0.0))
        grad_W.append(np.dot(g,h.T)/(N+0.0) + 2*lamb*W[-1-i])
             
        Ind = np.zeros(h.shape)
        Ind[h>0] = 1
        g = np.dot(W[-1-i].T,g)*Ind
    
    grad_b.append(np.expand_dims(g.sum(axis=1),axis=1)/(N+0.0))
    grad_W.append(np.dot(g,X.T)/(N+0.0) + 2*lamb*W[0])
    
    return list(reversed(grad_W)), list(reversed(grad_b))

def computeGradsNumSlow(X, Y, W, b, lamb, h):
    grad_W = []
    grad_b = []
    for k in range(len(W)):
        n_out,n_in = W[k].shape

        gw = np.zeros(W[k].shape)
        gb = np.zeros((n_out, 1))

        for i in range(len(b[k])):
            b[k][i] -= h
            l1,c1 = computeCost(X, Y, W, b, lamb)
            b[k][i] += 2*h
            l2,c2 = computeCost(X, Y, W, b, lamb)
            b[k][i] -= h
            gb[i] = (c2-c1)/(2*h)

        for i in range(n_out):
            for j in range(n_in):
                W[k][i,j] -= h
                l1,c1 = computeCost(X, Y, W, b, lamb)
                W[k][i,j] += 2*h
                l2,c2 = computeCost(X, Y, W, b, lamb)
                W[k][i,j] -= h
                gw[i,j] = (c2-c1)/(2*h)
        grad_W.append(gw)
        grad_b.append(gb)
    return grad_W, grad_b

def generateBatches(X,Y,n_batch):
    d,N = X.shape
    K = Y.shape[0]
    XBatch = np.zeros((d,n_batch,int(N/n_batch)))
    YBatch = np.zeros((K,n_batch,int(N/n_batch)))
    
    for i in range(int(N/n_batch)):
        i_start = i*n_batch
        i_end = (i+1)*n_batch
        XBatch[:,:,i] = X[:,i_start:i_end]
        YBatch[:,:,i] = Y[:,i_start:i_end]
    return XBatch, YBatch

def plotResults(x,y,name='loss',par='0'):
    fig = plt.figure()
    plt.plot(x,y[0],label='training '+name)
    plt.plot(x,y[1],label='validation '+name)
    plt.legend(loc=0)
    plt.xlabel('n_updates')
    plt.ylabel(name)
    fig.savefig('Figures/'+name+'_params'+par+'.pdf')
    plt.show()

# Plot W
def plotW(W,class_name,arg_par='0'):
    fig=plt.figure(figsize=(20,2))
    for i in range(10):
        im = W[i,:].reshape(3,32,32)
        img = np.transpose((im-np.min(im))/(np.max(im)-np.min(im)),(1,2,0))
        plt.subplot(1, 10, i+1)                 
        plt.title(class_name[i])
        plt.imshow(img)
        plt.axis('off')
                       
    plt.savefig('Figures/imW_params'+arg_par+'.pdf')
    plt.show()

# Check gradient computation
def checkGrads(X, Y, W, b, lamb, h=1e-6):
    eps = 1e-30
    error = 1e-6
    
    gW1,gb1 = computeGradsNumSlow(X,Y,W,b,lamb, h)
    gW2,gb2 = computeGradients(X,Y,W,b,lamb)
    
    total_error_w = 0
    total_error_b = 0
    max_w = np.zeros(len(W))
    max_b = np.zeros(len(W))
    
    for i in range(len(W)):
        error_w = np.abs(gW1[i]-gW2[i])/np.maximum(eps,np.abs(gW1[i])+np.abs(gW2[i]))
        error_b = np.abs(gb1[i]-gb2[i])/np.maximum(eps,np.abs(gb1[i])+np.abs(gb2[i]))
        total_error_w += np.sum(error_w>error)
        total_error_b += np.sum(error_b>error)
        max_w[i] = np.max(error_w)
        max_b[i] = np.max(error_b)
    print('The number of errors (relative error > 1e-6) of W:', total_error_w)
    print('The maximum of relative error of W:', np.max(max_w))
    print('The number of errors (relative error > 1e-6) of b:', total_error_b)
    print('The maximum of relative error of b:', np.max(max_b))


def plotCost(X,Y,y,W_train,b_train,lamb,step,name_par='0'):
    update = np.append(np.arange(0,len(W_train)-1,step),len(W_train)-1)
    n_update = len(update)
    loss_train = np.zeros(n_update)   
    loss_val = np.zeros(n_update)
    cost_train = np.zeros(n_update)   
    cost_val = np.zeros(n_update)
    error_train = np.zeros(n_update)   
    error_val = np.zeros(n_update)
    for ind,i in enumerate(update):
        loss_train[ind],cost_train[ind] = computeCost(X[0],Y[0],W_train[i],b_train[i],lamb)        
        loss_val[ind],cost_val[ind] = computeCost(X[1],Y[1],W_train[i],b_train[i],lamb)   
        
        error_train[ind] = computeAccuracy(X[0],y[0],W_train[i],b_train[i]) 
        error_val[ind] = computeAccuracy(X[1],y[1],W_train[i],b_train[i]) 
    plotResults(update,[loss_train, loss_val],'loss',name_par)
    plotResults(update,[cost_train, cost_val],'cost',name_par)
    plotResults(update,[error_train, error_val],'accuracy',name_par)


# Normalize dataset by mean and std of training set

def normalize(trainX, valX, testX):
    mean_train = np.mean(trainX,axis=1).reshape(-1,1)
    std_train = np.std(trainX,axis=1).reshape(-1,1)
    trainX = (trainX-mean_train)/std_train
    valX = (valX-mean_train)/std_train
    testX = (testX-mean_train)/std_train
    return trainX, valX, testX

def miniBatchGD(X,Y,y,GDparams,W,b,lamb):
    X_train = X[0]
    Y_train = Y[0]
    y_train = y[0]
    X_val = X[1]
    Y_val = Y[1]
    y_val = y[1]
    
    d,N = X_train.shape
    numBatches = int(np.floor(N/GDparams.n_batch))
    if type(GDparams.eta) is np.ndarray:
        eta = GDparams.eta
    else:
        eta = np.zeros(GDparams.n_epochs*numBatches)
        eta[:] = GDparams.eta
    
    W_train = []
    b_train = []
    W_star = W[:]
    b_star = b[:]
    W_train.append(W_star[:])
    b_train.append(b_star[:])
    
    XBatch,YBatch = generateBatches(X_train,Y_train,GDparams.n_batch)
    t = 0
    for i in range(GDparams.n_epochs):
        for j in range(numBatches):
            xTr = XBatch[:,:,j]
            yTr = YBatch[:,:,j]
            grad_w,grad_b = computeGradients(xTr,yTr,W_star,b_star,lamb)
            # update the weights and the thresholds
            for k in range(len(W)):
                W_star[k] = W_star[k] - eta[t]*grad_w[k]
                b_star[k] = b_star[k] - eta[t]*grad_b[k]
            t += 1
            W_train.append(W_star[:])
            b_train.append(b_star[:])

    return W_train,b_train

def cylicalEta(eta_min,eta_max,stepsize,n_cycle):
    iterations = np.arange(2*n_cycle*stepsize)
    cycle = np.floor(1 + iterations/(2*stepsize))
    x = np.abs(iterations/stepsize - 2*cycle + 1)
    eta = eta_min + (eta_max - eta_min)*np.maximum(0.0, (1-x))
    return eta


# Coarse-to-fine random search

def randomSearch(l_range,X,Y,y,n_cycle=2,n_lamb=10,name_par='0'):
    d,n = X[0].shape
    
    n_batch = 100
    eta_min = 1e-5
    eta_max = 1e-1
    n_s = int(2*np.floor(n/n_batch))
    eta = cylicalEta(eta_min,eta_max,n_s,n_cycle)
    n_epochs = int(2*n_cycle*n_s/n*n_batch)
    n_node = 50
    
    GDparams = Parameter(n_batch,eta,n_epochs)
    
    acc_val = np.zeros(n_lamb)
    lamb = np.zeros(n_lamb)
    W_best = []
    b_best = []
    n_nodes = np.array([d, n_node])
    n_layer = len(n_nodes)
    
    parameters = dict(n_batch=n_batch,n_epochs=n_epochs,eta_min=eta_min,eta_max=eta_max,step=n_s,
                      n_cycle=n_cycle,n_nodes=n_nodes,n_layer=n_layer)
                      for i in range(n_lamb):
                          W,b = initializeParams(n_nodes,10)
                              # set random lambda
                              l = l_range[0] + (l_range[1]-l_range[0])*np.random.rand(1)
                                  lamb[i] = 10**l
                                      
                                      W_train,b_train = miniBatchGD(X,Y,y,GDparams,W,b,lamb[i])
                                          
                                          acc_val[i] = computeAccuracy(X[1],y[1],W_train[-1],b_train[-1])
                                              W_best.append(W_train[-1])
                                                  b_best.append(b_train[-1])
                                              parameters['lambda'] = lamb
result = dict(parameters=parameters, accuracy=acc_val, weight=W_best, threshold=b_best)
    with open('search_'+name_par+'.txt', 'wb') as fp:
        pickle.dump(result, fp)


# Load data
path = 'Datasets-py/'
XBatch_1,YBatch_1,yBatch_1 = loadBatch(path+'data_batch_1')  # Training set
XBatch_2,YBatch_2,yBatch_2 = loadBatch(path+'data_batch_2')  # Validation set
XBatch_test,YBatch_test,yBatch_test = loadBatch(path+'test_batch')    # Test set

XTr,XVa, XTe = normalize(XBatch_1, XBatch_2, XBatch_test)
YTr = YBatch_1
YVa = YBatch_2
YTe = YBatch_test
yTr = yBatch_1
yVa = yBatch_2
yTe = yBatch_test


names = unpickle(path+'batches.meta')[b'label_names']  # list if size (10,1)

K = len(names)
class_name = []
for i in range(K):
    class_name.append(names[i].decode("utf-8"))

# Gradient check
X = XTr[:20,:5]
Y = YTr[:,:5]
n_nodes = np.array([X.shape[0], 50])

W,b = initializeParams(n_nodes,10)
checkGrads(X, Y, W, b, 0.01, 1e-5)


# Sanity check
X = [XTr[:,:100], XVa[:,:100]]
Y = [YTr[:,:100], YVa[:,:100]]
y = [yTr[:100], yVa[:100]]

n_nodes = np.array([X[0].shape[0], 50])


W,b = initializeParams(n_nodes,10)
lamb = 0
GDparams = Parameter(100,0.01,200)

W_train,b_train = miniBatchGD(X,Y,y,GDparams,W,b,lamb)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_train[-1],b_train[-1])*100,'%')

name_par = 'Ex1'
step = 1

plotCost(X,Y,y,W_train,b_train,lamb,step,name_par)

plotW(W_train[-1][0],class_name,name_par)


# Exercise 3:
# Train the network with cyclical learning rate

eta_min = 1e-5
eta_max = 1e-1
n_s = 500
n_cycle = 3
eta = cylicalEta(eta_min,eta_max,n_s,n_cycle)
plt.plot(range(len(eta)),eta)
plt.show()


X = [XTr, XVa]
Y = [YTr, YVa]
y = [yTr, yVa]

n_nodes = np.array([X[0].shape[0], 50])

W,b = initializeParams(n_nodes,K)

eta_min = 1e-5
eta_max = 1e-1
n_s = 500
n_cycle = 1
eta = cylicalEta(eta_min,eta_max,n_s,n_cycle)
n_batch = 100
lamb = 0.01

GDparams = Parameter(100,eta,10)

start = time.time()
W_train,b_train = miniBatchGD(X,Y,y,GDparams,W,b,lamb)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_train[-1],b_train[-1])*100,'%')

name_par = 'Ex3_0'
step = 100

plotCost(X,Y,y,W_train,b_train,lamb,step,name_par)

plotW(W_train[-1][0],class_name,name_par)


# Exercise 4:
# Train your network for real

W,b = initializeParams(n_nodes,K)

eta_min = 1e-5
eta_max = 1e-1
n_s = 800
n_cycle = 3
eta = cylicalEta(eta_min,eta_max,n_s,n_cycle)
n_batch = 100
lamb = 0.01
n_epochs = int(2*n_cycle*n_s/X[0].shape[1]*n_batch)

GDparams = Parameter(n_batch,eta,n_epochs)

start = time.time()
W_train,b_train = miniBatchGD(X,Y,y,GDparams,W,b,lamb)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_train[-1],b_train[-1])*100,'%')

name_par = 'Ex4_0'
step = 100

plotCost(X,Y,y,W_train,b_train,lamb,step,name_par)

plotW(W_train[-1][0],class_name,name_par)



# Coarse-to-fine random search

# Use all data to train
path = 'Datasets-py/'
XTr_1,YTr_1,yTr_1 = loadBatch(path+'data_batch_1')  # Training set
XTr_2,YTr_2,yTr_2 = loadBatch(path+'data_batch_2') 
XTr_3,YTr_3,yTr_3 = loadBatch(path+'data_batch_3')
XTr_4,YTr_4,yTr_4 = loadBatch(path+'data_batch_4')
XTr_5,YTr_5,yTr_5 = loadBatch(path+'data_batch_5')
XTrain = np.concatenate((XTr_1, XTr_2, XTr_3, XTr_4, XTr_5), axis=1)
YTr = np.concatenate((YTr_1, YTr_2, YTr_3, YTr_4, YTr_5), axis=1)
yTr = np.concatenate((yTr_1, yTr_2, yTr_3, yTr_4, yTr_5))

XValid = XTrain[:,-5000:] 
YVa = YTr[:,-5000:]
yVa = yTr[-5000:]
XTrain = XTrain[:,:-5000] 
YTr = YTr[:,:-5000]
yTr = yTr[:-5000]

XTest,YTe,yTe = loadBatch(path+'test_batch')    # Test set

XTr, XVa, XTe = normalize(XTrain, XValid, XTest)

names = unpickle(path+'batches.meta')[b'label_names']  # list if size (10,1)

K = len(names)
class_name = []
for i in range(K):
    class_name.append(names[i].decode("utf-8"))


# Coarse search

X = [XTr, XVa]
Y = [YTr, YVa]
y = [yTr, yVa]


l_range = np.array([-5, -1])
start = time.time()
randomSearch(l_range,X,Y,y,n_cycle=2,n_lamb=20,name_par='lamb_1')
print('Execution time: ',time.time()-start)


res = unpickle('search_lamb_1.txt')
print(res['accuracy'])
print(res['parameters']['lambda'])


la = res['parameters']['lambda']
ac = res['accuracy']
print(la[ac>0.51])
print(la[ac<=0.51])


print('The 3 best perfomances are: ',ac[ac.argsort()[-3:][::-1]])
print('The corresponding lambdas to the best ones: ', res['parameters']['lambda'][ac.argsort()[-3:][::-1]])


print('The best test accuracy is: ',ac.max())
print('The corresponding parameters to the best accuracy: ',res['parameters'], res['parameters']['lambda'][np.argmax(ac)])
print(res['parameters']['eta_min'])


# Fine search: base on the result of the previous task to narrow lambda range. Train more circle
l_range = np.array([-4, -2])
start = time.time()
randomSearch(l_range,X,Y,y,n_cycle=3,n_lamb=20,name_par='lamb_3')
print('Execution time: ',time.time()-start)

res = unpickle('search_lamb_3.txt')
la = res['parameters']['lambda']
ac = res['accuracy']
print(ac)
print(la)

print(la[ac>0.522])
print(la[ac<=0.522])

print('The 3 best perfomances are: ',ac[ac.argsort()[-3:][::-1]])
print('The corresponding lambdas to the best ones: ', res['parameters']['lambda'][ac.argsort()[-3:][::-1]])

print('The best test accuracy is: ',ac.max())
print('The corresponding parameters to the best accuracy: ',res['parameters'], res['parameters']['lambda'][np.argmax(ac)])
print(res['parameters']['eta_min'])


# Train the network with found lambda

XTrain = np.concatenate((XTr_1, XTr_2, XTr_3, XTr_4, XTr_5), axis=1)
YTr = np.concatenate((YTr_1, YTr_2, YTr_3, YTr_4, YTr_5), axis=1)
yTr = np.concatenate((yTr_1, yTr_2, yTr_3, yTr_4, yTr_5))

XValid = XTrain[:,-1000:] 
YVa = YTr[:,-1000:]
yVa = yTr[-1000:]
XTrain = XTrain[:,:-1000] 
YTr = YTr[:,:-1000]
yTr = yTr[:-1000]

XTr, XVa, XTe = normalize(XTrain, XValid, XTest)

X = [XTr, XVa]
Y = [YTr, YVa]
y = [yTr, yVa]


d,n = X[0].shape
n_node = 50
n_nodes = np.array([d, n_node])

W,b = initializeParams(n_nodes,K)
n_batch = 100

eta_min = 1e-5
eta_max = 1e-1
n_s = int(2*np.floor(n/n_batch))
n_cycle = 3
eta = cylicalEta(eta_min,eta_max,n_s,n_cycle)
lamb = res['parameters']['lambda'][np.argmax(ac)]
n_epochs = int(2*n_cycle*n_s/X[0].shape[1]*n_batch)

GDparams = Parameter(n_batch,eta,n_epochs)

start = time.time()
W_train,b_train = miniBatchGD(X,Y,y,GDparams,W,b,lamb)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_train[-1],b_train[-1])*100,'%')


# Plot

step = 100
name_par = 'best_3cycle'

plotCost(X,Y,y,W_train,b_train,lamb,step,name_par)

plotW(W_train[-1][0],class_name,name_par)
