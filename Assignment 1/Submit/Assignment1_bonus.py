import numpy as np
import matplotlib.pyplot as plt
import time, random
from sklearn import decomposition

class Parameter:
    "mini-batch Gradient Descent parameters"

    def __init__(self, n_batch, eta, n_epochs):
        self.n_batch = n_batch
        self.eta = eta
        self.n_epochs = n_epochs

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def loadBatch(filename):
    cifar = unpickle(filename)

    X = cifar[b'data'].T/255.0  # numpy array of size dxN    
    y = np.array(cifar[b'labels'])  # vector of lables of size 1xN
    Y = np.zeros((10,X.shape[1]))  # matrix of KxN
    Y[y,np.arange(len(y))] = 1

    return X,Y,y


def evaluateClassifier(X,W,b):
    s = np.dot(W,X) + b
    p = np.exp(s)/np.expand_dims(np.exp(s).sum(axis=0),axis=0)  # matrix of size KxN
    return p


def Hinge_loss(X,y,W,b):
    N = len(y)
    scores = np.dot(W,X) + b  # matrix of size KxN
    score_y = scores[y,np.arange(N)]  # vector of size 1xN
    loss = np.maximum(0, scores - score_y[np.newaxis,:] + 1)
    loss[y,np.arange(N)] = 0  # set 0 at score_y
    
    return loss

def computeCost(X,Y,y,W,b,lamb,loss_func='softmax'):
    if loss_func=='softmax':
        P = evaluateClassifier(X,W,b)
        J = np.sum(-np.log(np.diag(np.dot(Y.T,P))))/(X.shape[1]+0.0) + lamb*np.sum(W*W)
    if loss_func=='svm':
        loss = Hinge_loss(X,y,W,b)
        J = np.sum(loss)/(X.shape[1]+0.0) + 0.5*lamb*np.sum(W*W)
    return J


def computeAccuracy(X,y,W,b,loss_func='softmax'):
    if loss_func=='softmax':
        P = evaluateClassifier(X,W,b)
        y_pred = np.argmax(P, axis=0)
    if loss_func=='svm':
        scores = np.dot(W,X) + b
        y_pred = np.argmax(scores, axis=0)
        
    acc = np.sum(y==y_pred)/(len(y)+0.0)
    return acc

def computeGradients(X,Y,y,W,b,lamb,loss_func='softmax'):
    N = X.shape[1]
    if loss_func=='softmax':
        P = evaluateClassifier(X,W,b)
        g = P-Y
        lamb *= 2
        
    if loss_func=='svm':
        loss = Hinge_loss(X,y,W,b)
        g = np.zeros(loss.shape)
        g[loss > 0.0] = 1
        count = np.sum(g, axis=0)
        g[y,np.arange(N)] = -count
        
    grad_b = np.expand_dims(g.sum(axis=1),axis=1)/(N+0.0)
    grad_w = np.dot(g,X.T)/(N+0.0) + lamb*W               
    return grad_w, grad_b


# Compute analytic gradient
def computeGradsNumSlow(X, Y, y, W, b, lamb, h=1e-6, loss_func='softmax'):
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
        b[i] -= h
        c1 = computeCost(X, Y, y, W, b, lamb, loss_func)
        b[i] += 2*h
        c2 = computeCost(X, Y, y, W, b, lamb, loss_func)
        b[i] -= h
        grad_b[i] = (c2-c1)/(2*h)

    for i in range(no):
        for j in range(d):
            W[i,j] -= h
            c1 = computeCost(X, Y, y, W, b, lamb, loss_func)
            W[i,j] += 2*h
            c2 = computeCost(X, Y, y, W, b, lamb, loss_func)
            W[i,j] -= h
    
            grad_W[i,j] = (c2-c1)/(2*h)
        
    return grad_W, grad_b


def generateBatches(X,Y,y,n_batch):
    d,N = X.shape
    K = Y.shape[0]
    XBatch = np.zeros((d,n_batch,int(N/n_batch)))
    YBatch = np.zeros((K,n_batch,int(N/n_batch)))
    yBatch = np.zeros((n_batch,int(N/n_batch)))
    
    for i in range(int(N/n_batch)):
        i_start = i*n_batch
        i_end = (i+1)*n_batch
        XBatch[:,:,i] = X[:,i_start:i_end]
        YBatch[:,:,i] = Y[:,i_start:i_end]
        yBatch[:,i] = y[i_start:i_end]
    YBatch = YBatch.astype(int)  # convert to int in case the type is changed
    yBatch = yBatch.astype(int)
    return XBatch, YBatch, yBatch

def plotResults(x,y,name='loss',par='0'):
    fig = plt.figure()
    plt.plot(x,y[0],label='training '+name)
    plt.plot(x,y[1],label='validation '+name)
    plt.legend(loc=0)
    plt.xlabel('n_epochs')
    plt.ylabel(name)
    fig.savefig('Figures/'+name+'_params'+par+'.pdf')
    plt.show()


# Check gradient computation    
def checkGrads(X, Y, y, W, b, lamb, h=1e-6, loss_func='softmax'):
    eps = 1e-30
    error = 1e-6
    
    gW1,gb1 = computeGradsNumSlow(X, Y, y, W, b, lamb, h, loss_func)
    gW2,gb2 = computeGradients(X,Y,y,W,b,lamb,loss_func)
    print(gW1,gb1)
    print(gW2,gb2)
    
    error_w = np.abs(gW1-gW2)/np.maximum(eps,np.abs(gW1)+np.abs(gW2))
    error_b = np.abs(gb1-gb2)/np.maximum(eps,np.abs(gb1)+np.abs(gb2))
    print('The number of error (relative error > 1e-6) of W:', np.sum(error_w>error))
    print('The maximum of relative error of W:', np.max(error_w))
    print('The number of error (relative error > 1e-6) of b:', np.sum(error_b>error))
    print('The maximum of relative error of b:', np.max(error_b))


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
    

def miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,loss_func='softmax',eta_factor=1,arg_par='0',shuf=False,stop=False,plot=True):
    X_train = X[0]
    Y_train = Y[0]
    y_train = y[0]
    X_val = X[1]
    Y_val = Y[1]
    y_val = y[1]
    
    d,N = X_train.shape
    numBatches = int(N/GDparams.n_batch)
    
    # randomly initialize the weights and the thresholds
    W_star = W
    b_star = b
    
    if stop:
        acc_err = 0.0
        acc_val_pre = 0.0
    if plot:
        loss_train = np.zeros(GDparams.n_epochs)   
        loss_val = np.zeros(GDparams.n_epochs)
        error_train = np.zeros(GDparams.n_epochs)   
        error_val = np.zeros(GDparams.n_epochs) 
    if not shuf:
        XBatch,YBatch,yBatch = generateBatches(X_train,Y_train,y_train,GDparams.n_batch)
    
    for i in range(GDparams.n_epochs):
        # An improvements using shuffle to reorder training data
        if shuf:
            permute = list(range(N))
            random.shuffle(permute)
            X_train = X_train[:,permute]
            Y_train = Y_train[:,permute]
            y_train = y_train[permute]
            XBatch,YBatch,yBatch = generateBatches(X_train,Y_train,y_train,GDparams.n_batch)

        for j in range(numBatches):
            xTr = XBatch[:,:,j]
            yTr = YBatch[:,:,j]
            labelTr = yBatch[:,j]
            grad_w,grad_b = computeGradients(xTr,yTr,labelTr,W_star,b_star,lamb,loss_func)
            # update the weights and the thresholds
            W_star = W_star - GDparams.eta*grad_w
            b_star = b_star - GDparams.eta*grad_b
           
        GDparams.eta *= eta_factor # decay learning rate a factor
        
        if stop and i%5==0:
            acc_val = computeAccuracy(X_val,y_val,W_star,b_star,loss_func)
            if acc_val-acc_val_pre<acc_err and acc_val>0.3:
                print('Stop at epoch',i)
                break
            acc_val_pre = acc_val
            
        if plot:
            loss_train[i] = computeCost(X_train,Y_train,y_train,W_star,b_star,lamb,loss_func)        
            loss_val[i] = computeCost(X_val,Y_val,y_val,W_star,b_star,lamb,loss_func)
        
            error_train[i] = 1-computeAccuracy(X_train,y_train,W_star,b_star,loss_func)
            error_val[i] = 1-computeAccuracy(X_val,y_val,W_star,b_star,loss_func)
    if stop:
        GDparams.n_epochs = i
        loss_train = loss_train[:i]
        loss_val = loss_val[:i]
        error_train = error_train[:i]
        error_val = error_val[:i]
        
    if plot:
        plotResults(range(GDparams.n_epochs),[loss_train, loss_val],'loss',arg_par)
        plotResults(range(GDparams.n_epochs),[error_train, error_val],'error',arg_par)
    
    return W_star,b_star



# Load data
XTr,YTr,yTr = loadBatch('cifar-10-batches-py/data_batch_1')
XVa,YVa,yVa = loadBatch('cifar-10-batches-py/data_batch_2')
XTe,YTe,yTe = loadBatch('cifar-10-batches-py/test_batch')    # Test set
names = unpickle("cifar-10-batches-py/batches.meta")[b'label_names']  # list if size (10,1)

K = len(names)
class_name = []
for i in range(K):
    class_name.append(names[i].decode("utf-8"))


X = [XTr, XVa]
Y = [YTr, YVa]
y = [yTr, yVa]

d = XTr.shape[0]
#Initialize the parameters W and b
W = np.random.randn(K,d) * 0.01
b = np.random.randn(K,1) * 0.01


# (b) - Train for a longer time and use early stop
lamb = 0
GDparams = Parameter(100,0.01,200)
name_par = '(b)1'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,stop=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (b)
lamb = 0.1
GDparams = Parameter(100,0.01,200)
name_par = '(b)2'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,stop=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (d) - Decay the learning rate by a factor 0.9 after each epoch
lamb = 0
GDparams = Parameter(100,0.1,40)
name_par = '(d)0'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (d)
lamb = 0
GDparams = Parameter(100,0.01,40)
name_par = '(d)1'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (d)
lamb = 0.1
GDparams = Parameter(100,0.01,40)
name_par = '(d)2'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# (d) + (b)
lamb = 0
GDparams = Parameter(100,0.01,200)
name_par = '(bd)1'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par,stop=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (d) + (b)
lamb = 0.1
GDparams = Parameter(100,0.01,200)
name_par = '(bd)2'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par,stop=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (g) -> Shuffle the order of the training examples at the beginning of every epoch
lamb = 0
GDparams = Parameter(100,0.1,40)
name_par = '(g)0'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,shuf=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (g)
lamb = 0
GDparams = Parameter(100,0.01,40)
name_par = '(g)1'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,shuf=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (g)
lamb = 0.1
GDparams = Parameter(100,0.01,40)
name_par = '(g)2'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,shuf=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (g)
lamb = 1
GDparams = Parameter(100,0.01,40)
name_par = '(g)3'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,shuf=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (g) + (d)
lamb = 0
GDparams = Parameter(100,0.1,40)
name_par = '(dg)0'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par,shuf=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (g) + (d)
lamb = 0
GDparams = Parameter(100,0.01,40)
name_par = '(dg)1'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par,shuf=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (g) + (d)
lamb = 0.1
GDparams = Parameter(100,0.01,40)
name_par = '(dg)2'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par,shuf=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# (e) - Use Xavier initialization
sigma = 1/np.sqrt(d)
W = W*sigma/0.01
b = b*sigma/0.01

# (e)
lamb = 0
GDparams = Parameter(100,0.01,40)
name_par = '(e)1'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)

# (e)
lamb = 0.1
GDparams = Parameter(100,0.01,40)
name_par = '(e)2'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# (e)
lamb = 1
GDparams = Parameter(100,0.01,40)
name_par = '(e)0'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# (e)
lamb = 0
GDparams = Parameter(100,0.1,40)
name_par = '(e)0'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# (d) + (e)
lamb = 0
GDparams = Parameter(100,0.1,40)
name_par = '(de)0'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# (d) + (e)
lamb = 0
GDparams = Parameter(100,0.01,40)
name_par = '(de)1'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# (b) + (d) + (e)
lamb = 0
GDparams = Parameter(100,0.1,200)
name_par = '(bde)0'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par,stop=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# (b) + (d) + (e)
lamb = 0
GDparams = Parameter(100,0.01,200)
name_par = '(bde)1'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par,stop=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# (b) + (d) + (e)
lamb = 0.1
GDparams = Parameter(100,0.01,200)
name_par = '(bde)2'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par,stop=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# (d) + (e) + (g)
W = W*sigma/0.01
b = b*sigma/0.01

lamb = 0
GDparams = Parameter(100,0.1,40)
name_par = '(deg)2'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,eta_factor=0.9,arg_par=name_par,shuf=True)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# SVM
# Compare softmax+loss-entropy (smle) and Hinge loss (svm)
W = np.random.randn(K,d) * 0.01
b = np.random.randn(K,1) * 0.01


# Compare
lamb = 0
GDparams = Parameter(100,0.1,40)

# svm
name_par = 'svm-0'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,loss_func='svm')
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star,loss_func='svm')*100,'%')

plotW(W_star,class_name,name_par)

# smle
name_par = 'smle-0'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# Compare
lamb = 0
GDparams = Parameter(100,0.01,40)

# svm
name_par = 'svm-1'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,loss_func='svm')
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star,loss_func='svm')*100,'%')

plotW(W_star,class_name,name_par)

# smle
name_par = 'smle-1'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# Compare
lamb = 0.1
GDparams = Parameter(100,0.01,40)
# svm
name_par = 'svm-2'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,loss_func='svm')
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star,loss_func='svm')*100,'%')

plotW(W_star,class_name,name_par)

# smle
name_par = 'smle-2'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# Compare
lamb = 1
GDparams = Parameter(100,0.01,40)
# svm
name_par = 'svm-3'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,loss_func='svm')
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star,loss_func='svm')*100,'%')

plotW(W_star,class_name,name_par)

# smle
name_par = 'smle-3'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# Compare
lamb = 0
GDparams = Parameter(100,0.001,40)
# svm
name_par = 'svm-4'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,loss_func='svm')
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star,loss_func='svm')*100,'%')

plotW(W_star,class_name,name_par)

# smle
name_par = 'smle-4'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# Compare
lamb = 0.01
GDparams = Parameter(100,0.001,40)
# svm
name_par = 'svm-5'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,loss_func='svm')
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star,loss_func='svm')*100,'%')

plotW(W_star,class_name,name_par)

# smle
name_par = 'smle-5'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)


# Compare
lamb = 0.01
GDparams = Parameter(100,0.0001,40)
# svm
name_par = 'svm-6'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par,loss_func='svm')
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star,loss_func='svm')*100,'%')

plotW(W_star,class_name,name_par)

# smle
name_par = 'smle-6'

start = time.time()
W_star,b_star = miniBatchGD_improve(X,Y,y,GDparams,W,b,lamb,arg_par=name_par)
print('Execution time: ',time.time()-start)
print('The accuracy on test set: ',computeAccuracy(XTe,yTe,W_star,b_star)*100,'%')

plotW(W_star,class_name,name_par)
