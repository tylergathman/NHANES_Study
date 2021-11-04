import numpy as np

def PCA(X,num_dim=None):
    X_pca = X # placeholder

    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
    
    x,y = np.shape(X)
    
    for i in range(y): 
       X[:,i] = X[:,i] - np.mean(X[:,i])
       
    Cov_X = np.cov(np.transpose(X))
    
    w,v = np.linalg.eigh(Cov_X) 
    
    variance = 0
    i = 0
    

    
    # select the reduced dimensions that keep >95% of the variance
    if num_dim is None:

        while variance < 0.95: 
            variance = variance + w[len(w)-i-1] / np.sum(w)
            i = i + 1

        
        dimens = np.fliplr(v[:,(len(v)-i):len(v)])
        num_dim = i 


    else: 
        variance = variance + w[len(w)-num_dim-1] / np.sum(w)
        dimens = np.fliplr(v[:,(len(v)-num_dim):len(v)])

    



    X_pca = np.matmul(np.transpose(dimens),np.transpose(X))
    
    X_pca = np.transpose(X_pca) 
    # project the high-dimensional data to low-dimensional one

    return X_pca, num_dim
