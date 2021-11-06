import pyreadstat
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from Mykmeans import Kmeans
from MyPCA import PCA
import time

# Written by TJG on 11/5/21
# Pre-process SAS data for a dynamic dataset 

path = 'c:/NHANES_Study'
files = os.listdir(path)

a = {}

count = 0
for f in files: 
	split_tup = os.path.splitext(f)	  
	file_name = split_tup[0]
	file_extension = split_tup[1]
	if file_extension == '.XPT': 
		df, meta = pyreadstat.read_xport(f)
		df.set_index('SEQN',inplace=True)
		count = count + 1
		if count == 1: 
			result = df
		if count > 1: 
			result = pd.concat([result, df], axis=1, join="inner")

#result.to_csv('concat_from_sas.csv', index=False)


data=np.genfromtxt("Digits089.csv",delimiter=",")

X = result.values[0:3000,9:15]

X_dim = X.shape

found_nan = False
count = 0

for i in range (X_dim[0]-count): 
	found_nan = False 
	for j in range (X_dim[1]):
		if np.isnan(X[i-count,j]): 
			found_nan = True
	if found_nan: 
		X = np.delete(X, i-count, 0)
		count = count + 1

X_dim = X.shape
y = data[0:len(X),1]

print(X_dim) 
# Conduct computational analysis on pre-processed dataset

# read in data.



# apply kmeans algorithms to raw data
clf = Kmeans(k=8)
start = time.time()
num_iter, error_history, entropy_raw = clf.run_kmeans(X, y)
time_raw = time.time() - start

# plot the history of reconstruction error
fig = plt.figure()
plt.plot(np.arange(len(error_history)),error_history,'b-',linewidth=2)
fig.set_size_inches(10, 10)
fig.savefig('raw_data.png')
plt.show()

# apply kmeans algorithms to low-dimensional data (PCA) that captures >95% of variance
X_pca, num_dim = PCA(X)
clf = Kmeans(k=8)
start = time.time()
num_iter_pca, error_history_pca, entropy_pca = clf.run_kmeans(X_pca, y)
time_pca = time.time() - start

# plot the history of reconstruction error
fig1 = plt.figure()
plt.plot(np.arange(len(error_history_pca)),error_history_pca,'b-',linewidth=2)
fig1.set_size_inches(10, 10)
fig1.savefig('pca.png')
plt.show()

# apply kmeans algorithms to 1D feature obtained from PCA
X_pca, _ = PCA(X,1)
clf = Kmeans(k=8)
start = time.time()
num_iter_pca_1, error_history_pca_1, entropy_pca_1 = clf.run_kmeans(X_pca, y)
time_pca_1 = time.time() - start

# plot the history of reconstruction error
fig2 = plt.figure()
plt.plot(np.arange(len(error_history_pca_1)),error_history_pca_1,'b-',linewidth=2)
fig2.set_size_inches(10, 10)
fig2.savefig('pca_1d.png')
plt.show()

# print the average information entropy and number of iterations for convergence
print('Using raw data converged in %d iteration (%.2f seconds)' %(num_iter,time_raw))
print('Information entropy: %.3f' %entropy_raw)

print('#################')
print('Project data into %d dimensions with PCA converged in %d iteration (%.2f seconds)'%(num_dim,num_iter_pca,time_pca))
print('Information entropy: %.3f' %entropy_pca)

print('#################')
print('Project data into 1 dimension with PCA converged in %d iteration (%.2f seconds)'%(num_iter_pca_1,time_pca_1))
print('Information entropy: %.3f' %entropy_pca_1)
