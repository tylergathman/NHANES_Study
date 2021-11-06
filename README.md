# NHANES_Study

Repository for scripts that implement PCA, clustering, or other computational methods and the raw data. NHANES_Automated_PCA_Kmeans() implements myPCA() and myKmeans() for a given set of NHANES datasets in a given folder path. NHANES_Automated_PCA_Kmeans() takes in the NHANES SAS .XPT files, converts to data frame, checks to make sure every patient has the desired variables, and performs three operations. (1) k-means clustering for a given k clusters, (1) 1-dimensional PCA prior to k-means clustering, and (2) Multi-dimensional PCA prior to k-means clustering 
