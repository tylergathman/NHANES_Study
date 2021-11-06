#import libraries
import numpy as np
import math

class Kmeans:
    def __init__(self,k=8): # k is number of clusters
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005] # indices for the samples


        num_iter = 0 # number of iterations for convergence

        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False

        firstPass = True
        a,b = np.shape(X)
        updated_centers = np.zeros((self.num_cluster,b)) 


        # iteratively update the centers of clusters till convergence
        while not is_converged:

            # iterate through the samples and compute their cluster assignment (E step)
            updated_centers_count = np.zeros(self.num_cluster) 
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                if firstPass: 
                    c1 = np.linalg.norm(X[i] - X[init_idx[0]])**2
                    c2 = np.linalg.norm(X[i] - X[init_idx[1]])**2
                    c3 = np.linalg.norm(X[i] - X[init_idx[2]])**2
                    c4 = np.linalg.norm(X[i] - X[init_idx[3]])**2
                    c5 = np.linalg.norm(X[i] - X[init_idx[4]])**2
                    c6 = np.linalg.norm(X[i] - X[init_idx[5]])**2
                    c7 = np.linalg.norm(X[i] - X[init_idx[6]])**2
                    c8 = np.linalg.norm(X[i] - X[init_idx[7]])**2
                else:  
                    c1 = np.linalg.norm(X[i] - updated_centers[0])**2
                    c2 = np.linalg.norm(X[i] - updated_centers[1])**2
                    c3 = np.linalg.norm(X[i] - updated_centers[2])**2
                    c4 = np.linalg.norm(X[i] - updated_centers[3])**2
                    c5 = np.linalg.norm(X[i] - updated_centers[4])**2
                    c6 = np.linalg.norm(X[i] - updated_centers[5])**2
                    c7 = np.linalg.norm(X[i] - updated_centers[6])**2
                    c8 = np.linalg.norm(X[i] - updated_centers[7])**2
                    
                  
                    
                c_vector = [c1, c2, c3, c4, c5, c6, c7, c8]
                
                
                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                
                cluster_assign = np.where(c_vector == np.amin(c_vector))[0]
                cluster_assignment[i] = cluster_assign[0]
                                 
            
            
            # update the centers based on cluster assignment (M step)
            firstPass = False 
            updated_centers = np.zeros((self.num_cluster,b)) 
            updated_centers_count = np.zeros(self.num_cluster) 
            for i in range(len(X)): 
                cluster = cluster_assignment[i]
                updated_centers[cluster] = updated_centers[cluster] + X[i] 
                updated_centers_count[cluster] = updated_centers_count[cluster] + 1
            
            

            

            for i in range(len(updated_centers_count)): 
                updated_centers[i] = np.divide(updated_centers[i] , updated_centers_count[i])
                
            self.center = updated_centers
                
            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        # compute the information entropy for different clusters
        cluster_entropy = 0
        for i in range(self.num_cluster): 
            count_0 = 0
            count_9 = 0
            count_8 = 0
            total_count = 0
            for j in range(len(X)):
                if cluster_assignment[j] == i: 
                    if int(y[j]) == 0:
                        count_0 = count_0 + 1
                    if int(y[j]) == 8:
                        count_8 = count_8 + 1
                    if int(y[j]) == 9:
                        count_9 = count_9 + 1
            total_count = count_0 + count_8 + count_9 
            

            cluster_entropy = cluster_entropy + -1*((count_0 / total_count)*math.log(count_0 / total_count+0.000001,2)+ (count_8 / total_count)*math.log(count_8 / total_count+0.000001,2)+ (count_9 / total_count)*math.log(count_9 / total_count+0.000001,2))
        
        entropy = cluster_entropy / self.num_cluster # placeholder


        return num_iter, self.error_history, entropy

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for i in range(len(X)): 
           error = error + np.linalg.norm(X[i] - self.center[cluster_assignment[i]])**2
        
        return error

    def params(self):
        return self.center
