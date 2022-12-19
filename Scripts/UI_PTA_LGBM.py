# import libraries 
import pyreadstat
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import shap 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from statsmodels.tools.tools import add_constant
from numpy import mean
from numpy import std
from scipy.special import boxcox, inv_boxcox
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import seaborn as sns
import imblearn 
from sklearn.datasets import make_classification 
from imblearn.over_sampling import RandomOverSampler
from scipy import stats
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

    
feature_variables = [ 

# DEMO
'RIAGENDR',
'RIDAGEYR',
#'RIDRETH3',
# 'DMQMILIZ', 

# 'DBQ700',

# # # 'INDFMPIR',


# # # # # MCQ 
# 'MCQ160C',
# 'MCQ160F',
# 'MCQ220',
# 'MCQ160B',
# 'MCQ160D',
# 'MCQ160E',

# # # # # ALC 
# 'ALQ101',

# # # # # SMP / BP    
'SMQ020',
# 'BPQ020',
# 'DIQ010',
'BMXBMI',

# # # # AUQ
'AUQ054',
# 'AUQ060',
# 'AUQ070',
# 'AUQ080',
# 'AUQ090',
# 'AUQ144','AUQ141',
# 'AUQ331','AUQ330',
# 'AUQ340',
# 'AUQ350',
# 'AUQ361', 'AUQ360',
# 'AUQ300','AUQ211',
# 'AUQ370',
# 'AUQ381', 'AUQ380',
# 'AUQ310',

# 'AUQ136','AUQ430',


# 'AUQ100',
# 'AUQ138','AUQ139',
# 'AUQ110','AUQ101',

# 'HIQ011',

'WTMEC2YR'



                     ]


# equiv = [['AUQ331','AUQ330'],['AUQ381', 'AUQ380'],['AUQ361', 'AUQ360'],
#           #['HIQ031A', 'HIQ031B'],
#           ['AUQ136','AUQ430'],['AUQ138','AUQ139'],['AUQ144','AUQ141'],
#           ['AUQ300','AUQ211'], ['AUQ110','AUQ101']]

right_variables = [ 'AUXU500R','AUXU1K1R','AUXU2KR','AUXU4KR'
                   
                   ]    


left_variables = [ 'AUXU500L','AUXU1K1L','AUXU2KL','AUXU4KL'
                   
                   ]

label_variables = right_variables + left_variables


paths = ['c:\CSCI_5521/hw4/NHANES_18',
    'c:\CSCI_5521/hw4/NHANES_16',
  'c:\CSCI_5521/hw4/NHANES_12',
'c:\CSCI_5521/hw4/NHANES_10',

    
         ] 

   
    
sorted_vars = []


for d in range(np.size(paths)): 
    path = paths[d]
    files = os.listdir(path)
    
    a = {}
    
    count = 0
    result = []
    for f in files: 
        split_tup = os.path.splitext(f)   
        file_name = split_tup[0]
        file_extension = split_tup[1]
        if file_extension == '.XPT': 
            df, meta = pyreadstat.read_xport(path+'/'+f)
            df.set_index('SEQN',inplace=True)
            count = count + 1
            if count == 1: 
                result = df
            if count > 1: 
                result = pd.concat([result, df], axis=1, join="outer")
        
    
    
    # Define variables of interest 
    
   
                    
    var_dim = np.shape(feature_variables)
    
    
    columns = result.columns
    col_dim = columns.shape
    sorted_vars = []
    selected = []
    temp = []

    
    first = True 
    for i in range(var_dim[0]): 
        for j in range(col_dim[0]):
            if columns[j] == feature_variables[i]: 
                sorted_vars.append(columns[j])                
                temp = result[feature_variables[i]]
                if first: 
                    selected = temp 
                else: 
                    selected = pd.concat([selected, temp], axis=1, join="inner")
                first = False 
    
  
    
    var_dim_label = np.shape(label_variables)
    selected_labels = []
    
    first = True             
    for j in range(col_dim[0]):
        for i in range(var_dim_label[0]): 
            if columns[j] == label_variables[i]:  
                temp = result[label_variables[i]]
                if first: 
                    selected_labels = temp 
                else: 
                    selected_labels = pd.concat([selected_labels, temp], axis=1, join="inner")
                first = False 
                
                
                
    #################
                
    combined = pd.concat([selected, selected_labels], axis=1, join="inner")
    

    
    
    if d == 0: 
        cumul = combined
    else: 
        cumul = pd.concat([combined, cumul], axis=0)        

#################################
labl = feature_variables + right_variables + left_variables 
cumul = cumul[labl]





cumul = cumul.dropna(subset = right_variables)
for i in range(len(right_variables)): 
    cumul = cumul[cumul[right_variables[i]] <= 200 ]
    
cumul = cumul.dropna(subset = left_variables)
for i in range(len(left_variables)): 
    cumul = cumul[cumul[left_variables[i]] <= 200 ]




# for i in range(len(equiv)): 
#     cumul[equiv[i][0]].fillna(cumul[equiv[i][1]], inplace=True)
#     del cumul[equiv[i][1]]
    





cumul = cumul[(cumul['AUQ054'] > 0) ]

# NEW LINE
# cumul = cumul[(cumul['AUQ054'] < 99) ]
# cumul = cumul.drop('AUQ054',axis=1)




cumul = cumul[(cumul['RIDAGEYR'] > 19) ]





# cumul.loc[cumul['AUQ060'] == 1, 'AUQ070'] = 1
# cumul.loc[cumul['AUQ070'] == 1, 'AUQ080'] = 1
# cumul.loc[cumul['AUQ080'] == 1, 'AUQ090'] = 1


    

# cumul.loc[cumul['AUQ331'] == 2, 'AUQ340'] = 0
# cumul.loc[cumul['AUQ331'] == 3, 'AUQ340'] = 0

# cumul.loc[cumul['AUQ331'] == 2, 'AUQ350'] = 2
# cumul.loc[cumul['AUQ331'] == 3, 'AUQ350'] = 2

# cumul.loc[cumul['AUQ300'] == 2, 'AUQ310'] = 0

# cumul.loc[cumul['AUQ350'] == 2, 'AUQ361'] = 0

# cumul.loc[cumul['DIQ010'] == 3, 'DIQ010'] = 2
# cumul.loc[cumul['DIQ010'] == 7, 'DIQ010'] = 2
# cumul.loc[cumul['DIQ010'] == 9, 'DIQ010'] = 2

cumul.loc[cumul['SMQ020'] == 7, 'SMQ020'] = 2
cumul.loc[cumul['SMQ020'] == 9, 'SMQ020'] = 2

# cumul.loc[cumul['MCQ160C'] == 7, 'MCQ160C'] = 2
# cumul.loc[cumul['MCQ160C'] == 9, 'MCQ160C'] = 2

# cumul.loc[cumul['MCQ160F'] == 7, 'MCQ160F'] = 2
# cumul.loc[cumul['MCQ160F'] == 9, 'MCQ160F'] = 2

# cumul.loc[cumul['MCQ220'] == 7, 'MCQ220'] = 2
# cumul.loc[cumul['MCQ220'] == 9, 'MCQ220'] = 2


# cumul.loc[cumul['MCQ160B'] == 7, 'MCQ160B'] = 2
# cumul.loc[cumul['MCQ160B'] == 9, 'MCQ160B'] = 2

# cumul.loc[cumul['MCQ160D'] == 7, 'MCQ160D'] = 2
# cumul.loc[cumul['MCQ160D'] == 9, 'MCQ160D'] = 2

# cumul.loc[cumul['MCQ160E'] == 7, 'MCQ160E'] = 2
# cumul.loc[cumul['MCQ160E'] == 9, 'MCQ160E'] = 2

# cumul.loc[cumul['ALQ101'] == 7, 'ALQ101'] = 2
# cumul.loc[cumul['ALQ101'] == 9, 'ALQ101'] = 2

# cumul.loc[cumul['BPQ020'] == 7, 'BPQ020'] = 2
# cumul.loc[cumul['BPQ020'] == 9, 'BPQ020'] = 2

# cumul.loc[cumul['AUQ300'] == 7, 'AUQ300'] = 2
# cumul.loc[cumul['AUQ300'] == 9, 'AUQ300'] = 2


nan_count = np.count_nonzero(np.isnan(cumul)) / (np.count_nonzero(np.isnan(cumul)) + np.count_nonzero(~np.isnan(cumul)))


#########
# cumul.loc[cumul['DMQMILIZ'] == 7, 'DMQMILIZ'] = np.NaN
# cumul.loc[cumul['DMQMILIZ'] == 9, 'DMQMILIZ'] = np.NaN
# cumul.loc[cumul['DBQ700'] == 9, 'DBQ700'] = np.NaN
# cumul.loc[cumul['AUQ054'] == 99, 'AUQ054'] = np.NaN
# cumul.loc[cumul['AUQ331'] == 9, 'AUQ331'] = np.NaN
# cumul.loc[cumul['AUQ340'] == 99, 'AUQ340'] = np.NaN
# cumul.loc[cumul['AUQ350'] == 9, 'AUQ350'] = np.NaN
# cumul.loc[cumul['AUQ361'] == 99, 'AUQ361'] = np.NaN
# cumul.loc[cumul['AUQ381'] == 99, 'AUQ381'] = np.NaN
# cumul.loc[cumul['AUQ310'] == 7, 'AUQ310'] = np.NaN
# cumul.loc[cumul['AUQ144'] == 9, 'AUQ144'] = np.NaN
# cumul.loc[cumul['AUQ370'] == 9, 'AUQ370'] = np.NaN
# cumul.loc[cumul['AUQ136'] == 9, 'AUQ136'] = np.NaN
# cumul.loc[cumul['AUQ136'] == 7, 'AUQ136'] = np.NaN
# cumul.loc[cumul['AUQ100'] == 9, 'AUQ100'] = np.NaN
# cumul.loc[cumul['AUQ138'] == 7, 'AUQ138'] = np.NaN
# cumul.loc[cumul['AUQ138'] == 9, 'AUQ138'] = np.NaN
# cumul.loc[cumul['AUQ310'] == 9, 'AUQ310'] = np.NaN
# cumul.loc[cumul['AUQ381'] == 77, 'AUQ381'] = np.NaN

# cumul.loc[cumul['AUQ110'] == 9, 'AUQ110'] = np.NaN
# cumul.loc[cumul['HIQ011'] == 7, 'HIQ011'] = np.NaN
# cumul.loc[cumul['HIQ011'] == 9, 'HIQ011'] = np.NaN




# cumul = cumul.drop(right_variables,axis=1)
# cumul = cumul.drop(left_variables,axis=1)
# cumul.to_csv('C:/Users/Tyler Gathman/documents/ui_data.csv')

# ###########

# cumul = cumul.dropna(subset = feature_variables)







cat_vars = [


# AUQ
# 'RIDRETH3',
'AUQ054',
# 'AUQ060',
# 'AUQ070',
# 'AUQ080',
# 'AUQ090',

# 'AUQ144',

# 'AUQ340',
# 'AUQ361', #'AUQ360',
# 'AUQ381',# 'AUQ380',
# 'AUQ310',
    
# 'AUQ110',
# 'AUQ100',

# 'DBQ700'              
                                  
                            
] 






for i in range(len(cat_vars)): 
    
    cumul = pd.concat([pd.get_dummies(cumul[cat_vars[i]], prefix=cat_vars[i], drop_first=True),cumul],axis=1)
    cumul.drop([cat_vars[i]],axis=1, inplace=True)




##############################



cumul['Right_PTA']= cumul[right_variables].astype(int).mean(axis=1)
cumul['Left_PTA']= cumul[left_variables].astype(int).mean(axis=1)

cumul['Better_PTA'] = cumul[["Right_PTA", "Left_PTA"]].min(axis=1)

cumul = cumul.drop('Right_PTA',axis=1)
cumul = cumul.drop('Left_PTA',axis=1)

# cumul = cumul.drop('AUQ350',axis=1)
# cumul = cumul.drop('AUQ331',axis=1)
# cumul = cumul.drop('AUQ300',axis=1)



#plt.hist(cumul['Better_PTA'], bins=[-10, 15, 25, 40, 55, 70,90])



#cumul = cumul.values
#np.random.shuffle(cumul)   

y = cumul['Better_PTA'].values
cumul = cumul.drop(right_variables,axis=1)
cumul = cumul.drop(left_variables,axis=1)

weights = cumul['WTMEC2YR'].values

cumul = cumul.drop('Better_PTA',axis=1)
cumul = cumul.drop('WTMEC2YR',axis=1)

# cumul = cumul.drop('AUQ110_5.0',axis=1)
# cumul = cumul.drop('AUQ110_9.0',axis=1)


sort_variables = cumul.keys().values


X = cumul.values



# X = cumul[:,0:-10]
# y = np.sum(cumul[:,-10:],1)

##############################



print('The Sample Size is',len(X))


imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')           
X = imputer.fit_transform(X)

# X = add_constant(X)


# vif_data = pd.DataFrame()
# vif_data["feature"] = sort_variables
# vif_data["VIF"] = [variance_inflation_factor(X, i)
#                   for i in range(len(sort_variables))]

count = 0
col = X.shape[1]

vari = sorted_vars

        






        
        
space ={'num_leaves': sp_randint(6, 200), 
             'min_child_samples': sp_randint(50, 200), 
             'min_child_weight': [1e-8, 1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
           'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
			 'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
              #'learning_rate':[ 0.01, 0.05, 0.1],
              
             }



count = 0

cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
# enumerate splits
outer_results = list()
outer_results1 = list()
outer_results2 = list()
outer_results3 = list()
outer_results4 = list()
outer_results5 = list()
outer_results6 = list()
predicted = list() 
actual = list() 
auc_15 = list()
auc_25 = list()
auc_41 = list()

sens_opt = list()
spec_opt = list()
age_1 = list() 
age_2 = list()
age_3 = list()

sensitivity1 = list()
specificity1 = list()
accuracy1 = list()
train_score = list()


for train_ix, test_ix in cv_outer.split(X):

    # split data
    X_train_unt, X_test_unt = X[train_ix, :], X[test_ix, :]
    y_train_unt, y_test_unt = y[train_ix], y[test_ix]
    weights_train, weights_test = weights[train_ix], weights[test_ix]
    
    ypt = PowerTransformer(method = 'yeo-johnson')
    ypt.fit(y_train_unt.reshape(-1,1))
    y_train = ypt.transform(y_train_unt.reshape(-1,1))
    y_train = y_train.ravel() 
    y_test = ypt.transform(y_test_unt.reshape(-1,1))
    y_test = y_test.ravel() 
    
    
    pt = PowerTransformer(method = 'yeo-johnson')
    pt.fit(X_train_unt)
    X_train = pt.transform(X_train_unt)
    X_test = pt.transform(X_test_unt)
    
    
    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
    # define the model
    model = LGBMRegressor()
    # define search space
    # define search
    search = RandomizedSearchCV(model, space, scoring='neg_mean_absolute_error', cv=cv_inner, refit=True,n_iter = 50)

    

    # execute search
    result = search.fit(X_train,y_train,sample_weight = weights_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    
    # Calculate inner loop performance 
    
    t_pred = best_model.predict(X_train)
    t_pred = ypt.inverse_transform(t_pred.reshape(-1,1))
    y_train = ypt.inverse_transform(y_train.reshape(-1,1))

    
    
    t_score = np.average(np.abs(t_pred.ravel() - y_train.ravel()), weights=weights_train, axis=0)
    train_score.append(t_score)
    # create shap plots 

    explainer = shap.TreeExplainer(model = best_model, 
                              data = None, 
                               model_ouput = 'raw',
                               feature_pertubation = 'tree_path_dependent')
    if count == 0: 
        shap_values = explainer.shap_values(X_test)
        X_values = X_test
    else: 
        shap_values = np.vstack((shap_values,explainer.shap_values(X_test)))
        X_values = np.vstack((X_values, X_test))
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
    yhat = ypt.inverse_transform(yhat.reshape(-1,1))
    y_test = ypt.inverse_transform(y_test.reshape(-1,1))
    
    # evaluate the model
    #acc = mean_absolute_error(y_test.ravel(), yhat.ravel())
    acc = np.average(np.abs(y_test.ravel() - yhat.ravel()), weights=weights_test, axis=0)

    # store the result
    outer_results.append(acc)
    
    weights1 = weights_test.copy()
    weights1[np.where((y_test.ravel() > 15))] = 0
    acc1 = np.average(np.abs(y_test.ravel() - yhat.ravel()), weights=weights1, axis=0)
    outer_results1.append(acc1)
    
    weights2 = weights_test.copy()
    weights2[np.where((y_test.ravel() < 16) | (y_test.ravel() > 25))] = 0
    acc2 = np.average(np.abs(y_test.ravel() - yhat.ravel()), weights=weights2, axis=0)
    outer_results2.append(acc2)
    
    weights3 = weights_test.copy()
    weights3[np.where((y_test.ravel() < 25) | (y_test.ravel() > 40))] = 0
    acc3 = np.average(np.abs(y_test.ravel() - yhat.ravel()), weights=weights3, axis=0)
    outer_results3.append(acc3)
    
    weights4 = weights_test.copy()
    weights4[np.where((y_test.ravel() < 41) | (y_test.ravel() > 55))] = 0
    acc4 = np.average(np.abs(y_test.ravel() - yhat.ravel()), weights=weights4, axis=0)
    outer_results4.append(acc4)
    
    weights5 = weights_test.copy()
    weights5[np.where((y_test.ravel() < 56))] = 0
    acc5 = np.average(np.abs(y_test.ravel() - yhat.ravel()), weights=weights5, axis=0)
    outer_results5.append(acc5)
    
    weights6 = weights_test.copy()
    weights6[np.where((y_test.ravel() < 16))] = 0
    acc6 = np.average(np.abs(y_test.ravel() - yhat.ravel()), weights=weights6, axis=0)
    outer_results6.append(acc6)
    
    predicted= np.append(yhat,predicted)
    actual= np.append(y_test,actual)
    
    
    keep = y_test.copy()

    thresh = 15
    keep[np.where((keep <thresh))] = 0
    keep[np.where((keep >=thresh))] = 1
        
    auc = metrics.roc_auc_score(keep,  yhat, sample_weight = weights_test)
    auc_15.append(auc)
    # age_1.append(metrics.roc_auc_score(keep, X_test_unt[:,6], sample_weight = weights_test))
    
    
    keep = y_test.copy()

    thresh = 25
    keep[np.where((keep <thresh))] = 0
    keep[np.where((keep >=thresh))] = 1
        
    auc = metrics.roc_auc_score(keep,  yhat, sample_weight = weights_test)
    auc_25.append(auc)
    # age_2.append(metrics.roc_auc_score(keep, X_test_unt[:,6], sample_weight = weights_test))

    keep = y_test.copy()

    
    thresh = 41
    keep[np.where((keep <thresh))] = 0
    keep[np.where((keep >=thresh))] = 1
        
    auc = metrics.roc_auc_score(keep,  yhat, sample_weight = weights_test)
    auc_41.append(auc)
    # age_3.append(metrics.roc_auc_score(keep, X_test_unt[:,6], sample_weight = weights_test))


    
    
    fpr, tpr, _ = metrics.roc_curve(keep,  yhat, sample_weight = weights_test)
    optimal_idx = np.argmax(tpr - fpr)
    sens_opt.append(tpr[optimal_idx])
    spec_opt.append(fpr[optimal_idx])



    # report progress
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
    count = count + 1
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
print('Accuracy: %.3f (%.3f)' % (mean(outer_results1), std(outer_results1)))
print('Accuracy: %.3f (%.3f)' % (mean(outer_results2), std(outer_results2)))
print('Accuracy: %.3f (%.3f)' % (mean(outer_results3), std(outer_results3)))
print('Accuracy: %.3f (%.3f)' % (mean(outer_results4), std(outer_results4)))
print('Accuracy: %.3f (%.3f)' % (mean(outer_results5), std(outer_results5)))

ab  = np.abs(shap_values) 
am = np.mean(ab,0)
sort = np.sort(am)



shap.summary_plot(shap_values, X_values,sort_variables,25)


# create ROC curve
# fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
# plt.plot(fpr,tpr,label="AUC="+str(auc))
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc=4)
# plt.show()

# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

# #create ROC curve
# plt.plot(fpr,tpr)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

