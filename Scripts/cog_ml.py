# Created by TJG on 1/2022 

# Utilizes LightGBM ML for predicting cognitive measures from hearing data in NHANES 

# Import the necessary modules and libraries
import pyreadstat
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score
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


# Define feature variables which may include both hearing and non-hearing measures depending on 
# the basis of the model	

   
		 
# Define path for NHANES with cognition assessment 
paths = ['c:\CSCI_5521/hw4/NHANES_12']

	# Define cognition assessment variable			   
label_variables = ['CFDCSR']	



feature_variables = [ 
'RIDAGEYR',	 'RIAGENDR', 
'RIDRETH3','DMDEDUC2', 'INDFMPIR',

'AUQ054','AUQ144','AUQ146','AUQ191', 
'DMDMARTL',

'MCQ160C',
'MCQ160F',
'MCQ220',
'MCQ160B',
'MCQ160D',
'MCQ160E',
'DIQ010',
'BPQ020',
'HIQ011',
'HSD010',
'BMXBMI',


'AUXU500R','AUXU1K1R','AUXU2KR','AUXU4KR','AUXU6KR','AUXU8KR',

'AUXU500L','AUXU1K1L','AUXU2KL','AUXU4KL','AUXU6KL','AUXU8KL', 


'DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090','DPQ100',
'WTMEC2YR'

					 ]

depr = ['DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090','DPQ100']

	




   
	
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
labl = feature_variables + label_variables 
cumul = cumul[labl]









#cumul = cumul[(cumul['RIDAGEYR'] > 20) ]

cumul['Cog'] = cumul[label_variables].sum(axis=1)
cumul = cumul.dropna(subset = label_variables)
cumul = cumul.drop(columns = label_variables)

cumul['PHQ-9'] = cumul[depr].sum(axis=1)
cumul = cumul.drop(columns = depr)

cumul = cumul.dropna(subset = ['AUXU2KR'])


# cumul['500H'] = cumul[["AUXU500R", "AUXU500L"]].max(axis=1)
# cumul['1000H'] = cumul[["AUXU1K1R", "AUXU1K1L"]].max(axis=1)
# cumul['2000H'] = cumul[["AUXU2KR", "AUXU2KL"]].max(axis=1)
# cumul['4000H'] = cumul[["AUXU4KR", "AUXU4KL"]].max(axis=1)
# cumul['6000H'] = cumul[["AUXU6KR", "AUXU6KL"]].max(axis=1)
# cumul['8000H'] = cumul[["AUXU8KR", "AUXU8KL"]].max(axis=1)



cumul['500L'] = cumul[["AUXU500R", "AUXU500L"]].min(axis=1)
cumul['1000L'] = cumul[["AUXU1K1R", "AUXU1K1L"]].min(axis=1)
cumul['2000L'] = cumul[["AUXU2KR", "AUXU2KL"]].min(axis=1)
cumul['4000L'] = cumul[["AUXU4KR", "AUXU4KL"]].min(axis=1)
cumul['6000L'] = cumul[["AUXU6KR", "AUXU6KL"]].min(axis=1)
cumul['8000L'] = cumul[["AUXU8KR", "AUXU8KL"]].min(axis=1)

# cumul['PTA_R'] = cumul[["AUXU500R", "AUXU1K1R",'AUXU2KR','AUXU4KR']].mean(axis=1)
# cumul['PTA_L'] = cumul[["AUXU500L", "AUXU1K1L",'AUXU2KL','AUXU4KL']].mean(axis=1)

# cumul['PTA_low'] = cumul[["PTA_R", "PTA_L"]].min(axis=1)

# cumul = cumul.drop(columns = ['PTA_R','PTA_L'])



audios = ['AUXU500R','AUXU1K1R','AUXU2KR','AUXU4KR','AUXU6KR','AUXU8KR',

'AUXU500L','AUXU1K1L','AUXU2KL','AUXU4KL','AUXU6KL','AUXU8KL']

cumul = cumul.drop(columns = audios)






weights = cumul['WTMEC2YR'].values




cumul['C'] = 1

cat_vars = [
  'RIAGENDR', 
  'RIDRETH3',
  #'DMDEDUC2',
  'DMDMARTL',
  'HIQ011',
  

#'AUQ054',
'AUQ144','AUQ146','AUQ191',

'MCQ160C',
'MCQ160F',
'MCQ220',
'MCQ160B',
'MCQ160D',
'MCQ160E',
'DIQ010',
'BPQ020'
						  
							
] 






for i in range(len(cat_vars)): 
	
	cumul = pd.concat([pd.get_dummies(cumul[cat_vars[i]], prefix=cat_vars[i], drop_first=True),cumul],axis=1)
	cumul.drop([cat_vars[i]],axis=1, inplace=True)





y = cumul.Cog.values 
X =cumul.drop(columns = ['Cog','1000L','4000L','6000L','WTMEC2YR']).values


sort_variables =cumul.drop(columns = ['Cog','1000L','4000L','6000L','WTMEC2YR']).keys().values




print('The Sample Size is',len(X))


			
imputer = KNNImputer(n_neighbors=1)
X = imputer.fit_transform(X)




vif_data = pd.DataFrame()
vif_data["feature"] = sort_variables
vif_data["VIF"] = [variance_inflation_factor(X, i)
				  for i in range(len(sort_variables))]



count = 0
col = X.shape[1]

vari = sorted_vars

		



# y,maxlog = stats.boxcox(y) 


pt = PowerTransformer(method = 'yeo-johnson')
pt.fit(X)
X = pt.transform(X)

y = y.reshape(-1,1)
pt2 = PowerTransformer(method = 'yeo-johnson')
pt2.fit(y)
y = pt2.transform(y)

y = y.ravel()

weights_train = 0
		
		
space ={'num_leaves': sp_randint(2, 200), 
			 'min_child_samples': sp_randint(5, 200), 
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
for train_ix, test_ix in cv_outer.split(X):

	# split data
	X_train, X_test = X[train_ix, :], X[test_ix, :]
	y_train, y_test = y[train_ix], y[test_ix]
	weights_train, weights_test = weights[train_ix], weights[test_ix]
	# configure the cross-validation procedure
	cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
	# define the model
	model = LGBMRegressor()
	# define search space
	# define search
	search = RandomizedSearchCV(model, space, scoring='r2', cv=cv_inner, refit=True,n_iter = 100)
	#early_stopping = EarlyStopping( monitor='val_loss', min_delta=0, patience=0, verbose=0,
	#mode='auto', baseline=None, restore_best_weights=False)
	
	# execute search
	result = search.fit(X_train, y_train)
	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_
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
	yhat = pt2.inverse_transform(yhat.reshape(-1,1)).ravel()
	y_test = pt2.inverse_transform(y_test.reshape(-1,1)).ravel()
	# evaluate the model
	acc = r2_score(y_test, yhat)
	# store the result
	outer_results.append(acc)
	# report progress
	print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
	count = count + 1
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))

ab	= np.abs(shap_values) 
am = np.mean(ab,0)
sort = np.sort(am)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(yhat, y_test)
# plt.xlim(0,100)
# plt.ylim(0,100)
#plt.scatter(y_1,y_valid)
shap.summary_plot(shap_values, X_values,sort_variables,15)

# calculate cummulative mean shap value for each response level 
# for loop over each KIQ variable 


# determine mean shap for pos / neg 