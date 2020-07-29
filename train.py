import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from copy import deepcopy
from datetime import datetime

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.normalization import BatchNormalization

#import data
print("Reading train.csv.")
trainDF = pd.read_csv("train.csv")

#clean up wrong X and Y values
xy_scaler = preprocessing.StandardScaler()
xy_scaler.fit(trainDF[["X","Y"]])
trainDF[["X","Y"]] = xy_scaler.transform(trainDF[["X","Y"]])
trainDF = trainDF[abs(trainDF["X"]) < 5]
trainDF = trainDF[abs(trainDF["Y"]) < 5]
trainDF.index = range(len(trainDF))

#plt.plot(trainDF["X"],trainDF["Y"],'.')
#plt.show()

def parse_time(x):
	DD = datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
	time = DD.hour
	day = DD.day
	month = DD.month
	year = DD.year
	return time,day,month,year

def get_season(x):
	summer = 0
	fall = 0
	winter = 0
	spring = 0
	if (x in [5,6,7]):
		summer = 1
	if (x in [8,9,10]):
		fall = 1
	if (x in [11,0,1]):
		winter = 1
	if (x in [2,3,4]):
		spring = 1
	return summer,fall,winter,spring

def parse_data(df,logodds,logoddsPA):
	feature_list = df.columns.tolist()
	if "Descript" in feature_list:
		feature_list.remove("Descript")
	if "Resolution" in feature_list:
		feature_list.remove("Resolution")
	if "Category" in feature_list:
		feature_list.remove("Category")
	if "Id" in feature_list:
		feature_list.remove("Id")
	newData = df[feature_list]
	newData.index = range(len(df))
	print("Creating address features.")
	address_features = newData["Address"].apply(lambda x: logodds[x])
	address_features.columns = ["LogOdds" + str(x) for x in range(len(address_features.columns))]
	print("Parsing dates.")
	newData["Time"],newData["Day"],newData["Month"],newData["Year"] = zip(*newData["Dates"].apply(parse_time))
	days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
	print("Creating one-hot variables.")
	dummy_ranks_PD = pd.get_dummies(newData["PdDistrict"],prefix = 'PD')
	dummy_ranks_DAY = pd.get_dummies(newData["DayOfWeek"],prefix = 'DAY')
	newData["IsIntersection"] = newData["Address"].apply(lambda x: 1 if "/" in x else 0)
	newData["LogOddsPA"] = newData["Address"].apply(lambda x: logoddsPA[x])
	print("Droping processed columns.")
	newData = newData.drop("PdDistrict",axis = 1)
	newData = newData.drop("DayOfWeek",axis = 1)
	newData = newData.drop("Address",axis = 1)
	newData = newData.drop("Dates",axis = 1)
	feature_list = newData.columns.tolist()
	print("Joining one-hot features.")
	features = newData[feature_list].join(dummy_ranks_PD.ix[:,:]).join(dummy_ranks_DAY.ix[:,:]).join(address_features.ix[:,:])
	print("Creating new features.")
	features["IsDup"] = pd.Series(features.duplicated() | features.duplicated(take_last = True)).apply(int)
	features["Awake"] = features["Time"].apply(lambda x: 1 if (x == 0 or (x >= 8 and x <= 23)) else 0)
	features["Summer"],features["Fall"],features["Winter"],features["Spring"] = zip(*features["Month"].apply(get_season))
	if "Category" in df.columns:
		labels = df["Category"].astype('category')
	else:
		labels = None
	return features,labels

addresses = sorted(trainDF["Address"].unique())
categories = sorted(trainDF["Category"].unique())
A_CNT = trainDF.groupby(["Address"]).size()
C_CNT = trainDF.groupby(["Category"]).size()
A_C_CNT = trainDF.groupby(["Address","Category"]).size()
logodds = {}
logoddsPA = {}
MIN_CAT_CNT = 2
default_logodds = np.log(C_CNT / float(len(trainDF))) - np.log(1.0 - C_CNT / float(len(trainDF)))
for add in addresses:
	PA = A_CNT[add] / float(len(trainDF))
	logodds[add] = deepcopy(default_logodds)
	logoddsPA[add] = np.log(PA) - np.log(1.0 - PA)
	for cat in A_C_CNT[add].keys():
		if A_C_CNT[add][cat] > MIN_CAT_CNT and A_C_CNT[add][cat] < A_CNT[add]:
			PA = A_C_CNT[add][cat] / float(A_CNT[add])
			logodds[add][categories.index(cat)] = np.log(PA) - np.log(1.0 - PA)
	logodds[add] = pd.Series(logodds[add])
	logodds[add].index = range(len(categories))
features,labels = parse_data(trainDF,logodds,logoddsPA)
#print(features.columns.tolist())

collist = features.columns.tolist()
scaler = preprocessing.StandardScaler()
scaler.fit(features)
features[collist] = scaler.transform(features)

new_PCA = PCA(n_components = 60)
new_PCA.fit(features)
#print(new_PCA.explained_variance_ratio_)

#plt.plot(new_PCA.explained_variance_ratio_)
#plt.yscale('log')
#plt.title("PCA explained ratio of features")
#plt.show()

#plt.plot(new_PCA.explained_variance_ratio_.cumsum())
#plt.title("Cumsum of PCA explained ratio")
#plt.show()

sss = StratifiedShuffleSplit(labels,train_size = 0.5)
for train_index,test_index in sss:
	features_train,features_test = features.iloc[train_index],features.iloc[test_index]
	labels_train,labels_test = labels[train_index],labels[test_index]
features_train.index = range(len(features_train))
features_test.index = range(len(features_test))
labels_train.index = range(len(labels_train))
labels_test.index = range(len(labels_test))
features.index = range(len(features))
labels.index = range(len(labels))

def build_and_fit_model(X_train,y_train,X_test = None,y_test = None,hn = 32,dp = 0.5,layers = 1,epochs = 1,batches = 64,verbose = 0):
	input_dim = X_train.shape[1]
	output_dim = len(labels.unique())
	Y_train = np_utils.to_categorical(y_train.cat.rename_categories(range(len(y_train.unique()))))
	
	model = Sequential()
	model.add(Dense(hn,input_shape = (input_dim,)))
	model.add(PReLU())
	model.add(Dropout(dp))

	'''
	model.add(Dense(hn)
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dp)
	'''

	for i in range(layers):
		model.add(Dense(array_hn[i]))
		model.add(PReLU())
		model.add(BatchNormalization())
		model.add(Dropout(array_dp[i]))

	model.add(Dense(output_dim))
	model.add(Activation('softmax'))
	model.compile(loss = 'categorical_crossentropy',optimizer = 'adam')

	if X_test is not None:
		Y_test = np_utils.to_categorical(y_test.cat.rename_categories(range(len(y_test.unique()))))
		fitting = model.fit(X_train,Y_train,nb_epoch = epochs,batch_size = batches,verbose = verbose,validation_data = (X_test,Y_test))
		test_score = log_loss(y_test,model.predict_proba(X_test,verbose = 0))
	else:
		model.fit(X_train,Y_train,nb_epoch = epochs,batch_size = batches,verbose = verbose)
		fitting = 0
		test_score = 0
	return test_score,fitting,model

N_EPOCHS = 50 #80
N_HN = 500
N_LAYERS = 4 #1
DP = 0.5

array_hn = [666,333,222,111]
array_dp = [0.5,0.3,0.2,0.1]

print("Using neural network.")
score,fitting,model = build_and_fit_model(features.as_matrix(),labels,hn = N_HN,layers = N_LAYERS,epochs = N_EPOCHS,verbose = 2,dp = DP)
#score,fitting,model = build_and_fit_model(features_train.as_matrix(),labels_train,X_test = features_test.as_matrix(),y_test = labels_test,hn = N_HN,layers = N_LAYERS,epochs = N_EPOCHS,verbose = 2,dp = DP)

#print("Using logistic regression.")
#model = LogisticRegression()

#print("Using random forest classifier.")
#model = RandomForestClassifier(n_estimators = 100,min_samples_split = 4,verbose = True,n_jobs = 4,max_features = None)

#print("Using gradient boosting classifier.")
#model = GradientBoostingClassifier(n_estimators = 100,max_depth = 3,verbose = 0)

#model.fit(features,labels)

#print("all",log_loss(labels,model.predict_proba(features.as_matrix())))
#print("train",log_loss(labels_train,model.predict_proba(features_train.as_matrix())))
#print("test",log_loss(labels_test,model.predict_proba(features_test.as_matrix())))

print("Reading test.csv.")
testDF = pd.read_csv("test.csv")
testDF[["X","Y"]] = xy_scaler.transform(testDF[["X","Y"]])
testDF["X"] = testDF["X"].apply(lambda x: 0 if abs(x) > 5 else x)
testDF["Y"] = testDF["Y"].apply(lambda y: 0 if abs(y) > 5 else y)

new_addresses = sorted(testDF["Address"].unique())
new_A_CNT = testDF.groupby("Address").size()
only_new = set(new_addresses + addresses) - set(addresses)
only_old = set(new_addresses + addresses) - set(new_addresses)
in_both = set(new_addresses).intersection(addresses)
for add in only_new:
	PA = new_A_CNT[add] / float(len(testDF) + len(trainDF))
	logoddsPA[add] = np.log(PA) - np.log(1.0 - PA)
	logodds[add] = deepcopy(default_logodds)
	logodds[add].index = range(len(categories))
for add in in_both:
	PA = (A_CNT[add] + new_A_CNT[add]) / float(len(testDF) + len(trainDF))
	logoddsPA[add] = np.log(PA) - np.log(1.0 - PA)

features_sub,_ = parse_data(testDF,logodds,logoddsPA)
collist = features_sub.columns.tolist()
features_sub[collist] = scaler.transform(features_sub[collist])

predDF = pd.DataFrame(model.predict_proba(features_sub.as_matrix()),columns = sorted(labels.unique()))
predDF.head()
#predDF.to_csv("mySubmission_NeuNet_Layer_1.csv",index_label = "Id")
predDF.to_csv("mySubmission_NeuNet_Layer_4.csv",index_label = "Id")
#predDF.to_csv("mySubmission_LogReg.csv",index_label = "Id")
#predDF.to_csv("mySubmission_RanFor.csv",index_label = "Id")
#predDF.to_csv("mySubmission_GraBoo.csv",index_label = "Id")