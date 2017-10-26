
# coding: utf-8

# In[141]:

#uploading different packages- to remove the one we do not need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut,KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Imputer
import os
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics 
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut,ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from random import seed
seed(123)
rcParams['figure.figsize'] = 12, 4


# In[46]:

#set path to imprt and save files from and in
path = 'C:/Users/Yonathan/Desktop/Rafael'

#upload data
train = pd.read_csv(os.path.join(path,r'train.csv'),index_col='Unnamed: 0')
test = pd.read_csv(os.path.join(path,r'test.csv'),index_col='Unnamed: 0')


# In[47]:

#remove labels names from data
train=train.drop('targetName', 1)
#remove unnecessary time cells from data
col_names = list(train)
for name in col_names:
    if name[:4] == "Time":
        train=train.drop(name, 1)
col_names = list(test)
for name in col_names:
    if name[:4] == "Time":
        test=test.drop(name, 1)
#train.head()


# In[49]:

# build a function vec_size which measures vector magnitude
def vec_size(x,y,z):
    return (np.sqrt(z**2+x**2+y**2))
#create a df vel_mag with the magnitude of the velocity and val_mean which average the velocity of the samples(row)   
def vel(data):    
    vel_magn=pd.DataFrame(np.zeros(shape=(len(data),30)))
    for i, j in zip(range(3,179,6), range(0,31)):
        x = data.iloc[:,i]
        y = data.iloc[:,i+1]
        z = data.iloc[:,i+2]
    #velocity magnitude matrix    
        vel_magn.iloc[:,j] =vec_size(x,y,z)
    return (vel_magn)

vel_mag_train = vel(train)
vel_mag_test = vel(test)
#mean velocity magnitudevector
vel_mean_train=np.mean(vel_mag_train, axis=1)
vel_mean_test=np.mean(vel_mag_test, axis=1)


# In[50]:

#a function to calculate the acceleration between each step
def acc(data, vel_res):   
    acc_df=pd.DataFrame(np.zeros(shape=(len(data),30)))
    for i in range(0,29):
        vel1=vel_res.iloc[:,i]
        vel2=vel_res.iloc[:,i+1]
        acc_df.iloc[:,i]=vel2-vel1
    return (acc_df) 

acc_df_train =acc(train, vel_mag_train)
acc_df_test =acc(test, vel_mag_test)
#mean acc 
acc_mean_train=np.mean(acc_df_train, axis=1)
acc_mean_test=np.mean(acc_df_test, axis=1)
#print (acc_df)


# In[52]:

#angle calculation
def calc_angle(n):
        x_prev = train.iloc[:,n]
        x_curr = train.iloc[:,n-6]
        y_prev = train.iloc[:,n+1]
        y_curr = train.iloc[:,n-5]
        z_prev = train.iloc[:,n+2]
        z_curr = train.iloc[:,n-4]
        curr_point_vec = [x_curr-x_prev,y_curr-y_prev,z_curr-z_prev]
        curr_point_vec_mag = vec_size(curr_point_vec[0],curr_point_vec[1],curr_point_vec[2])
        curr_point_vec_norm = [curr_point_vec[0]/curr_point_vec_mag,curr_point_vec[1]/curr_point_vec_mag,curr_point_vec[2]/curr_point_vec_mag]
        plain_vec =[x_curr-x_prev,y_curr-y_prev,0] 
        plain_vec_mag = vec_size(plain_vec[0],plain_vec[1],0)
        plain_vec_norm = [plain_vec[0]/plain_vec_mag,plain_vec[1]/plain_vec_mag,0]
        res = curr_point_vec_norm[0]*plain_vec_norm[0] +curr_point_vec_norm[1]*plain_vec_norm[1] +curr_point_vec_norm[2]* plain_vec_norm[2] 
        angle = np.arccos(res)
        return (angle*180.0/ np.pi)
    
#run it on a whole df
def angle(data):    
    angle_df=pd.DataFrame(np.zeros(shape=(len(data),29)))
    for i, j in zip(range(6,182,6), range(0,28)):
        #print (train.iloc[:,i])
        angle_df.iloc[:,j] =calc_angle(i)
    return(angle_df)

angle_df_train=angle(train)
angle_df_test=angle(test)

#calculate it mean
angle_mean_train=np.mean(angle_df_train, axis=1)
angle_mean_test=np.mean(angle_df_test, axis=1)
#print (angle_df)


# In[53]:

#a df to calculate the change in angles between each step
def angle_che(data,angle_df):
    angle_change_df=pd.DataFrame(np.zeros(shape=(len(data),30)))
    for i in range(0,28):
        ang1=angle_df.iloc[:,i]
        ang2=angle_df.iloc[:,i+1]
        angle_change_df.iloc[:,i]=np.abs(ang2-ang1)
        
    return(angle_change_df)
#print (angle_change_df)


angle_change_df_train = angle_che(train,angle_df_train)
angle_change_df_test = angle_che(test,angle_df_test)

#calculate the mean
angle_change_mean_train=np.mean(angle_change_df_train, axis=1)
angle_change_mean_test=np.mean(angle_change_df_test, axis=1)
#print (angle_change_mean)


# In[54]:

#count how many time steps each samples has (as non NaN)
def count_time(data):
    time_vec=[]
    for i in range(0,len(data)):
        sample=data.iloc[i,:]
        time_vec.append((29-sample.isnull().sum()/6)/2)
    return (time_vec)
time_epoch_train = pd.DataFrame(data=count_time(train))
time_epoch_test = pd.DataFrame(data=count_time(test)) 
#time_epoch


# In[55]:

#create new variable to store the data
new_train =train.copy(deep=True)
new_test =test.copy(deep=True)      


# In[56]:

#remove Position columns from data
col_names=list(new_train)
for name in col_names:
    if str(name)[:1] == "p":
        new_train=new_train.drop(name, 1)

col_names=list(new_test)
for name in col_names:
    if  str(name)[:1] == "p":
        new_test=new_test.drop(name, 1)  


# In[58]:

#create a vector of all the matrices and array which we which to add to the data

df_add_vec_train=[new_train,vel_mag_train,
                  vel_mean_train,acc_df_train,
                  acc_mean_train,angle_df_train,
                  angle_mean_train,angle_change_df_train,
                  angle_change_mean_train,
                  time_epoch_train]

df_add_vec_test=[new_test,vel_mag_test,
                  vel_mean_test,acc_df_test,
                  acc_mean_test,angle_df_test,
                  angle_mean_test,angle_change_df_test,
                  angle_change_mean_test,
                  time_epoch_test]

#merge them to the data
final_train = pd.concat(df_add_vec_train, axis=1)
final_test = pd.concat(df_add_vec_test, axis=1)


# In[62]:

#rearenge the column name to enable modeling
final_train.columns=list(final_train)[:91]+[i for i in range(1,125)]
final_test.columns=list(final_train)[:90]+[i for i in range(1,125)]


# In[74]:

#keep the clean data in CSV files
final_train.to_csv(os.path.join(path,r'final_train_Rafael.csv'),header=True,index=True)
final_test.to_csv(os.path.join(path,r'final_test_Rafael.csv'),header=True,index=True)


# In[67]:

#seperatre training data into features and labels
Y =pd.DataFrame(final_train['class'])
X =final_train.drop('class', axis=1)


# In[70]:

#split data to training and testinf sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
y_train,y_test=np.ravel(y_train),np.ravel(y_test)


# In[ ]:

"""
# in case needed it is posible to add an intex to the data 
ind=list(range(0,28746))
final_train=final_train.assign(Index = ind)
#final_train
"""


# In[139]:

# a very simple XGboost model
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 26
param['min_child_weight']=1
param['gamma']=0
param['subsample']=1
param['scale_pos_weight']=1
param['colsample_bytree']=1
param['learning_rate'] =0.1
param['n_estimators']=1000
param['seed']=123

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 1000
bst = xgb.train(param, xg_train, num_round, watchlist )
# get prediction
pred = np.int_(bst.predict( xg_test ))


# In[140]:

#calculation
f1 = f1_score(y_test, pred,average='macro')
print("F1: %.2f%%" % (f1 * 100.0))


# In[ ]:

#A simple Gradient boosting model
train_gbdt=final_train.replace([np.inf],[np.nan])
train_gbdt.fillna(0,inplace=True)
y_gbdt=pd.DataFrame(train_gbdt['class'])
X_gbdt=train_gbdt.drop('class', axis=1)

ss = ShuffleSplit(n_splits=5,random_state=123345,test_size=0.2)
for train_index, test_index in ss.split(X_gbdt,np.ravel(y_gbdt)):
    X_train , X_test = X_gbdt.loc[train_index,:] , X_gbdt.loc[test_index,:]
    y_train , y_test = y_gbdt.loc[train_index] , y_gbdt.loc[test_index]

gbdt = GradientBoostingClassifier(max_depth=5,subsample=0.8,n_estimators=30)
gbdt.fit(X_train,y_train)
pred = gbdt.predict(X_test)
print (classification_report(pred,y_test))


# In[120]:

#pred


# In[121]:

#y_test


# In[ ]:

submission_results = model.predict_proba(test)[:,1]


# In[ ]:

sub.to_csv(os.path.join(path,r'Rafael_submission1.csv'),header=True,index=True, index_label='id')

