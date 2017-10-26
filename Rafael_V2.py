
# coding: utf-8

# In[4]:

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
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from random import seed
seed(123)
rcParams['figure.figsize'] = 12, 4
import json


# In[2]:

"""
#set path to imprt and save files from and in
with open('PATHS.json') as json_data:
    paths = json.load(json_data)

#upload data
train = pd.read_csv(paths["TRAIN_PATH"],index_col='Unnamed: 0')
test = pd.read_csv(paths["TEST_PATH"],index_col='Unnamed: 0')
"""


# In[3]:

#set path to imprt and save files from and in
path = 'C:/Users/Yonathan/Desktop/Rafael'

#upload data- no need if you have the clean files
train = pd.read_csv(os.path.join(path,r'train.csv'),index_col='Unnamed: 0')
test = pd.read_csv(os.path.join(path,r'test.csv'),index_col='Unnamed: 0')
#upload data- no need if you have the clean files, and skip to cell # 93 (seperate training and testing data)
final_train = pd.read_csv(os.path.join(path,r'final_train_Rafael.csv'),index_col='df')
final_test = pd.read_csv(os.path.join(path,r'final_test_Rafael.csv'), index_col='df)


# In[5]:

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


# In[6]:

# build a function vec_size which measures vector magnitude
def vec_size(x,y,z):
    return (np.sqrt(z**2+x**2+y**2))
#create a df vel_mag with the magnitude of the velocity and val_mean which average the velocity of the samples(row)   
def vel(data):    
    vel_magn=pd.DataFrame()
    for i in range(30):
        x = data["velX_"+str(i)]
        y = data["velY_"+str(i)]
        z = data["velZ_"+str(i)]
    #velocity magnitude matrix    
        vel_magn["vel_magnitude_"+str(i)] =vec_size(x,y,z)
    return (vel_magn)

vel_mag_train = vel(train)
vel_mag_test = vel(test)
#mean velocity magnitudevector
vel_mean_train = pd.DataFrame(np.mean(vel_mag_train, axis=1),columns=["vel_mean"])
vel_mean_test= pd.DataFrame(np.mean(vel_mag_test, axis=1),columns=["vel_mean"])


# In[7]:

#a function to calculate the acceleration between each step
def acc(data, vel_res):   
    acc_df=pd.DataFrame()
    for i in range(0,29):
        vel1=vel_res["vel_magnitude_"+str(i)]
        vel2=vel_res["vel_magnitude_"+str(i+1)]
        acc_df["accel_"+str(i)]=vel2-vel1
    return (acc_df) 

acc_df_train =acc(train, vel_mag_train)
acc_df_test =acc(test, vel_mag_test)
#mean acc 
acc_mean_train=pd.DataFrame(np.mean(acc_df_train, axis=1),columns=["acc_mean"])
acc_mean_test=pd.DataFrame(np.mean(acc_df_test, axis=1),columns=["acc_mean"])
#print (acc_df)


# In[10]:

#angle calculation
def calc_angle(data,pos_index): 
        x_prev = data["posX_"+str(pos_index)] 
        x_curr = data["posX_"+str(pos_index+1)]
        
        y_prev = data["posY_"+str(pos_index)] 
        y_curr = data["posY_"+str(pos_index+1)]
        
        z_prev = data["posZ_"+str(pos_index)] 
        z_curr = data["posZ_"+str(pos_index+1)]
        
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
def angle(data, num_of_func = 1):    
    angle_df=pd.DataFrame()
    for i in range(29): #TODO: Why range only until 29?
        #print (train.iloc[:,i])
        if num_of_func==1:
            angle_df["angle_"+str(i)] =calc_angle(data,i)
        else:
            angle_df["angle_"+str(i)] =calc_angle(data,i)
    return angle_df


# In[11]:

angle_df_train=angle(train,1)
angle_df_test=angle(test,1)

angle_df_train2 =angle(train,2)
angle_df_test2 =angle(test,2)

#calculate the mean
angle_mean_train = pd.DataFrame(np.mean(angle_df_train, axis=1),columns=["angle_mean"])
angle_mean_test = pd.DataFrame(np.mean(angle_df_test, axis=1),columns=["angle_mean"])
#print (angle_df)


# In[12]:

#a df to calculate the change in angles between each step
def angle_che(data,angle_df):
    angle_change_df=pd.DataFrame()
    for i in range(0,28):
        ang1=angle_df["angle_"+str(i)]
        ang2=angle_df["angle_"+str(i+1)]
        angle_change_df["angle_change_"+str(i)]=np.abs(ang2-ang1)
        
    return(angle_change_df)
#print (angle_change_df)


angle_change_df_train = angle_che(train,angle_df_train)
angle_change_df_test = angle_che(test,angle_df_test)

#calculate the mean
angle_change_mean_train = pd.DataFrame(np.mean(angle_change_df_train, axis=1),columns=["angle_change_mean"])
angle_change_mean_test = pd.DataFrame(np.mean(angle_change_df_test, axis=1),columns=["angle_change_mean"])
#print (angle_change_mean)


# In[13]:

#count how many time steps each samples has (as non NaN)
def count_time(data):
    res = pd.DataFrame()
    time_vec=[]
    for i in range(0,len(data)):
        sample=data.iloc[i,:]
        time_vec.append((29-sample.isnull().sum()/6)/2)
    res["time_count"] = time_vec
    return res
time_epoch_train = count_time(train)
time_epoch_test = count_time(test) 
#time_epoch


# In[14]:

def create_go_up_and_go_down(table):
    up_and_down = pd.DataFrame()
    goes_up = np.zeros(len(table))
    goes_down = np.zeros(len(table))
    for i in range(len(table)):
        if(does_go_up(table.loc[i])):
            goes_up[i] = 1
        if(does_go_down(table.loc[i])):
            goes_down[i] = 1

    up_and_down["goes_up"] = goes_up
    up_and_down["goes_down"] = goes_down
    return up_and_down


def does_go_up(row):
    for i in range(30):
        if(row["velZ_"+str(i)] > 0):
            return True
    return False

def does_go_down(row):
    for i in range(30):
        if(row["velZ_"+str(i)] < 0):
            return True
    return False


#get parabula parameters
def calc_S(row):
    S = []
    for i in range(30):
        if(np.isnan(row["posX_"+str(i)])):
            break
        S.append(((row["posX_"+str(i)])**2 + (row["posY_"+str(i)])**2)**0.5)
    return S

def calc_parabola_params(table):
    res = pd.DataFrame()
    params = np.array([[0 for i in range(len(table))] for j in range(3)])
    for i in range(len(table)):
        row = table.loc[i]
        S = calc_S(row)
        current_params = np.polyfit(S, [row["posZ_"+str(i)] for i in range(len(S))],2)
        cur_f = np.multiply(current_params[0],(np.power(S,2))) + np.multiply(S,current_params[1]) + current_params[2]
        '''
        if(i%1000 == 0):
            plt.plot(S, [row["posZ_"+str(i)] for i in range(len(S))])
            plt.plot(S, cur_f)
            plt.savefig("images/yair_" + str(i) + ".png")
            plt.clf()
            print("iteration " + str(i))
        '''
        params[0][i], params[1][i], params[2][i] = current_params[0], current_params[1], current_params[2]
        res["parabola_parameter_a"] = params[0]
        res["parabola_parameter_b"] = params[1]
        res["parabola_parameter_c"] = params[2]
    return res


# In[41]:

#get goes_up and goes_down features:
up_and_down_train = create_go_up_and_go_down(train)
up_and_down_test = create_go_up_and_go_down(test)


# In[77]:

up_and_down_test = create_go_up_and_go_down(test)


# In[42]:

#get parbola parameters for each features
parabola_params_train = calc_parabola_params(train)
parabola_params_test = calc_parabola_params(test)


# In[17]:

#create new variable to store the data
new_train =train.copy(deep=True)
new_test =test.copy(deep=True)      


# In[18]:

#remove Position columns from data
col_names=list(new_train)
for name in col_names:
    if str(name)[:3] == "pos":
        new_train=new_train.drop(name, 1)

col_names=list(new_test)
for name in col_names:
    if  str(name)[:3] == "pos":
        new_test=new_test.drop(name, 1)  


# In[57]:

pca_train = pd.read_csv(os.path.join(path,r'PCA2dim.csv'),header=None)
pca_test = pd.read_csv(os.path.join(path,r'PCA2dimUnlabeled.csv'),header=None)


# In[74]:

pca_train = pca_train.iloc[:,0:2]
pca_test = pca_test.iloc[:,0:2]


# In[91]:

#create a vector of all the matrices and array which we which to add to the data

df_add_vec_train=[new_train,vel_mag_train,
                  vel_mean_train,acc_df_train,
                  acc_mean_train,angle_df_train,
                  angle_mean_train,angle_change_df_train,
                  angle_change_mean_train,
                  time_epoch_train, up_and_down_train, 
                  parabola_params_train,pca_train]

df_add_vec_test=[new_test,vel_mag_test,
                  vel_mean_test,acc_df_test,
                  acc_mean_test,angle_df_test,
                  angle_mean_test,angle_change_df_test,
                  angle_change_mean_test,
                  time_epoch_test, up_and_down_test, 
                  parabola_params_test,pca_test]

#merge them to the data
final_train = pd.concat(df_add_vec_train, axis=1)
final_test = pd.concat(df_add_vec_test, axis=1)


# In[40]:

pd.set_option('display.max_columns', 500)
#final_test


# In[83]:

#keep the clean data in CSV files
final_train.to_csv(os.path.join(path,r'final_train_Rafael.csv'), na_rep='NaN',header=True,index=True)
final_test.to_csv(os.path.join(path,r'final_test_Rafael.csv'), na_rep='NaN',header=True,index=True)


# In[93]:

#seperatre training data into features and labels
Y =pd.DataFrame(final_train['class'])
X =final_train.drop('class', axis=1)


# In[94]:

#split data to training and testinf sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
y_train,y_test=np.ravel(y_train),np.ravel(y_test)


# In[98]:

#or
ss = ShuffleSplit(n_splits=5,random_state=123345,test_size=0.2)
for train_index, test_index in ss.split(X_gbdt,np.ravel(y_gbdt)): 
    X_train , X_test = X.loc[train_index,:] , X.loc[test_index,:]
    y_train , y_test = Y.loc[train_index] , Y.loc[test_index]


# In[24]:

"""
only for calibration


xgb_model = xgb.XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have 
#much fun of fighting against overfit 
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['multi:softmax'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [10], #number of trees, change it to 1000 for better results
              'seed': [1337]}


clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(y_train,n_folds=5, shuffle=True), 
                   scoring='f1_macro',
                   verbose=10, refit=True)

clf.fit(X_train, y_train)

#trust your CV!
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('F1 score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
    
"""


# In[101]:

# a very simple XGboost model
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
#param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 26
param['min_child_weight']=11
param['gamma']=0
param['subsample']=0.8
param['scale_pos_weight']=1
param['colsample_bytree']=0.7
param['learning_rate'] =0.05
param['n_estimators']=10 #number of trees, change it to 1000 for better results
param['seed']=123

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 150 #to increase to 400
bst = xgb.train(param, xg_train, num_round, watchlist )
# get prediction
pred = np.int_(bst.predict( xg_test ))


# In[102]:

#calculation
f1 = f1_score(y_test, pred,average='macro')
print("F1: %.2f%%" % (f1 * 100.0))


# In[ ]:

""" 
not in use

#A simple Gradient boosting model
train_gbdt=final_train.replace([np.inf],[np.nan])
train_gbdt.fillna(0,inplace=True)
y_gbdt=np.ravel(pd.DataFrame(train_gbdt['class']))
X_gbdt=train_gbdt.drop('class', axis=1)

ss = ShuffleSplit(n_splits=5,random_state=123345,test_size=0.2)
for train_index, test_index in ss.split(X_gbdt,np.ravel(y_gbdt)): 
    X_train , X_test = X_gbdt.loc[train_index,:] , X_gbdt.loc[test_index,:]
    y_train , y_test = y_gbdt.loc[train_index] , y_gbdt.loc[test_index]

gbdt = GradientBoostingClassifier(max_depth=5,subsample=0.8,n_estimators=30)
gbdt.fit(X_train,y_train)
pred = gbdt.predict(X_test)
print (classification_report(pred,y_test))
"""


# In[ ]:

#pred


# In[ ]:

#y_test


# In[103]:

#anomaly detection
# use the same dataset
tr_data = X
tr_data.fillna(0,inplace=True)


# In[ ]:

clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1, verbose=True)
clf.fit(tr_data)


# In[82]:

final_test.fillna(0,inplace=True)
pred = clf.predict(final_test)

# inliers are labeled 1, outliers are labeled -1
abnormal = tr_data[pred == -1]


# In[ ]:

submission_results = bst.predict(final_test)[:,1]


# In[ ]:

submission_results[abnormal]=26


# In[ ]:

submission_results.to_csv(os.path.join(path,r'Rafael_submission1.csv'),header=True,index=True, index_label='id')

