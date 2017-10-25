
# coding: utf-8

# In[112]:

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


# In[113]:

#set path to imprt and save files from and in
path = 'C:/Users/Yonathan/Desktop/Rafael'

#upload data
train = pd.read_csv(os.path.join(path,r'train.csv'),index_col='Unnamed: 0')
test = pd.read_csv(os.path.join(path,r'test.csv'),index_col='Unnamed: 0')


# In[115]:

#remove labels name from data
train=train.drop('targetName', 1)
col_names = list(train)
for name in col_names:
    if name[:4] == "Time":
        train=train.drop(name, 1)
col_names = list(test)
for name in col_names:
    if name[:4] == "Time":
        test=test.drop(name, 1)
#train.head()


# In[116]:

#train.shape
#test.head()
#train.head()


# In[117]:

#create a df vel_mag with the magnitude of the velocity and val_mean which average the velocity of the samples(row)
def vec_size(x,y,z):
    return (np.sqrt(z**2+x**2+y**2))
    
def vel(data):    
    vel_magn=pd.DataFrame(np.zeros(shape=(len(data),30)))
    for i, j in zip(range(4,179,6), range(0,31)):
        x = data.iloc[:,i]
        y = data.iloc[:,i+1]
        z = data.iloc[:,i+2]
    #velocity magnitude matrix    
        vel_magn.iloc[:,j] =vec_size(x,y,z)
    return (vel_magn)

vel_mag_train = vel(train)
#vel_mag_test = vel(test)
#mean velocity magnitudevector
vel_mean_train=np.mean(vel_mag_train, axis=1)
vel_mean_test=np.mean(vel_mag_test, axis=1)
#print (vel_mag_train)


# In[118]:

#a df to calculate the acceleration between each step
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


# In[119]:

def vec_size(x,y,z):
    return (np.sqrt(z**2+x**2+y**2))


# In[120]:

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
    
def angle(data):    
    angle_df=pd.DataFrame(np.zeros(shape=(28746,29)))
    for i, j in zip(range(6,182,6), range(0,28)):
        #print (train.iloc[:,i])
        angle_df.iloc[:,j] =calc_angle(i)
    return(angle_df)

angle_df_train=angle(train)
angle_df_test=angle(test)

angle_mean_train=np.mean(angle_df_train, axis=1)
angle_mean_test=np.mean(angle_df_test, axis=1)
#print (angle_df)


# In[121]:

#a df to calculate the acceleration between each step
def angle_che(data):
    angle_change_df=pd.DataFrame(np.zeros(shape=(28746,30)))
    for i in range(0,28):
        ang1=angle_df.iloc[:,i]
        ang2=angle_df.iloc[:,i+1]
        angle_change_df.iloc[:,i]=np.abs(ang2-ang1)
    return(angle_change_df)
#print (angle_change_df)

angle_change_df_train = angle_che(train)
angle_change_df_test = angle_che(test)


angle_change_mean_train=np.mean(angle_change_df_train, axis=1)
angle_change_mean_test=np.mean(angle_change_df_test, axis=1)
#print (angle_change_mean)


# In[122]:

def count_time(data):
    time_vec=[]
    for i in range(0,len(data)):
        sample=data.iloc[i,:]
        time_vec.append((29-sample.isnull().sum()/6)/2)
    return (time_vec)
time_epoch_train = pd.DataFrame(data=count_time(train))
time_epoch_test = pd.DataFrame(data=count_time(test)) 
#time_epoch


# In[125]:

new_train =train.copy(deep=True)
new_test =test.copy(deep=True)
def pos_off(data):
    col_names=list(data)
    for name in col_names:
        if name[:3] == "pos":
            data=data.drop(name, 1)
new_train = pos_off(new_train)
new_test = pos_off(new_test)          
#new_train


# In[139]:

df_add_vec_train=[new_train,vel_mag_train,
                  vel_mean,acc_df_train,
                  acc_mean,angle_df_train,
                  angle_mean,angle_change_df_train,
                  angle_change_mean_train,
                  time_epoch_train]

df_add_vec_test=[new_train,vel_mag_test,
                  vel_mean,acc_df_test,
                  acc_mean,angle_df_test,
                  angle_mean,angle_change_df_test,
                  angle_change_mean_test,
                  time_epoch_test]

final_train=pd.concat(df_add_vec_train)
final_test=pd.concat(df_add_vec_test)


# In[140]:

pd.options.display.max_columns=500
final_train.head()


# In[ ]:

class_column=final_train[.class]
final_train.drop('class',inplace=True)
final_train=pd.concat(final_train, class_column)

#final_train.head()


# In[111]:

print(final_train)


# In[ ]:




# In[100]:

final_train.columns=[i for i in range(1,124)]+['class']
final_test.columns=[i for i in range(1,125)]
#final_train.head()
#final_test.head()


# In[101]:

X = final_train.loc[:, final_train.columns != 'class']
Y = final_train.loc[:, final_train.columns == 'class']


# In[102]:

Y


# In[79]:

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
y_train,y_test=np.ravel(y_train),np.ravel(y_test)


# In[82]:

model = XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=0,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=5e-06,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
#eval_set = [(X_test, y_test)] 
eval_set = [(X_train, y_train), (X_test, y_test)] 


# In[90]:

model.fit(X_train, y_train, eval_metric="auc", eval_set=eval_set, early_stopping_rounds=9, verbose=True)


# In[ ]:

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[ ]:

AUC = auc(y_test, predictions)
print("AUC: %.2f%%" % (AUC * 100.0))


# In[ ]:

submission_results = model.predict_proba(test)[:,1]


# In[ ]:




# In[ ]:

sub.to_csv(os.path.join(path,r'Rafael_submission1.csv'),header=True,index=True, index_label='id')

