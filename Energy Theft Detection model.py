import numpy as np 
import glob as glob
import matplotlib.pyplot as mtp  
import pandas as pd  

import sklearn
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering as ac


from datetime import datetime, date

daily_data = pd.read_csv('D:/explo/daily_dataset_tog/daily_dataset.csv')
daily_data.reset_index(inplace=True)

daily_data =daily_data.pivot(index='day', columns='LCLid', values='energy_sum')

daily_data.reset_index(inplace=True)

daily_data['day']= pd.to_datetime(daily_data['day'])

daily_data['day']= (daily_data['day'].dt.strftime('%Y/%m'))

grouped =daily_data.groupby('day').mean()

result= grouped.transpose()

interpolated = result.interpolate(method='linear', limit_direction='both')
res= interpolated.dropna(axis=0, how='any')

result= grouped.transpose()

cluster=ac(n_clusters=3, affinity='euclidean', linkage='ward')

arr = pd.DataFrame(cluster.fit_predict(res))


result.reset_index(inplace=True)
mer =result.merge(arr, left_index=True, right_index=True)

mer = mer.set_index('LCLid')


# cl are clusters 

cl1= mer[mer[0]==0]
cl1= cl1.drop([0], axis=1)

cl2= mer[mer[0]==1]
cl2= cl2.drop([0], axis=1)

cl3= mer[mer[0]==2]
cl3= cl3.drop([0], axis=1)

# plot initial 20 ids of each cluster to see the trend of energy consumption
print(" combined plot for 20 ids")
for i in range(20):
    mtp.plot(cl3.iloc[i],'r')
    mtp.plot(cl2.iloc[i],'g')
    mtp.plot(cl1.iloc[i],'b')

# now we need to make theft customers data by applying some changes in original data
# we took cluster 3 which falls in between having suitable number of IDs


thefta = cl3.sample(n=700)
theftb = cl1.sample(n=3300)

theft= pd.concat([thefta,theftb])
    # took 4000 random ids to make changes to their data

index_names= theft.index  # renaming their index name so that we do not have same name as theft and honest
for q in range(4000):
    theft.rename(index={index_names[q]:'theft'+str(q)}, inplace=True)

import random
# divided into 4 types of theft
theft1= theft[:1000]
theft2= theft[1000:2300]
theft3= theft[2300:3500]
theft4= theft[3500:4000]

# consumption of energy decreased by same factor for all months
for i in range(len(theft1)):
    val=random.uniform(0.1,0.8)
    for k, va in theft1.iteritems():
        theft1[k][i]= theft1[k][i]*val
    
theft1['label'] = 1  # labeled as theft

# decresd energy consumotion with random factor
for i in range(len(theft2)):
    for k, va in theft2.iteritems():
        val=random.uniform(0.1,0.8)
        theft2[k][i]= theft2[k][i]*val

theft2['label'] = 1

# energy consumed is always less than mean  by random factor
for i in range(len(theft3)):
    mea=np.mean(theft3.iloc[i])
    for k, va in theft3.iteritems():
        val=random.uniform(0.1,0.8)
        theft3[k][i]= mea*val
  
theft3['label'] = 1

# constant consumption of energy  
for i in range(len(theft4)):
    mea=np.mean(theft4.iloc[i])
    for k, va in theft4.iteritems():
        theft4[k][i]= mea

theft4['label'] = 1

# we can visualize the energy consumption trend by using this plot
for i in range(100):
    mtp.plot(theft1.iloc[i,:-1],'r')
    mtp.plot(theft2.iloc[i,:-1],'g')
    mtp.plot(theft3.iloc[i,:-1],'b')
    mtp.plot(theft4.iloc[i,:-1],'y')
mtp.show()


# initial data except the cluster having least no. of IDs because it almost contain all those IDswhose data was not available to us

data=pd.concat([cl1,cl3])

data['label']= 0  #labeled all as honest

# now making model with final data habving theft as well as honest customers
final= pd.concat([data,theft1,theft2,theft3,theft4])

final.fillna(value= -999, inplace= True)  # nan values fileed by -999 so that out random forest almost ignore them

X = final.iloc[:, 0:28].values
y = final.iloc[:, 28].values   # label column

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# to scale the data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

for i in range(len(y_pred)):
    if y_pred[i]>=0.6:
        y_pred[i]=1
    elif y_pred[i]<0.6:
        y_pred[i]=0
   

# model prediction remarks     
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


    

