#!/usr/bin/env python
# coding: utf-8

# In[1]:


#manipulations
import numpy as np
import pandas as pd

#viulization
import matplotlib.pyplot as plt
import seaborn as sns

#interactivity
from ipywidgets import interact


# In[2]:


#read data set
data = pd.read_csv('Crop_recommendation.csv')


# In[3]:


#check shape of data
print("Shape of Data set :", data.shape)


# In[4]:


#head of data set
data.head()


# In[5]:


#check for missing values
data.isnull().sum()


# In[6]:


#crops present in data
data['label'].value_counts()


# In[8]:


#summary for crops
print("Average Ratio of Nitrogen in the Soil : {0:.2f}".format(data['N'].mean()))
print("Average Ratio of Phosphorous in the Soil : {0:.2f}".format(data['P'].mean()))
print("Average Ratio of Potassium in the Soil : {0:.2f}".format(data['K'].mean()))
print("Average Temperature in Celsius : {0:.2f}".format(data['temperature'].mean()))
print("Average Relative Humidity in % : {0:.2f}".format(data['humidity'].mean()))
print("Average PH value of soil : {0:.2f}".format(data['ph'].mean()))
print("Average Rainfall in mm : {0:.2f}".format(data['rainfall'].mean()))


# In[14]:


#Statistics for each crop
@interact 
def summary(crops = list(data['label'].value_counts().index)):
                         x = data[data['label'] == crops]
                         print("------------------------------")
                         print("Statistics for Nitrogen")
                         print("Minimum Nitrogen required :", x['N'].min())
                         print("Average Nitrogen required :", x["N"].mean())
                         print("Maximum Nitrogen required :", x["N"].max())
                         print("------------------------------")
                         print("Statistics for Phosphorous")
                         print("Minimum Phosphorous required :", x['P'].min())
                         print("Average Phosphorous required :", x["P"].mean())
                         print("Maximum Phosphorous required :", x["P"].max())
                         print("------------------------------")
                         print("Statistics for Potassium")
                         print("Minimum Potassium required :", x['K'].min())
                         print("Average Potassium required :", x["K"].mean())
                         print("Maximum Potassium required :", x["K"].max())
                         print("------------------------------")
                         print("Statistics for Temperature")
                         print("Minimum Temperature required :", x['temperature'].min())
                         print("Average Temperature required :", x["temperature"].mean())
                         print("Maximum Temperature required :", x["temperature"].max())
                         print("------------------------------")
                         print("Statistics for Humidity")
                         print("Minimum Humidity required :", x['humidity'].min())
                         print("Average Humidity required :", x["humidity"].mean())
                         print("Maximum Humidity required :", x["humidity"].max())
                         print("------------------------------")
                         print("Statistics for PH")
                         print("Minimum PH required :", x['ph'].min())
                         print("Average PH required :", x["ph"].mean())
                         print("Maximum PH required :", x["ph"].max())
                         print("------------------------------")
                         print("Statistics for Rainfall")
                         print("Minimum Rainfall required :", x['rainfall'].min())
                         print("Average Rainfall required :", x["rainfall"].mean())
                         print("Maximum Rainfall required :", x["rainfall"].max())


# In[43]:


# Compare Average Requirment for each crops with average conditions
@interact
def compare(conditions =['N','P','K','temperature','ph','humidity','rainfall']):
    print("Average Value for", conditions,"is {0:.2f}".format(data[conditions].mean()))
    print("-----------------------------------------------------------")
    print("Rice : {0:.2f}".format(data[(data['label'] == 'rice')][conditions].mean()))
    print("Black Grams : {0:.2f}".format(data[(data['label'] == 'blackgram')][conditions].mean()))
    print("Banana : {0:.2f}".format(data[(data['label'] == 'banana')][conditions].mean()))
    print("Jute : {0:.2f}".format(data[(data['label'] == 'jute')][conditions].mean()))
    print("Coconut : {0:.2f}".format(data[(data['label'] == 'coconut')][conditions].mean()))
    print("Apple : {0:.2f}".format(data[(data['label'] == 'apple')][conditions].mean()))
    print("Papaya : {0:.2f}".format(data[(data['label'] == 'papaya')][conditions].mean()))
    print("Muskmelon : {0:.2f}".format(data[(data['label'] == 'muskmelon')][conditions].mean()))
    print("Grapes : {0:.2f}".format(data[(data['label'] == 'grapes')][conditions].mean()))
    print("Watermelon : {0:.2f}".format(data[(data['label'] == 'watermelon')][conditions].mean()))
    print("Kidney Bean : {0:.2f}".format(data[(data['label'] == 'kidneybean')][conditions].mean()))
    print("Mung Bean : {0:.2f}".format(data[(data['label'] == 'mungbean')][conditions].mean()))
    print("Oranges : {0:.2f}".format(data[(data['label'] == 'orange')][conditions].mean()))
    print("Chick Pea : {0:.2f}".format(data[(data['label'] == 'chickpea')][conditions].mean()))
    print("Lentils : {0:.2f}".format(data[(data['label'] == 'lentil')][conditions].mean()))
    print("Cotton : {0:.2f}".format(data[(data['label'] == 'cotton')][conditions].mean()))
    print("Maize : {0:.2f}".format(data[(data['label'] == 'maize')][conditions].mean()))
    print("Moth Beans : {0:.2f}".format(data[(data['label'] == 'mothbeans')][conditions].mean()))
    print("Pigeon Peas : {0:.2f}".format(data[(data['label'] == 'pigeonbpeas')][conditions].mean()))
    print("Mango : {0:.2f}".format(data[(data['label'] == 'mango')][conditions].mean()))
    print("Pomegranate : {0:.2f}".format(data[(data['label'] == 'pomegranate')][conditions].mean()))
    print("Coffee : {0:.2f}".format(data[(data['label'] == 'coffee')][conditions].mean()))


# In[19]:


#continued
@interact
def compare(conditions = ['N','P','K','tempature','ph','humidity','rainfall']):
    print("crops which require greater than average", conditions,'\n')
    print(data[data[conditions]>data[conditions].mean()]['label'].unique())
    print("------------------------------------------------")
    print("Crops which require less than average", conditions,'\n')
    print(data[data[conditions]<=data[conditions].mean()]['label'].unique())


# # Distribution

# In[67]:


#Distribution of Agriculture Conditions
plt.rcParams['figure.figsize'] =(15,7)

plt.subplot(2,4,1)
sns.histplot(data['N'], color = 'lightblue', kde=True)
plt.xlabel('Ratio of Nitrogen', fontsize = 12)
plt.grid()

plt.subplot(2,4,2)
sns.histplot(data['P'], color = 'darkblue', kde=True)
plt.xlabel('Ratio of Phosporous', fontsize = 12)
plt.grid()

plt.subplot(2,4,3)
sns.histplot(data['K'], color = 'purple', kde=True)
plt.xlabel('Ratio of Potassium', fontsize = 12)
plt.grid()

plt.subplot(2,4,4)
sns.histplot(data['temperature'], color = 'yellow', kde=True)
plt.xlabel('Temperature in Celsius', fontsize = 12)
plt.grid()

plt.subplot(2,4,5)
sns.histplot(data['humidity'], color = 'orange', kde=True)
plt.xlabel('Relative Humidity', fontsize = 12)
plt.grid()

plt.subplot(2,4,6)
sns.histplot(data['ph'], color = 'pink', kde=True)
plt.xlabel('PH value of soil', fontsize = 12)
plt.grid()

plt.subplot(2,4,7)
sns.histplot(data['rainfall'], color = 'green', kde=True)
plt.xlabel('Rainfall in mm', fontsize = 12)
plt.grid()


# In[68]:


#Patterns
print("Interesting Patterns")
print("---------------------------")
print("Crops which requires very High Ratio of Nitrogen Content in Soil:", data[data['N']>120]['label'].unique())
print("Crops which requires very High Ratio of Phosphorous Content in Soil:", data[data['P']>100]['label'].unique())
print("Crops which requires very High Ratio of Potassium Content in Soil:", data[data['K']>200]['label'].unique())
print("Crops which requires very High Rainfall Content in Soil:", data[data['rainfall']>200]['label'].unique())
print("Crops which requires very Low Temerature:", data[data['temperature']<10]['label'].unique())
print("Crops which requires very High Temerature:", data[data['temperature']>40]['label'].unique())
print("Crops which requires very Low Humidity:", data[data['humidity']<20]['label'].unique())
print("Crops which requires very Low pH:", data[data['ph']< 4]['label'].unique())
print("Crops which requires very High pH:", data[data['ph']>9]['label'].unique())


# In[71]:


#Seasonality of crops
print("Summer Crops")
print(data[(data['temperature']>30)& (data['humidity']>50)]['label'].unique())
print("--------------------------------------------------------")
print("Winter Crops")
print(data[(data['temperature']<20) & (data['humidity']>30)]['label'].unique())
print("--------------------------------------------------------")
print("Rainy Crops")
print(data[(data['rainfall']<200) & (data['humidity']>30)]['label'].unique())


# In[76]:


#Cluster
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

#spending score and annual income
x= data.loc[:, ['N', "P", "K", "temperature", 'ph', 'humidity', 'rainfall']].values

print(x.shape)

x_data =pd.DataFrame(x)
x_data.head()


# In[78]:


#Optimum Number of Clusters within Dataset

plt.rcParams['figure.figsize'] = (10,4)

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("the Elbow Method", fontsize=20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# In[82]:


#Kmeans algorithm to perform Clustering analysis
km=KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means=km.fit_predict(x)

a=data['label']
y_means=pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis=1)
z = z.rename(columns = {0: 'cluster'})

print("Lets check the Results After Applying the K Means Clustering Analysis \n")
print("Crops in First Cluster:", z[z['cluster']==0]['label'].unique())
print('---------------------------------------------------')
print("Crops in Second Cluster:", z[z['cluster']==1]['label'].unique())
print('---------------------------------------------------')
print("Crops in Third Cluster:", z[z['cluster']==2]['label'].unique())
print('---------------------------------------------------')
print("Crops in Fourth Cluster:", z[z['cluster']==3]['label'].unique())
print('---------------------------------------------------')


# In[84]:


#split the Dataset for Predictive Modelling

y= data['label']
x =data.drop(['label'], axis=1)

print("shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[85]:


#Training and Testing Sets for Validation of Results
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

print("The Shape of x train:", x_train.shape)
print("The Shape of x train:", x_test.shape)
print("The Shape of y train:", y_train.shape)
print("The Shape of y train:", y_test.shape)


# In[87]:


#Predictive Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[88]:


#Evaluate the Model Performance
from sklearn.metrics import classification_report

cr= classification_report(y_test, y_pred)
print(cr)


# In[90]:


data.head()


# # Prediction

# In[ ]:


# prediction = model.predict((np.array([[90,40,40,20,80,7,200]])))
print("The Suggested Crop for Given Climatic Condition is:", prediction)


# In[92]:


data[data['label']=='orange'].head()


# In[93]:


prediction = model.predict((np.array([[20,30,20,25,90,7.5,100]])))
print("The Suggested Crop for Given Climatic Condition is:", prediction)


# In[ ]:




