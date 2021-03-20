#!/usr/bin/env python
# coding: utf-8

# **GRIP: The Sparks Foundation**
#  
# **Data Science And Business Analytics Internship**  
#  
# **Name : Harsh Wani**  
#  
# **Task-2 Prediction using Unsupervised learning on IRIS dataset**
# 

# **Step-1 Import the libraries and the required data**

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

# To ignore the warnings 
import warnings as wg
wg.filterwarnings("ignore")


# In[4]:


# Reading data iris dataset 
df = pd.read_csv('Downloads/Iris.csv')


# In[6]:


df.head(10)


# **Step-2 Visualising the Data**

# In[8]:


df.shape


# In[10]:


df.info()


# In[11]:


df.columns


# In[12]:


df['Species'].unique()


# In[13]:


df.describe()


# In[14]:


# now we will drop the label column because it is an unsupervised learning problem 
iris = pd.DataFrame(df)
iris_df = iris.drop(columns= ['Species' ,'Id'] )
iris_df.head()


# **Step-3 Find the optimum number of clusters**

# In[15]:


# Calculating the within-cluster sum of square

within_cluster_sum_of_square = []

clusters_range = range(1,15)
for k in clusters_range:
    km = KMeans(n_clusters=k)
    km = km.fit(iris_df)
    within_cluster_sum_of_square.append(km.inertia_)


# In[16]:


# Plotting the "within-cluster sum of square" against clusters range

plt.plot(clusters_range, within_cluster_sum_of_square, 'go--', color='green')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of square')
plt.grid()
plt.show()


# **Step-4 Apply K means clustering**

# In[17]:


from sklearn.cluster import KMeans

model = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
predictions = model.fit_predict(iris_df)


# **Step-5 Visualising the clusters**

# In[18]:


x = iris_df.iloc[:, [0, 1, 2, 3]].values
plt.scatter(x[predictions == 0, 0], x[predictions == 0, 1], s = 25, c = 'red', label = 'Iris-setosa')
plt.scatter(x[predictions == 1, 0], x[predictions == 1, 1], s = 25, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[predictions == 2, 0], x[predictions == 2, 1], s = 25, c = 'green', label = 'Iris-virginica')

# Plotting the cluster centers

plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




