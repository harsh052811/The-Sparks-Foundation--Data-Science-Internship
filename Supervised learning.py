#!/usr/bin/env python
# coding: utf-8

#  # GRIP: The Sparks Foundation
#  
# **Data Science And Business Analytics Internship**  
# 
# **Name : Harsh Wani**  
# 
# **Task 1: Prediction using Supervised ML**  
#  

# **Predict the percentage of an student based on the no. of study hours**

# In[7]:


# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Reding the data**

# In[9]:


url='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
data=pd.read_csv(url)


# **Exploring the data**

# In[12]:


data.head()


# In[13]:


data.info()


# In[14]:


data.describe()


# In[17]:


data.plot(kind='scatter',x='Hours',y='Scores');
plt.show()


# In[18]:


data.corr(method='pearson')


# In[19]:


hours=data['Hours']
scores=data['Scores']


# In[26]:


sns.displot(hours)


# In[27]:


sns.displot(scores)


# **Linear Regression**

# In[38]:


X= data.iloc[:, :-1].values
y= data.iloc[:,1].values


# In[39]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=50)


# In[42]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)


# In[43]:


m=reg.coef_
c=reg.intercept_
line=m*X+c
plt.scatter(X,y)
plt.plot(X,line);
plt.show()


# In[44]:


y_pred=reg.predict(X_test)


# In[45]:


actual_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_pred})
actual_predicted


# In[46]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# **What will be predicted score if a student studies for 9.25 hrs/ day?**

# In[48]:


v=9.25
s=reg.predict([[v]])
print(s)


# In[50]:


print("So if a student studies for 9.25 hours a day then he/she gets",s,"% marks in the exam")


# **Model Evaluation**

# In[51]:


from sklearn import metrics
from sklearn.metrics import r2_score


# In[52]:


print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred))


# In[53]:


print('R2 score:',r2_score(y_test,y_pred))


# In[ ]:




