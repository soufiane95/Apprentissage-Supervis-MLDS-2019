#!/usr/bin/env python
# coding: utf-8

# # Projet Apprentissage Supervisé
# 
# 
# <b>Réaslié par :</b>
# <li>FHIYIL Soufiane</li>
# <li>MOUHDA Mohammed Reda</li>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, KFold

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('../creditcard.csv')


# In[3]:


data.shape


# In[4]:


data.head()


# ### Exploratory data analysis

# In[28]:


sns.set()
columns = ['Time','Amount']
sns.pairplot(data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()


# ### plot data on the first principal components

# In[180]:


from numpy.random import rand

plt.figure(figsize=(10,8))

plt.scatter(data.iloc[:, 2], data.iloc[:,3],
            c=data.Class, edgecolor='none', alpha=0.5, s=50, cmap=plt.cm.get_cmap('jet'))


plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();


# ### boxplot

# In[60]:


plt.figure(figsize=(10,8))
data.boxplot(column=['V1', 'V2', 'V3', 'V4', 'V5'])


# ## Resampling (under-sampling & over-sampling)

# A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more examples from the minority class (over-sampling).

# <img src="image.png" />

# The simplest implementation of over-sampling is to duplicate random records from the minority class, which can cause overfitting. In under-sampling, the simplest technique involves removing random records from the majority class, which can cause loss of information.

# <b style="color:red">---> to avoid this loss of information or overfitting, we'll use both methods and combine them in order ta balance our data</b>

# ### Combine both methods (under-sampling & over-sampling)

# #### Split the data (Training data 70% & Test data 30%)

# In[5]:


from sklearn.model_selection import train_test_split

# Split our data
x_train, x_test, y_train, y_test = train_test_split(data.ix[:,1:29],
                                                          data['Class'],
                                                          test_size=0.30,
                                                          random_state=0)


# In[6]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# <b>---> we'll use combining method on training sample<b>

# In[7]:


from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=0)


# In[8]:


import collections

X_resampled, y_resampled = smote_enn.fit_resample(x_train, y_train)


# In[9]:


print(sorted(collections.Counter(y_resampled).items()))


# In[11]:


X_resampled.shape, y_resampled.shape


# ---
# ## Bayesien Naïf

# In[7]:


# View class distributions
data.Class.value_counts()


# In[123]:


from sklearn.naive_bayes import GaussianNB

# Initialize our classifier
bayes = GaussianNB()


# In[132]:


# Train our classifier
model_bayes = bayes.fit(X_resampled, y_resampled)


# In[133]:


# Predict test data
y_pred = model_bayes.predict(x_test)


# #### Accuracy

# In[134]:


from sklearn.metrics import accuracy_score,log_loss


# In[135]:


accuracy_score(y_test,y_pred)


# <b>---> On ne peut pas utiliser l'accuracy comme une métrique de performance, car les classes sont déséquilibres, on aura toujours une accuracy qui est très élevé et c'est à cause de la classe 0 qui représente 90% des données </b>

# In[136]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# #### ROC AUC

# In[138]:


from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred)


# #### Cross Validation for Bayes

# In[139]:


from sklearn.model_selection import cross_val_score, KFold

#init
scores = cross_val_score(bayes, X_resampled, y_resampled, cv=5, scoring='roc_auc')


# In[140]:


scores


# ## LDA

# In[181]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_resampled, y_resampled)

#predict test data
y_pred = lda.predict(x_test)


# #### Cross-validation for LDA

# In[182]:


cross_val_score(lda, X_resampled, y_resampled, cv=5, scoring='roc_auc')


# In[183]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n')
print("=== AUC ===")
print(roc_auc_score(y_test,y_pred))


# ## Quadratic Discriminant Analysis

# In[187]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#init
qda = QuadraticDiscriminantAnalysis()

#fit the data
qda.fit(X_resampled, y_resampled)


# In[188]:


y_pred = qda.predict(x_test)


# #### Cross-validation for Quadratic Discriminant Analysis

# In[189]:


cross_val_score(qda, X_resampled, y_resampled, cv=5, scoring='roc_auc')


# In[190]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n')
print("=== AUC ===")
print(roc_auc_score(y_test,y_pred))


# ## Linear SVM

# In[11]:


from sklearn import svm

#init
svm = svm.SVC(gamma='scale')

#fit the data
svm.fit(X_resampled, y_resampled)


# In[12]:


y_pred = svm.predict(x_test)


# In[15]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n')
print("=== AUC ===")
print(roc_auc_score(y_test,y_pred))


# #### cross-validation

# In[18]:


cross_val_score(svm, X_resampled, y_resampled, cv=5, scoring='roc_auc')


# ## KNN

# In[19]:


knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_resampled, y_resampled)


# In[20]:


# Use the .predict() method to make predictions from the X_test subset
pred = knn.predict(x_test)


# In[21]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n')
print("=== AUC ===")
print(roc_auc_score(y_test,y_pred))


# #### cross-validation

# In[22]:


cross_val_score(knn, X_resampled, y_resampled, cv=5, scoring='roc_auc')


# ## Decision Trees

# In[23]:


#init
dtc = DecisionTreeClassifier()

#fit the model
dtc.fit(X_resampled, y_resampled)

print(dtc)


# In[24]:


y_pred = dtc.predict(x_test)


# In[25]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n')
print("=== AUC ===")
print(roc_auc_score(y_test,y_pred))


# In[26]:


cross_val_score(dtc, X_resampled, y_resampled, cv=5, scoring='roc_auc')


# ## Random Forests

# In[27]:


from sklearn import model_selection


# In[28]:


# random forest model creation
rfc = RandomForestClassifier()

#fit the model
rfc.fit(X_resampled, y_resampled)

# predictions
y_pred = rfc.predict(x_test)


# In[29]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n')
print("=== AUC ===")
print(roc_auc_score(y_test,y_pred))


# In[30]:


cross_val_score(rfc, X_resampled, y_resampled, cv=5, scoring='roc_auc')


# ## LogisticRegression

# In[12]:


# random forest model creation
lr = LogisticRegression()

# fit the model
lr.fit(X_resampled, y_resampled)

# predictions
y_pred = lr.predict(x_test)


# In[13]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n')
print("=== AUC ===")
print(roc_auc_score(y_test,y_pred))


# In[14]:


cross_val_score(lr, X_resampled, y_resampled, cv=5, scoring='roc_auc')


# In[ ]:




