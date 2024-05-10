#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 


# In[3]:


data = pd.read_csv(r'C:\Users\arjun\Documents\DataScience\Assignment\dataset_part_2.csv')
data


# In[4]:


x = pd.read_csv(r'C:\Users\arjun\Documents\DataScience\Assignment\dataset_part_3.csv')


# In[5]:


x


# # TASK 1

# Create a NumPy array from the column Class in data, by applying the method to_numpy() then assign it to the variable Y,make sure the output is a Pandas series (only one bracket df['name of column']).

# In[6]:


# Create a NumPy array from the 'Class' column and convert it to a Pandas Series
Y = pd.Series(data['Class'].to_numpy())

# Print the Series
print(Y)


# # TASK 2

# 
# Standardize the data in X then reassign it to the variable X using the transform provided below.

# In[7]:


transform = preprocessing.StandardScaler()


# In[8]:


X = transform.fit_transform(x)


# In[9]:


X


# We split the data into training and testing data using the function train_test_split. The training data is divided into validation data, a second set used for training data; then the models are trained and hyperparameters are selected using the function GridSearchCV.

# # TASK  3

# Use the function train_test_split to split the data X and Y into training and test data. Set the parameter test_size to 0.2 and random_state to 2. The training data and test data should be assigned to the following labels.

# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[11]:


Y_test.shape


# # TASK 4

# Create a logistic regression object then create a GridSearchCV object logreg_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.

# In[12]:


parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}


# In[13]:


parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
logreg=LogisticRegression()


# We output the GridSearchCV object for logistic regression. We display the best parameters using the data attribute best_params_ and the accuracy on the validation data using the data attribute best_score_.

# In[14]:


logreg_cv = GridSearchCV(estimator=logreg, param_grid=parameters, cv=10)


# In[15]:


# Fit the GridSearchCV object to the training data
logreg_cv.fit(X_train, Y_train)


# In[16]:


print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# # TASK 5

# Calculate the accuracy on the test data using the method score:

# In[17]:


# Calculate the accuracy on the test data
accuracy = logreg_cv.best_estimator_.score(X_test, Y_test)


# In[18]:


print(f"Accuracy on test data: {accuracy:.2f}")


# Lets look at the confusion matrix:

# In[19]:


yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# # TASK 6

# Create a support vector machine object then create a GridSearchCV object svm_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.

# In[20]:


parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()


# In[21]:


svm_cv = GridSearchCV(estimator=svm, param_grid=parameters, cv=10)


# In[22]:


svm_cv.fit(X_train, Y_train)


# In[23]:


print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# # TASK 7

# Calculate the accuracy on the test data using the method score:

# In[24]:


# Calculate the accuracy on the test data
accuracy = svm_cv.best_estimator_.score(X_test, Y_test)
accuracy


# In[25]:


yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# # TASK 8

# Create a decision tree classifier object then create a GridSearchCV object tree_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.

# In[26]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()


# In[27]:


tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=10)


# In[28]:


tree_cv.fit(X_train, Y_train)


# In[29]:


print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# # TASK 9

# Calculate the accuracy of tree_cv on the test data using the method score:

# In[30]:


accuracy = tree_cv.best_estimator_.score(X_test,Y_test)
accuracy


# In[31]:


yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# # TASK 10

# Create a k nearest neighbors object then create a GridSearchCV object knn_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.

# In[32]:


parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()


# In[33]:


knn_cv = GridSearchCV(estimator=KNN, param_grid=parameters, cv=10)


# In[34]:


knn_cv.fit(X_train, Y_train)


# In[35]:


print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)


# # TASK 11

# Calculate the accuracy of knn_cv on the test data using the method score:

# In[36]:


accuracy = knn_cv.best_estimator_.score(X_test, Y_test)
accuracy


# In[37]:


yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# In[38]:


# Calculate accuracy for each model on the test data
logreg_accuracy = logreg_cv.best_estimator_.score(X_test, Y_test)
svm_accuracy = svm_cv.best_estimator_.score(X_test, Y_test)
tree_accuracy = tree_cv.best_estimator_.score(X_test, Y_test)
knn_accuracy = knn_cv.best_estimator_.score(X_test, Y_test)

# Print the accuracies for each model
print(f"Logistic Regression accuracy on test data: {logreg_accuracy:.6f}")
print(f"SVM accuracy on test data: {svm_accuracy:.6f}")
print(f"Decision Tree accuracy on test data: {tree_accuracy:.6f}")
print(f"K-Nearest Neighbors accuracy on test data: {knn_accuracy:.6f}")

# Compare the accuracies to find the method that performs best
best_accuracy = max(logreg_accuracy, svm_accuracy, tree_accuracy, knn_accuracy)

# Determine which model has the best accuracy
if best_accuracy == logreg_accuracy:
    print("Logistic Regression performs the best.")
elif best_accuracy == svm_accuracy:
    print("SVM performs the best.")
elif best_accuracy == tree_accuracy:
    print("Decision Tree performs the best.")
else:
    print("K-Nearest Neighbors performs the best.")


# In[39]:


knn_cv.best_score_


# In[ ]:




