# LDA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset1 = pd.read_csv('test.csv')
X_train = dataset.iloc[:,0:-1].values
X_test = dataset1.iloc[:,1:].values
Y_train = dataset.iloc[:,20:21].values
Y_train=np.ravel(Y_train)
Y_train.shape


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#fitting ligsitic to trainig set
from sklearn.tree import DecisionTreeClassifier 
classifier=DecisionTreeClassifier(criterion='gini', random_state=0)
classifier.fit(X_train,Y_train)



#predicting the test resutl
 
Y_pred=classifier.predict(X_test)
