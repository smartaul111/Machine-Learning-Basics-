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

# Splitting the dataset into the Training set and Test set


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components =2)
X_train = lda.fit_transform(X_train, Y_train)
X_test = lda.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#r2score
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)*100)


