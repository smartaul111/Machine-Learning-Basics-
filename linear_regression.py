import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
 
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#pred
Y_pred=regressor.predict(X_test)
#accuracy
from sklearn.metrics import r2_score
print(r2_score(Y_test,Y_pred)*100)
#visual

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary VS Experience (training test)')
plt.xlabel('Years of experience')
plt.ylabel('Salary of the employee')
plt.show()

plt.scatter(X_test,Y_test)
plt.plot(X_train,regressor.predict(X_train),color='green') 
plt.title('Salary VS Experience (testing test)')
plt.xlabel('Years of experience')
plt.ylabel('Salary of the employee')
plt.show()

regressor.predict(5)

