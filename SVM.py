import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

'''from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.7,random_state=0)
''' 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)
Y=sc_Y.fit_transform(Y.reshape(-1,1))

#fitting svr to dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y.ravel())



y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#visual linear

plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color="green",linestyle = 'dashed')
plt.title("Truth or Bluff(SVR)")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show() 

#visual smooth graph
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue',linewidth = 0.8)
plt.title("Truth or Bluff(SVR)")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show() 


