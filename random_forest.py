import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

'''from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.7,random_state=0)
''' 
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""
#linear
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=150,random_state=0)
regressor.fit(X,Y)

#polynomial

y_pred=regressor.predict(6.5)

#visual poly
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Truth or Bluff")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()