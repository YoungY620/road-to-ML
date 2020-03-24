import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model,metrics

# load the data set
X, y = datasets.load_boston(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test \
    = train_test_split(X,y,test_size=0.4, random_state=1)

# Linear Regression model
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)

print("Coefficient:  ",reg.coef_)
print("score:        ", reg.score(X_test,y_test))

# visualize
plt.scatter(reg.predict(X_train),reg.predict(X_train)-y_train,
            color='red', s=50,label='Train data')
plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test,
            color='blue', s=10,label='Test data')
plt.hlines(y=0,xmin=0,xmax=50, linewidth=2)

plt.legend(loc='upper right')

plt.title("Residual error")

plt.show()
