# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path

#Code starts here
df = pd.read_csv(path)
print(df.head())
#Store all the features(independent values)
X = df.drop(["Price"],axis=1)

#Store the target variable(dependent value)
y = df["Price"]

#Split the dataframe
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 6)

#the correlation between the features
corr = X_train.corr()
print(corr)








# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
r2 = regressor.score(X_test,y_test)
print(r2)








# --------------
from sklearn.linear_model import Lasso

# Code starts here
lasso = Lasso()
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)
r2_lasso = lasso.score(X_test,y_test)
print(r2_lasso)




# --------------
from sklearn.linear_model import Ridge

# Code starts here
ridge = Ridge()

#Fit the model on the training data
ridge.fit(X_train,y_train)

#predictions
ridge_pred = ridge.predict(X_test)

#the r^2 score
r2_ridge = ridge.score(X_test,y_test)
print(r2_ridge)


# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
regressor = LinearRegression()

#Calculate the cross_val_score
score = cross_val_score(regressor,X_train,y_train,scoring = 'r2',cv=10)

#Calculate the mean of
mean_score = np.mean(score)
print(mean_score)



# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
model = make_pipeline(PolynomialFeatures(2), LinearRegression())

#Fit the model on the training data
model.fit(X_train,y_train)

#Make predictions
y_pred = model.predict(X_test)

#Find the r^2 score
r2_poly = model.score(X_test,y_test)
print(r2_poly)







