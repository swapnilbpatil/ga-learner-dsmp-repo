# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
print(df.head())

#Remove the $ and , from columns
for col in ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']:
    df[col] = df[col].str.replace("$",'')
    df[col] = df[col].str.replace(",",'')

   
    
print(df.head())
#The features
X = df.iloc[:,:-1]

#The target variable
y = df.iloc[:,-1]

#Calculate the value counts of target variable
count = y.value_counts()

#Split the dataframe
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 6)

# Code ends here


# --------------
# Code starts here
for col in ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']:
    X_train[col] = X_train[col].astype('float')
    X_test[col] = X_test[col].astype('float')

print(X_train.info())
print(X_test.info())
# Code ends here


# --------------
# Code starts here
#Drop the rows from columns
X_train.dropna(subset= ['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(subset= ['YOJ','OCCUPATION'],inplace=True)

#Update the index of y_train
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

#fill the missing values for columns
for i in ['AGE','CAR_AGE','INCOME','HOME_VAL']:
    X_train.fillna(X_train[col].mean(),inplace=True)
    X_test.fillna(X_train[col].mean(),inplace=True)



# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for col in columns:
    le = LabelEncoder()
    X_train[col]=le.fit_transform(X_train[col].astype(str))
    X_test[col]=le.fit_transform(X_test[col].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state = 6)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test,y_pred)
print('The accuracy_score:',score)

# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state = 9)

X_train,y_train = smote.fit_sample(X_train,y_train)

scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# Code ends here


# --------------
# Code Starts here

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score = accuracy_score(y_test,y_pred)
print(' The Accuracy Score:',score)
# Code ends here


