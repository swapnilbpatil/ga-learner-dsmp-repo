# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset = pd.read_csv(path)


# look at the first five columns
print(dataset.head())

# Check if there's any column which is not useful and remove it like the column id
dataset = dataset.drop('Id',axis=1)

# check the statistical description
dataset.describe()








# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = 'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18','Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type'

#number of attributes (exclude target)
size =len(cols)

#x-axis has target attribute to distinguish between classes
x = dataset['Cover_Type']

#y-axis shows values of an attribute
y = dataset.iloc[:, :-1]

#Plot violin for all attributes
ax = sns.violinplot(data = dataset)



# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
#Select the first 10 features
subset_train = dataset.iloc[:, :10]

#Calculate the Pearson correlation
data_corr = subset_train.corr()

#Plot a heatmap
sns.heatmap(data=data_corr)

#List the correlation
correlation = data_corr.unstack().sort_values(kind='quicksort')

corr_var_list_1 = correlation[(correlation > upper_threshold) | (correlation < lower_threshold)]

corr_var_list = corr_var_list_1[corr_var_list_1 != 1]
print(corr_var_list)

# Code ends here




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
import numpy as np

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)



# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1:]

#Split the data into chunks
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, 
random_state=0)
#Standardized
size = 10
scaler = StandardScaler()
#Apply transform only for continuous data
X_train_temp = scaler.fit_transform(X_train.iloc[:, 0:size])
X_test_temp = scaler.fit_transform(X_test.iloc[:, 0:size])

#Concatenate scaled continuous data and categorical
X_train1 = np.concatenate((X_train_temp, X_train.iloc[:,size:]), axis=1)
X_test1 = np.concatenate((X_test_temp, X_test.iloc[:,size:]), axis=1) 

#Create a dataframe of rescaled data
scaled_features_train_df = pd.DataFrame(X_train1, index= X_train.index, columns=X_train.columns)
print(scaled_features_train_df.head()) 
scaled_features_test_df = pd.DataFrame(X_test1, index= X_test.index, columns=X_train.columns)
print(scaled_features_test_df.head()) 









# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
skb = SelectPercentile(score_func=f_classif,percentile=90)
predictors = skb.fit_transform(X_train1, Y_train)
scores = list(skb.scores_)

Features = scaled_features_train_df.columns

dataframe = pd.DataFrame({'Features':Features,'Scores':scores})

dataframe=dataframe.sort_values(by='Scores',ascending=False)

top_k_predictors = list(dataframe['Features'][:predictors.shape[1]])

print(top_k_predictors)










# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression())

##Computing values for the all-features model
model_fit_all_features = clf1.fit(X_train, Y_train)
predictions_all_features = clf1.predict(X_test)
score_all_features = accuracy_score(Y_test, predictions_all_features)
print(score_all_features)
##Computing values for the top-features model
model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], Y_train)
predictions_top_features = model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])
score_top_features = accuracy_score(Y_test, predictions_top_features)
print(score_top_features)









