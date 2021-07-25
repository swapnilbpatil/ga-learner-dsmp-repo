# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Code starts here
data = pd.read_csv(path)
data ["Rating"].plot(kind = 'hist')
plt.show()
data = data[data["Rating"] <= 5]
data ["Rating"].plot(kind = 'hist')
plt.show()

#Code ends here


# --------------
# code starts here

# Original Data (data)
total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())
missing_data = pd.concat([total_null,percent_null],axis = 1, keys = ["Total","Percent"])
print(missing_data)

# Drop null value in data and creat New Data
data_1 = data.dropna()

# New Data(data_1)
total_null_1 = data_1.isnull().sum()
percent_null_1 = (total_null_1/data_1.isnull().count())
missing_data_1 = pd.concat([total_null_1,percent_null_1],axis = 1, keys = ["Total","Percent"])
print(missing_data_1)

# code ends here


# --------------

#Code starts here
a = sns.catplot(x="Category",y="Rating",data=data,kind="box",height = 10)
a.set_xticklabels(rotation=90)
a.set_titles('Rating vs Category [BoxPlot]')


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
#Exploration Instructions
# value counts of Installs
data["Installs"].value_counts()

#Cleaning Instructions
#Remove , and + from Installs
data['Installs'] = data['Installs'].str.replace('+','').str.replace(',', '')
data.sort_values('Installs')

#Convert the Installs column to datatype int
data['Installs'] = data['Installs'].astype(int)

#Create a label encoder
le = LabelEncoder()
le.fit(data['Installs'])
data['Installs']=le.transform(data['Installs'])

#Using seaborn, plot the regplot
a = sns.regplot(x="Installs", y="Rating",data=data)
a.set_title('Rating vs Installs [RegPlot]')

#Code ends here



# --------------
#Code starts here
#Print value counts of Price
data['Price'].value_counts()

#Remove dollar sign from Price
data['Price'] = data['Price'].str.replace('$','')

#Convert the Price column to datatype float
data['Price'] = data['Price'].astype(float)

#Using seaborn, plot the regplot
a = sns.regplot(x="Price", y="Rating",data=data)
a.set_title('Rating vs Price [RegPlot]')


#Code ends here


# --------------

#Code starts here
data['Genres'].unique()

#Split the values of column
data['Genres'] = data['Genres'].str.split(';', n=1, expand=True)

#groupby
gr_mean = data.groupby(['Genres'], as_index=False)['Rating'].mean()

gr_mean.describe()

#Sort the values
gr_mean = gr_mean.sort_values(['Rating'])
print(gr_mean)


#Code ends here


# --------------

#Code starts here


data['Last Updated']

#Convert Last Updated to DateTime format
data['Last Updated'] =  pd.to_datetime(data['Last Updated'])

#Find out the max value in Last Updated
max_date = data['Last Updated'].max()

#Create new column Last Updated Days which is the difference between max_date and values of column Last Updated
data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days

#Using seaborn, plot the regplot
a = sns.regplot(x="Last Updated Days", y="Rating",data=data)
a.set_title('Rating vs Last Updated [RegPlot]')



#Code ends here


