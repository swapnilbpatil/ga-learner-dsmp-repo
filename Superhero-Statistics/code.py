# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#path of the data file- path
data=pd.read_csv(path)
#Code starts here 
replace=data['Gender'].replace("-","Agender",inplace=True)
gender_count=data["Gender"].value_counts()
#IMP(.index,gender_count)
plt.bar(gender_count.index,gender_count)
plt.show()


# --------------
#Code starts here
alignment=data["Alignment"].value_counts()
print(alignment)
plt.pie(alignment,labels=alignment)
plt.title('Character Alignment')
plt.show()


# --------------
#Code starts here
#always remember use [[]] & .copy()
sc_df = data[['Strength','Combat']].copy()

#Finding covariance between 'Strength' and 'Combat'
sc_covariance = sc_df.cov().iloc[0,1]
print("Covariance between Strength and Combat is :", sc_covariance)

#Finding the standard deviation of 'Strength' and 'Combat'
sc_strength = sc_df ['Strength'].std()
sc_combat = sc_df ['Combat'].std()
print("Standard Deviation of Strength is :", sc_strength)
print("Standard Deviation of combat is :", sc_combat)
#Calculating the Pearson's correlation between 'Strength' and 'Combat'
sc_pearson = sc_covariance/(sc_strength*sc_combat)
print("Pearson's Correlation Coefficient between Strength and Combat : ",sc_pearson)


#Subsetting the data with columns ['Intelligence', 'Combat']
ic_df = data[['Intelligence','Combat']].copy()

#Finding covariance between 'Intelligence' and 'Combat'
ic_covariance = ic_df.cov().iloc[0,1]
print("Covariance between Intelligence and Combat is :", ic_covariance)

#Finding the standard deviation of 'Intelligence' and 'Combat'
ic_intelligence = ic_df ['Intelligence'].std()
ic_combat = ic_df ['Combat'].std()
print("Standard Deviation of intelligence is :", ic_intelligence)
print("Standard Deviation of combat is :", ic_combat)
#Calculating the Pearson's correlation between 'intelligence' and 'Combat'
ic_pearson = ic_covariance/(ic_intelligence*ic_combat)
print("Pearson's Correlation Coefficient between Intelligence and Combat : ",ic_pearson)






# --------------
#Code starts here
total_high = data['Total'].quantile(q=0.99)
print(total_high)
super_best = data[data['Total'] > total_high]
print(super_best)
super_best_names = list(super_best['Name'])
print(super_best_names)


# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3)=plt.subplots(1,3,figsize=[40,30])
plt.show()
ax_1.boxplot(super_best['Intelligence'])
#in title
ax_1.set(title=('Intelligence'))
ax_2.boxplot(super_best['Speed'])
ax_2.set(title=('Speed'))
ax_3.boxplot(super_best['Power'])
ax_3.set(title=('Power'))
plt.show()



