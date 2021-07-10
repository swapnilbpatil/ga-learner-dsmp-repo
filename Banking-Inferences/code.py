# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
data = pd.read_csv(path)

data_sample = data.sample(2000,random_state=0)

sample_mean = data_sample.installment.mean()

sample_std = data_sample.installment.std()

margin_of_error = z_critical*(sample_std/math.sqrt(sample_size))

confidence_interval = (sample_mean - margin_of_error,sample_mean + margin_of_error)

true_mean = data.installment.mean()

print(true_mean)









# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
#Creating different subplots
fig ,axes = plt.subplots(3,1, figsize=(10,20))

#Running loop to iterate through rows
for i in range (len(sample_size)):
    #initialising a list
    m = []

    #Loop to implement the no. of samples
    for j in range(1000):
        mean = data['installment'].sample(sample_size[i]).mean()

        #Appending the mean to the list
        m.append(mean)
    
    #Converting the list to series
    mean_series = pd.Series(m)

    #Plotting the histogram for the series
    axes[i].hist(mean_series)
    
#Displaying the plot
plt.show()


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
# Removing the last character from the values in column
data['int.rate'] = data["int.rate"].map(lambda x: str(x)[:-1])
#Dividing the column values by 100
data['int.rate'] = data['int.rate'].astype(float)/100
data.head()


z_statistic,p_value = ztest(x1 = data[data['purpose']=='small_business']['int.rate'],value = data['int.rate'].mean(), alternative='larger')

print("z-statistic is:", z_statistic)
print("p-value is:", p_value)






# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic,p_value = ztest(x1 = data[data['paid.back.loan']=='No']['installment'],x2 = data[data['paid.back.loan']=='Yes']['installment'])

print("z-statistic is:", z_statistic)
print("p-value is:", p_value)


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here

yes = data[data['paid.back.loan'] == 'Yes']['purpose'].value_counts()

no = data[data['paid.back.loan'] == 'No']['purpose'].value_counts()

observed = pd.concat([yes.transpose(),no.transpose()],1,keys= ['Yes','No'])

chi2, p, dof, ex = chi2_contingency(observed)

print("Critical value is:", critical_value)

print("chi statistic is:", chi2)


