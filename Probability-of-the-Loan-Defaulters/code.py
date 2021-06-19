# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv (path)
p_a = df[df['fico'] > 700].shape[0]/df.shape[0]
p_b = df.loc[df['purpose'] == 'debt_consolidation'].shape[0]/df.shape[0]
df1 = df.loc[df['purpose'] == 'debt_consolidation'].shape[0]
p_a_b = df[df['fico'] > 700].shape[0] / df1
result = (p_a == p_a_b)
print(result)
# code ends here


# --------------
#Calculate the probability p(A)
prob_lp = df[df['paid.back.loan'] == 'Yes'].shape[0]/df.shape[0]
#Calculate the probability p(B)
prob_cs = df[df['credit.policy'] == 'Yes'].shape[0]/df.shape[0]

new_df = df[df['paid.back.loan'] == 'Yes']
#Calculate the probablityp(B|A)
prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0]/new_df.shape[0]

#Calculate the conditional probability
bayes = (prob_pd_cs * prob_lp )/prob_cs
print(bayes)








# --------------
# code starts here
df['purpose'].value_counts(normalize=True).plot(kind='bar')
df1 = df[df['paid.back.loan'] == 'No']

df1['purpose'].value_counts(normalize=True).plot(kind='bar')
# code ends here


# --------------
# code starts here
inst_median = df.installment.median()
inst_mean = df.installment.mean()
plt.hist(df.installment)
l = df['log.annual.inc']
plt.hist(l)

# code ends here


