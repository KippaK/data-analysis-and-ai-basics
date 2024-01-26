import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('unemployment.xlsx')


df_period = df.loc[:,['Period', 'Unemployed']].groupby('Period').sum().reset_index()

sns.lineplot(df_period, x='Period', y='Unemployed')
plt.ticklabel_format(style='plain', axis='y')
plt.show()


df_period_g = df.loc[:,['Period', 'Unemployed', 'Gender']].groupby(['Period', 'Gender']).sum().reset_index()
genders = ['Men', 'Women']
for g in genders:
    sns.lineplot(df_period_g[df_period_g['Gender']==g], x='Period',y='Unemployed',label=g)

plt.ticklabel_format(style='plain', axis='y')
plt.ylabel('Unemployed')
plt.show()


df_period_a = df.loc[:,['Period', 'Age', 'Unemployed']].groupby(['Period','Age']).sum().reset_index()
age_groups = ['16 to 19 years', '20 to 24 years', '25 to 34 years', '35 to 44 years', '45 to 54 years', '55 to 64 years', '65 years and over']

for age in age_groups:
    sns.lineplot(df_period_a[df_period_a['Age']==age], x='Period',y='Unemployed',label=age)

plt.ticklabel_format(style='plain', axis='y')
plt.ylabel('Unemployed')
plt.show()
