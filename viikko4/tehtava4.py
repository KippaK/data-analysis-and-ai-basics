import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('unemployment.xlsx')

df_period = df.loc[:,['Period', 'Unemployed']].groupby('Period').sum().reset_index()

sns.lineplot(df_period, x='Period', y='Unemployed')
plt.ticklabel_format(style='plain', axis='y')
plt.show()

df_period_g = df.loc[:,['Period', 'Unemployed', 'Gender']].groupby(['Period', 'Gender']).sum().reset_index()
df_period_f = df_period_g[df_period_g['Gender']=='Women']
df_period_m = df_period_g[df_period_g['Gender']=='Men']
df_period_g = pd.merge(df_period_m, df_period_f, on='Period', suffixes=('Men','Women'))
df_period_g = df_period_g.loc[:,['Period','UnemployedMen','UnemployedWomen']]

df_period_g.rename({'UnemployedWomen':'Women','UnemployedMen':'Men'}, axis=1, inplace=True)

sns.lineplot(df_period_g,x='Period',y='Men', label='Men')
sns.lineplot(df_period_g,x='Period',y='Women', label='Women')
plt.ticklabel_format(style='plain', axis='y')
plt.ylabel('Unemployed')
plt.show()

# Viimeinen
# Jokaiselle ikäryhmälle oma viiva
# df_period_a = df.loc[.,['Period','Unemployed','Age']].groupby([''])