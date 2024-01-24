import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('unemployment.xlsx')

df_period = df.loc[:,['Period', 'Unemployed']].groupby('Period').sum().reset_index()

sns.lineplot(df_period, x='Period', y='Unemployed')
plt.ticklabel_format(style='plain', axis='y')
plt.show()