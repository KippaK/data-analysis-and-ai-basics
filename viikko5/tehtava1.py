import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('tt.xlsx')

koulutus = ['Peruskoulu', '2. aste', 'Korkeakoulu','Ylempi korkeakoulu']

df_freq = pd.crosstab(df['koulutus'], 'Lukumäärä')
df_freq.index = koulutus

df_freq['%'] = df_freq['Lukumäärä'] / df_freq['Lukumäärä'].sum() * 100

df_freq = df_freq.reset_index()
df_freq.columns = ['Koulutus','Lukumäärä','%']

sns.barplot(data=df_freq,x='Lukumäärä',y='Koulutus')
plt.show

print(df_freq)