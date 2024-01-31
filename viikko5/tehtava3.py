import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

df = pd.read_excel('tt.xlsx')

koulutus = ['Peruskoulu', '2. aste', 'Korkeakoulu','Ylempi korkeakoulu']
sukup = ['Mies', 'Nainen']

df_freq = pd.crosstab(df['koulutus'],df['sukup'])

df_freq.columns = sukup
df_freq.index = koulutus

p = chi2_contingency(df_freq)[1]

if p>0.05:
    print(f'Riippuvuus ei ole tilastollisesti merkeitsevä p={p}')
else:
    print(f'Riippuvuus on tialstollisesti merkitsevä p={p}')
    
bar_width = 0.6
bar_position=0.5

df_freq.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()