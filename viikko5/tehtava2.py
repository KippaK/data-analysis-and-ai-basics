import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('tt.xlsx')

koulutus = ['Peruskoulu', '2. aste', 'Korkeakoulu','Ylempi korkeakoulu']
sukup = ['Mies', 'Nainen']

df_freq = pd.crosstab(df['koulutus'],df['sukup'])

df_freq.columns = sukup
df_freq.index = koulutus