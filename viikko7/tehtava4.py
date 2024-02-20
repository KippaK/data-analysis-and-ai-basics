import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sc

df = pd.read_excel('tt.xlsx')

df_corr = df[['sukup','ikä','perhe','koulutus','palkka']]

corr_matrix = df_corr.corr()

sns.heatmap(corr_matrix, annot=True)
plt.show()

age_money_corr = sc.pearsonr(df_corr['ikä'], df_corr['palkka'])
print('Pearson correlation coeffiecent: ',age_money_corr[0])
print('Pearson p-value: ',age_money_corr[1])

age_money_corr = sc.spearmanr(df_corr['ikä'], df_corr['palkka'])
print('Spearman correlation coeffiecent: ',age_money_corr[0])
print('Spearman p-value: ',age_money_corr[1])

sns.regplot(data=df_corr,y='palkka',x='ikä', ci=None)
plt.show()