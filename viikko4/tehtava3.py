import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('emp-dep.csv', dtype={'phone1':str, 'phone2':str})

g_counts = df['gender'].value_counts()
g_str = {0: 'miehet', 1:'naiset'}

pie = plt.pie(x=g_counts.values, labels=g_str.values(), autopct='%.1f%%')
plt.show()

age_group_m_counts = df[df['gender']==0]['age_group'].value_counts().reset_index().sort_values('age_group')
age_group_f_counts = df[df['gender']==1]['age_group'].value_counts().reset_index().sort_values('age_group')
age_group_m_counts.rename(columns={'count':'miehet'}, inplace=True)
age_group_f_counts.rename(columns={'count':'naiset'}, inplace=True)
age_group_g_counts = pd.Series(np.sort(df['age_group'].unique())).to_frame()
age_group_g_counts.rename(columns={0:'age_group'}, inplace=True)

age_group_g_counts = pd.merge(age_group_g_counts, age_group_m_counts, on='age_group', how='outer')
age_group_g_counts = pd.merge(age_group_g_counts, age_group_f_counts, on='age_group', how='outer')

age_group_g_counts.fillna(value=0, inplace=True)


categories = np.sort(df['age_group'].unique())

bar_pos = 0.34

bar_width = 0.5
bar_position=0.5

ax = age_group_g_counts.plot(kind='bar', x='age_group', width=bar_width, position=bar_position, figsize=(10, 6), label='Miehet')

plt.yticks(range(int(age_group_g_counts[['miehet','naiset']].max().max()+1)))
plt.xlabel('Ikäryhmä')
plt.ylabel('Lukumäärä')

plt.show()

