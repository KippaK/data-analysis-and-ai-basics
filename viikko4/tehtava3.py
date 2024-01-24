import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('emp-dep.csv', dtype={'phone1':str, 'phone2':str})

g_counts = df['gender'].value_counts()
g_str = {0: 'miehet', 1:'naiset'}

pie = plt.pie(x=g_counts.values, labels=g_str.values(), autopct='%.1f%%')
plt.show()

age_group_m_counts = pd.Series(df[df['gender']==0]['age_group'].value_counts()).reset_index().sort_values('age_group')
age_group_f_counts = pd.Series(df[df['gender']==1]['age_group'].value_counts()).reset_index().sort_values('age_group')


age_group_f_counts.fillna(0, inplace=True)
age_group_m_counts.fillna(0, inplace=True)

categories = df['age_group'].unique()

# Set the width of the bars
bar_width = 0.35

# Calculate the positions of the bars
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width

# Plotting the bar graph
plt.bar(bar_positions1, age_group_m_counts, width=bar_width, label='Data 1', color='blue')
plt.bar(bar_positions2, age_group_f_counts, width=bar_width, label='Data 2', color='orange')

# Adding labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Graph with Two Arrays')

# Adding legend
plt.legend()

# Display the plot
plt.show()