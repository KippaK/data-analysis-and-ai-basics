import pandas as pd
import matplotlib.pyplot as plot

df = pd.read_csv('emp-dep.csv', dtype={'phone1':str, 'phone2':str})

age_group_counts = df['age_group']
bar = plot.bar(age_group_counts)
plot.show()