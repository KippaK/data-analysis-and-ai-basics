import pandas as pd
import matplotlib.pyplot as plot

df = pd.read_csv('emp-dep.csv', dtype={'phone1':str, 'phone2':str})

scat = plot.scatter(df['age'], df['salary'])
plot.show()

dep_counts = df['dname'].value_counts()

bar = plot.bar(dep_counts.index, dep_counts.values)
plot.show()