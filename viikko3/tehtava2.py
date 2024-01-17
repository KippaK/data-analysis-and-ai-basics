import pandas as pd
from datetime import datetime, timedelta

df_dep = pd.read_csv('departments.csv')
df_emp = pd.read_csv('employees.csv', dtype={'phone1':str, 'phone':str})

desc = df_emp.describe()

print(df_emp['lname'].unique())
print(df_emp['salary'].nlargest(5))

df = pd.merge(df_emp, df_dep, how='inner', on='dep')

df.drop('image', axis=1, inplace=True)

emp_count = df['id'].count()

# f_count = df['gender'].value_counts()[1]
# m_count = df['gender'].value_counts()[0]

m_count = sum(df['gender']==0)
f_count = sum(df['gender']==1)

df['age'] = (datetime.now() - pd.to_datetime(df['bdate'])) // timedelta(365.2425)

bins = list(range(15, 75, 5))
labels = bins[:-1]
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)