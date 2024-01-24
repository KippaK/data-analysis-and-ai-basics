import pandas as pd
import math
from datetime import datetime, timedelta
from statistics import median

df_data = pd.read_csv('Titanic_data.csv')
df_name = pd.read_csv('Titanic_names.csv')

desc_data = df_data.describe()
desc_name = df_name.describe()

info_data = df_data.info()
info_name = df_name.info()

df = pd.merge(df_data, df_name, how='inner', on='id')

count = df['id'].count()

m_count = sum(df['GenderCode']==0)
f_count = sum(df['GenderCode']==1)

age_avg = sum(df['Age']) / df['Age'].count()
age_mean = median(df['Age'])

no_age_count = sum(df['Age']==0)