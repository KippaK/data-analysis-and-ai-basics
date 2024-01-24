import pandas as pd
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

no_age_removed = df[~(df['Age']==0)]

age_avg_fix = sum(no_age_removed['Age']) / no_age_removed['Age'].count()

df['Age'] = df['Age'].apply(lambda x: x if x>0 else x+age_avg_fix)

star_class_name = df[df['PClass']=='*']['Name']

dead_count = sum(df['Survived']==0)
surv_count = sum(df['Survived']==1)

dead_precentage = round(dead_count / count * 100, 1)
surv_precentage = 100 - dead_precentage

dead_m_count = sum(df[df['GenderCode']==0]['Survived']==0)
dead_f_count = sum(df[df['GenderCode']==1]['Survived']==0)

surv_m_count = m_count = dead_m_count
surv_f_count = d_count = dead_m_count

dead_m_precentage = round(dead_m_count / m_count * 100, 1)
dead_f_precentage = round(dead_f_count / f_count * 100, 1)

surv_m_precentage = 100 - dead_m_precentage
surv_f_precentage = 100 - dead_f_precentage