import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB

def bar_chart(feature):
    survived = original[original['Income']==1][feature].value_counts()
    dead = original[original['Income']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


''' Retrieve and fix data '''
original = pd.read_csv('Dataset.csv', header=None, index_col=None)
original.columns = ['Age', 'Work', 'Edu-Lvl', 'Edu-Years', 'Marriage-Status', 'Occupation', 'Relationship', 'Gender',
                    'Cap-Gain', 'Cap-Loss', 'Hours', 'Income']

original.replace(["?", "? ", " ?", " ? "], np.nan, inplace=True)
original.replace(" <=50K", 0, inplace=True)
original.replace(" >50K", 1, inplace=True)
# creating new column instead of replace
# original["Income_cleaned"]=original["Income"].astype('category')
# original["Income_cleaned"]=original["Income_cleaned"].cat.codes
original.dropna()


''' Analyze Data '''
# print(original)
print(original.head(5))
# print(original.columns.values)
# print("Data shape", original.shape)

# Info on numerical and categorical values
# print(original.info())
# print(original.describe(include=['O']))
# print(original.describe())

# Number of empty values
# print(original.isnull().sum())
# print(original.head(80))
# print(original.describe(include="all"))

# original["Income_cleaned"]=original["Income"].astype('category')
# original["Income_cleaned"]=original["Income_cleaned"].cat.codes


# print(original[['Work', 'Income_cleaned']].groupby(['Work'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )
# print(original[['Edu-Lvl', 'Income_cleaned']].groupby(['Edu-Lvl'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )
# print(original[['Marriage-Status', 'Income_cleaned']].groupby(['Marriage-Status'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )
# print(original[['Occupation', 'Income_cleaned']].groupby(['Occupation'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )
# print(original[['Relationship', 'Income_cleaned']].groupby(['Relationship'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )
# print(original[['Gender', 'Income_cleaned']].groupby(['Gender'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )

# print(original[['Age', 'Income_cleaned']].groupby(['Age'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )
# print(original[['Edu-Years', 'Income_cleaned']].groupby(['Edu-Years'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )
# print(original[['Cap-Gain', 'Income_cleaned']].groupby(['Cap-Gain'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )
# print(original[['Cap-Loss', 'Income_cleaned']].groupby(['Cap-Loss'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )
# print(original[['Hours', 'Income_cleaned']].groupby(['Hours'], as_index=False).mean().sort_values(by='Income_cleaned', ascending=False) )


