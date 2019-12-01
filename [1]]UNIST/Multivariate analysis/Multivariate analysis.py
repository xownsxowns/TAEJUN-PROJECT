import pandas as pd

df = pd.read_csv('/Users/Taejun/Documents/GitHub/Python_project/[1]]UNIST/Multivariate analysis/day.csv')

def generate_dummies(df, dummy_column):
    dummies = pd.get_dummies(df[dummy_column], prefix=dummy_column)
    df = pd.concat([df, dummies], axis=1)
    return df

data = pd.DataFrame.copy(df)
dummy_columns = ["season", "yr", "mnth", "weekday", "weathersit"]
for dummy_column in dummy_columns:
    data = generate_dummies(data, dummy_column)

# remove the original categorical variables: "season", "yr", "mnth", "weekday", "weathersit"
for dummy_column in dummy_columns:
    del data[dummy_column]

data.to_csv('data.csv', header=True)
