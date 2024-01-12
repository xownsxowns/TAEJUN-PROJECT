import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from autogluon.tabular import TabularDataset, TabularPredictor
pd.DataFrame.iteritems = pd.DataFrame.items

train = pd.read_csv('E:/[0] Study/[9] 데이터분석/playground-series-s4e1/train.csv')
test = pd.read_csv('E:/[0] Study/[9] 데이터분석/playground-series-s4e1/test.csv')
origin = pd.read_csv('E:/[0] Study/[9] 데이터분석/playground-series-s4e1/Churn_Modelling.csv')
train = pd.concat([train.drop(["id", "Surname"], axis=1), origin.drop(["RowNumber", "Surname"], axis=1)], ignore_index=True)
test.drop(["id", "Surname"], axis=1, inplace=True)

train.dropna(inplace=True)
train.drop_duplicates(inplace = True)

train['HasCrCard'] = train['HasCrCard'].astype('int')
train['IsActiveMember'] = train['IsActiveMember'].astype('int')

test['HasCrCard'] = test['HasCrCard'].astype('int')
test['IsActiveMember'] = test['IsActiveMember'].astype('int')

def add_features(df):
    df['Geo_Gender'] = df['Geography'] + "_" + df['Gender']
    df['AgeGroup'] = df['Age'] // 10 * 10
    df['IsSenior'] = df['Age'].apply(lambda x: 1 if x >= 65 else 0)
    #df['IsYoung'] = df['Age'].apply(lambda x: 1 if x < 25 else 0)
    df['Balance_to_Salary_Ratio'] = (df['Balance'] / df['EstimatedSalary']).astype('float32')
    df['CreditScoreTier'] = pd.cut(df['CreditScore'], bins=[0, 650, 750, 850], labels=['Low', 'Medium', 'High'])
#     df['Products_Per_Tenure'] = (df['NumOfProducts'] / df['Tenure']).astype('float32')
    df['IsActive_by_CreditCard'] = (df['HasCrCard'] * df['IsActiveMember']).astype('float32')
    df['Customer_Status'] = df['Tenure'].apply(lambda x: 'New' if x < 2 else 'Long-term')
    return df

train = add_features(train)
test = add_features(test)

train = TabularDataset(train)
test = TabularDataset(test)

automl = TabularPredictor(label='Exited', problem_type='binary', eval_metric='roc_auc')
automl.fit(train, presets='best_quality')

automl.leaderboard()

prediction = automl.predict_proba(test)

data_submit = pd.read_csv('E:\[0] Study\[9] 데이터분석\playground-series-s4e1\sample_submission.csv')
data_submit.Exited = prediction[1]
data_submit[['id', 'Exited']].to_csv('PG_s4e1_autogluon.csv', index=False)
