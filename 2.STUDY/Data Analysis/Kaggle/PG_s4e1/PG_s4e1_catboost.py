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
pd.DataFrame.iteritems = pd.DataFrame.items

train = pd.read_csv('/Volumes/TAEJUN/[0] Study/[9] 데이터분석/playground-series-s4e1/train.csv')
train.head()
test = pd.read_csv('/Volumes/TAEJUN/[0] Study/[9] 데이터분석/playground-series-s4e1/test.csv')
test.head()

submission = pd.read_csv('/Volumes/TAEJUN/[0] Study/[9] 데이터분석/playground-series-s4e1/sample_submission.csv')


train['HasCrCard'] = train['HasCrCard'].astype('int')
train['IsActiveMember'] = train['IsActiveMember'].astype('int')

test['HasCrCard'] = test['HasCrCard'].astype('int')
test['IsActiveMember'] = test['IsActiveMember'].astype('int')

def add_features(df):
    df['Geo_Gender'] = df['Geography'] + "_" + df['Gender']
    df['AgeGroup'] = df['Age'] // 10 * 10
    df['IsSenior'] = df['Age'].apply(lambda x: 1 if x >= 65 else 0)
    #df['IsYoung'] = df['Age'].apply(lambda x: 1 if x < 25 else 0)
    df['Balance_to_Salary_Ratio'] = df['Balance'] / df['EstimatedSalary']
    df['CreditScoreTier'] = pd.cut(df['CreditScore'], bins=[0, 650, 750, 850], labels=['Low', 'Medium', 'High'])
    df['Products_Per_Tenure'] = df['NumOfProducts'] / df['Tenure']
    df['IsActive_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']
    df['Customer_Status'] = df['Tenure'].apply(lambda x: 'New' if x < 2 else 'Long-term')
    return df

train = add_features(train)
test = add_features(test)

cat_cols = ['Surname', 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Tenure', 'NumOfProducts', 'CreditScoreTier','Geo_Gender', 'Customer_Status']
num_cols = ['CreditScore','Age','Balance','EstimatedSalary',  'AgeGroup',
        'Balance_to_Salary_Ratio', 'Products_Per_Tenure', 'IsActive_by_CreditCard', 'IsSenior']
all_features = cat_cols + num_cols
target = 'Exited'

n_splits = 3 
clfs = []
scores = [] 

X = train[all_features]
y = train[target]

kf = KFold(n_splits=n_splits, shuffle=True, random_state=7575)
for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    train_dataset = Pool(data=X_train, label=y_train, cat_features=cat_cols)
    eval_dataset = Pool(data=X_test, label=y_test, cat_features=cat_cols)

    clf = CatBoostClassifier(
        depth=4,
        iterations=3500,
        learning_rate=0.06,
        loss_function="Logloss",
        custom_metric=["AUC"],
        
        cat_features=cat_cols,
     
        colsample_bylevel=0.098,
        subsample=0.95,
        l2_leaf_reg=9,
        min_data_in_leaf=243,
        max_bin=187,
        random_strength=1,
        
        task_type="CPU",    
        thread_count=-1,
        bootstrap_type="Bernoulli", 
        
        random_seed=42,
        auto_class_weights="Balanced",#"SqrtBalanced",
        early_stopping_rounds=50)

    clfs.append(clf)

    clf.fit(
        train_dataset,
        eval_set=eval_dataset,
        verbose=500,
        use_best_model=True,
        plot=False)

    scores.append(np.mean([v for k, v in clf.best_score_["validation"].items() if "AUC" in k], dtype="float16"))

print("mean AUC score on validation", np.mean(scores, dtype="float16"), np.mean(scores, dtype="float16") - np.std(scores, dtype="float16"))

y_pred = np.zeros((test.shape[0], train[target].nunique()))

for n, clf in enumerate(clfs):
    y_pred += clf.predict_proba(test[all_features])

submission[target] = y_pred[:,1]/len(clfs)

submission.to_csv('submission.csv',index=False)
submission