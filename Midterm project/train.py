import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# parameters
OUTPUT_FILE = 'model_rf.bin'

model_params = {
    'n_estimators':556,
    'max_depth': 20,
    'min_samples_split': 11,
    'random_state':42
}

# data preparation
df = pd.read_csv("creditcard.csv")

df.columns = df.columns.str.lower()
df = df.drop_duplicates()


X_full_train, X_test = train_test_split(df, test_size=0.2, random_state=1, stratify=df['class'])

y_full_train = X_full_train['class'].values
y_test = X_test['class'].values

del X_full_train['class']
del X_test['class']

# training

def train(X_train, y_train, model_params):
    cols_scale = ['time', 'amount']
    
    scaler = StandardScaler()
    X_train[cols_scale] = scaler.fit_transform(X_train[cols_scale])

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(**model_params)
    model.fit(X_train_sm, y_train_sm)

    return scaler, model

def predict(X_test, scaler, model):
    cols_scale = ['time', 'amount']
    X_test[cols_scale]  = scaler.transform(X_test[cols_scale])

    y_pred = model.predict_proba(X_test)[:, 1]

    return y_pred

# training the final model

print('training the final model')
scaler, model = train(X_full_train, y_full_train, model_params)
y_pred = predict(X_test, scaler, model)

auc_roc = roc_auc_score(y_test, y_pred)
prec, rec, thresh = precision_recall_curve(y_test, y_pred)
pr_auc = auc(rec, prec)

print(f'pr_auc={pr_auc}')

with open(OUTPUT_FILE, 'wb') as f_out:
    pickle.dump((scaler, model), f_out)

print(f'the model is saved to {OUTPUT_FILE}')