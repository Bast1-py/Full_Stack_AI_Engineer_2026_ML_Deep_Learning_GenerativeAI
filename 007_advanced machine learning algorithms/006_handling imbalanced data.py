import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
df = pd.read_csv(url)

print('Datasets Info: \n')
print(df.info())
print('\nClass Distribution: \n')
print(df['Class'].value_counts())

X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('\n Classification Report: \n')
print(classification_report(y_test, y_pred))

roc_auc_score = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(f'\n ROC AUC Score: \n{roc_auc_score:.3f}')


smote = SMOTE(random_state=42)
X_sampled, y_sampled = smote.fit_resample(X_train, y_train)
print('\nClass Distribution After SMOTE: \n')
print(pd.Series(y_sampled).value_counts())

rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_sampled, y_sampled)
y_pred_smote = rf_model_smote.predict(X_test)
print('\n Classification Report (SMOTE): \n')
print(classification_report(y_test, y_pred_smote))

roc_auc_score_smote = roc_auc_score(y_test, rf_model_smote.predict_proba(X_test)[:,1])
print(f'\nROC-AUC Score (SMOTE): {roc_auc_score_smote:.3f}')