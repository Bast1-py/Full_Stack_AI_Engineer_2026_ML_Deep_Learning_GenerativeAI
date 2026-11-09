import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE


df = pd.read_csv('Telco-Customer-Churn.csv')
print('Datasets Preview: \n')
print(df.info())
print('\Class Distribution: \n')
print(df['churn'].value_counts())
print('\nSample Data: \n', df.head())

# handling missing value
df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
df.fillna(
    {
        'total_charges': df['total_charges'].median()
    }, inplace=True
)

# encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'churn':
        df[column] = label_encoder.fit_transform(df[column])

# encode target variable
df['churn'] = label_encoder.fit_transform(df['churn'])

# scale numerical features
scaler = StandardScaler()
numerical_features = [
    'tenure',
    'monthly_charges',
    'total_charges'
]
df[numerical_features] = scaler.fit_transform(df[numerical_features])


# features and target
X = df.drop(columns=['churn'])
y = df['churn']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print('\n Class Distribution after SMOTE: \n')
print(pd.Series(y_train_resampled).value_counts())


# train random forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])


# train XGBoost
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])


# train LightGBM
lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_lgb = lgb_model.predict(X_test)
roc_auc_lgb = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])


# classification reports
print('Random Forest Report: \n', classification_report(y_test, y_pred_rf))
print('XGBoost Report: \n', classification_report(y_test, y_pred_xgb))
print('LightGBM Report: \n', classification_report(y_test, y_pred_lgb))


# ROC_AUC comparison
print('ROC-AUC Scores: \n')
print(f'Random Forest: {roc_auc_rf:.3f}')
print(f'XGBoost: {roc_auc_xgb:.3f}')
print(f'LightGBM: {roc_auc_lgb:.3f}')