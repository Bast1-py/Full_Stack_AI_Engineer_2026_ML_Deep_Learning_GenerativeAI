import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
from catboost import CatBoostClassifier


url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

features = [
    'Pclass', 'Sex', 'Age', 'Fare', 'Embarked'
]
target = 'Survived'

df.fillna(
    {
        'Age': df['Age'].median()
    }, inplace=True
),

df.fillna(
    {
        'Embarked': df['Embarked'].mode()[0]
    }, inplace=True
)

label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Training Data Shape: {X_train.shape}')
print(f'Testing Data Shape: {X_test.shape}')

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
print(f'\nLightGBM Accuracy Score: {accuracy_score(y_test, lgb_pred):.4f}')
print(f'\nClassification Report: \n{classification_report(y_test, lgb_pred)}')



cat_features = ['Pclass', 'Sex', 'Embarked']
cat_model = CatBoostClassifier(cat_features=cat_features, verbose=0)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
print(f'\nCatBoost Accuracy Score: {accuracy_score(y_test, cat_pred):.4f}')
print(f'\nClassification Report: \n{classification_report(y_test, cat_pred)}')