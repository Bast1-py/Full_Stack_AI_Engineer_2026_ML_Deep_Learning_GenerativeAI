from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
df = pd.read_csv(url)

print('Datasets Info: \n')
print(df.info())
print('\n Class Distribution:\n')
print(df.describe())

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate model
rf_model = RandomForestClassifier(random_state=42)
scores_kfold = cross_val_score(rf_model, X_train, y_train, cv=kf)


print(f'K-fold cross validation scores: {scores_kfold}')
print(f'\nMean Accuracy (K-Fold): {scores_kfold.mean():.4f}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# train and evaluate model
scores_stratified = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='accuracy')
print(f'Stratified K-Fold cross validation scores: {scores_kfold}')
print(f'Mean Accuracy (K-Fold): {scores_kfold.mean():.4f}')