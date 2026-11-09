# hands-on exercise: evaluate a model using cross-validation to obtain a more accurate estimate of model performance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier(random_state=42)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print('Cross-Validation Scores: ', cv_scores)
print('Mean Accuracy: ', cv_scores.mean())



# hands-on project: generate and interpret a confusion matrix for a classification model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.show()

print('\n\nClassification Report:')
print(classification_report(y_test, y_pred))