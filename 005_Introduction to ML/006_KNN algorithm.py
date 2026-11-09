# hands-on exercise: implement KNN for a classification task, experimenting with diffrent values of k
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# for k in range(1,11):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#
#     y_pred = knn.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f'k = {k}, Accuracy = {accuracy:.2f}')



# hands-on exercise: compare KNN results to logistic regression, analyzing accuracy and other metrics

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print('\n\nLogistics Regression Accuracy: ', accuracy_lr)

best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy k={best_k}: ', accuracy_knn)

# detail comparison
print('\nLogistic Regression Classification: ')
print(classification_report(y_test, y_pred_knn))

print('\n\nKNN Classification: ')
print(classification_report(y_test, y_pred_knn))

