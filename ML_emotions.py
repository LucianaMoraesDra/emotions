import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pickle

# load data
dataset_norm = pd.read_csv('./data/training_dataset_RoLuKe_distances.csv')

# Separate X and y
y = dataset_norm['Target']
X = dataset_norm.drop(columns ='Target', axis=1)

# Split dataset train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 41, test_size = 0.3)


# SVC Linear
clf_model = SVC(kernel='linear', gamma='auto')
clf_model.fit(X_train.values, y_train)
y_pred = clf_model.predict(X_train)
y_pred_test = clf_model.predict(X_test)

# salve model
pickle.dump(clf_model, open('./model_smile_sad_neutre.sav', 'wb'))

print('LINEAR : f1_score (y_train, y_pred):', f1_score(y_train, y_pred, average = "macro"))
print('LINEAR : accuracy_score (y_train, y_pred):', accuracy_score(y_train, y_pred))
print()
print('LINEAR : f1_score (y_test, y_pred_test):', f1_score(y_test, y_pred_test, average = "macro"))
print('LINEAR : accuracy_score (y_test, y_pred_test):', accuracy_score(y_test, y_pred_test))
print()
print()

# SVC Poly
# clf_model_poly = SVC(kernel='poly', gamma='auto')
# clf_model_poly.fit(X_train, y_train)
# y_pred_1 = clf_model_poly.predict(X_train)
# y_pred_1_test = clf_model_poly.predict(X_test)

# print('POLY : f1_score (y_train, y_pred_1):', f1_score(y_train, y_pred_1, average = "macro"))
# print('POLY : accuracy_score (y_train, y_pred_1):', accuracy_score(y_train, y_pred_1))
# print()
# print('POLY : f1_score (y_test, y_pred_1_test):', f1_score(y_test, y_pred_1_test, average = "macro"))
# print('POLY : accuracy_score (y_test, y_pred_1_test):', accuracy_score(y_test, y_pred_1_test))

