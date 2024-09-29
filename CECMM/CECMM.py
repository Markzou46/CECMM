import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the datasets
df_train = pd.read_csv('merged_train.csv')
df_valid = pd.read_csv('merged_valid.csv')
df_test = pd.read_csv('merged_test.csv')

# Define features (X) and labels (y) for training, validation, and testing
X_train = df_train[['CEMMscore', 'Male0', 'preAFP_bi', 'diameter_bi', 'Age_bi']]
y_train = df_train['y_true']
X_valid = df_valid[['CEMMscore', 'Male0', 'preAFP_bi', 'diameter_bi', 'Age_bi']]
y_valid = df_valid['y_true']
X_test = df_test[['CEMMscore', 'Male0', 'preAFP_bi', 'diameter_bi', 'Age_bi']]
y_test = df_test['y_true']

# Create and train the Support Vector Classifier (SVC) model
model = SVC(probability=True)
model.fit(X_train, y_train)

# Make predictions on the training set and save results
y_train_pred = model.predict(X_train)
y_train_prob = model.predict_proba(X_train)[:, 1]
df_train['CECMMscore_pred'] = y_train_pred
df_train['CECMMscore_prob'] = y_train_prob
df_train.to_csv('CECMMscore_train.csv', index=False)

# Evaluate the model performance on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_prob)
train_fpr, train_tpr, _ = metrics.roc_curve(y_train, y_train_prob)
print(f'Train ACC: {train_accuracy:.4f}, AUC: {train_auc:.4f}')

# Make predictions on the validation set and save results
y_valid_pred = model.predict(X_valid)
y_valid_prob = model.predict_proba(X_valid)[:, 1]
df_valid['CECMMscore_pred'] = y_valid_pred
df_valid['CECMMscore_prob'] = y_valid_prob
df_valid.to_csv('CECMMscore_valid.csv', index=False)

# Evaluate the model performance on the validation set
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
valid_auc = roc_auc_score(y_valid, y_valid_prob)
valid_fpr, valid_tpr, _ = metrics.roc_curve(y_valid, y_valid_prob)
print(f'Valid ACC: {valid_accuracy:.4f}, AUC: {valid_auc:.4f}')

# Make predictions on the test set and save results
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]
df_test['CECMMscore_pred'] = y_test_pred
df_test['CECMMscore_prob'] = y_test_prob
df_test.to_csv('CECMMscore_test.csv', index=False)

# Evaluate the model performance on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)
test_fpr, test_tpr, _ = metrics.roc_curve(y_test, y_test_prob)
print(f'Test ACC: {test_accuracy:.4f}, AUC: {test_auc:.4f}')
