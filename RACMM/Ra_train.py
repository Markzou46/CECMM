# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, f1_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pymrmr
import pickle
import matplotlib.pyplot as plt

# Experiment identifier
EXP = 'RA_EXP'

# Set random seed for reproducibility
def set_seed(seed=48):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()  # Initialize seed

# Load training set features and labels
with open('train_features.pkl', 'rb') as f:
    train_features = pickle.load(f)
with open('train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)
with open('train_ids.pkl', 'rb') as f:
    train_ids = pickle.load(f)

# Load validation set features and labels
with open('valid_features.pkl', 'rb') as f:
    valid_features = pickle.load(f)
with open('valid_labels.pkl', 'rb') as f:
    valid_labels = pickle.load(f)
with open('valid_ids.pkl', 'rb') as f:
    valid_ids = pickle.load(f)

# Load test set features and labels
with open('test_features.pkl', 'rb') as f:
    test_features = pickle.load(f)
with open('test_labels.pkl', 'rb') as f:
    test_labels = pickle.load(f)
with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

# Convert features from dictionaries to NumPy arrays
train_features_np = np.array([[list(patient_period.values()) for patient_period in patient] for patient in train_features])
valid_features_np = np.array([[list(patient_period.values()) for patient_period in patient] for patient in valid_features])
test_features_np = np.array([[list(patient_period.values()) for patient_period in patient] for patient in test_features])

# Reshape features for model input
train_features_reshaped = train_features_np.reshape(train_features_np.shape[0], -1)
valid_features_reshaped = valid_features_np.reshape(valid_features_np.shape[0], -1)
test_features_reshaped = test_features_np.reshape(test_features_np.shape[0], -1)

# Normalize features
scaler = MinMaxScaler()
train_features_norm = scaler.fit_transform(train_features_reshaped)
valid_features_norm = scaler.transform(valid_features_reshaped)
test_features_norm = scaler.transform(test_features_reshaped)

# Convert labels to NumPy arrays
train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
test_labels = np.array(test_labels)

# Feature selection using MRMR
column_names = [str(i) for i in range(train_features_norm.shape[1])]
df_train = pd.DataFrame(np.column_stack((train_labels, train_features_norm)), columns=['label'] + column_names)
selected_features = pymrmr.mRMR(df_train, 'MIQ', 30)
selected_features_indices = [df_train.columns.get_loc(f) for f in selected_features]

# Apply selected features to train, valid, and test sets
selected_features_train = train_features_norm[:, [idx-1 for idx in selected_features_indices]]
selected_features_valid = valid_features_norm[:, [idx-1 for idx in selected_features_indices]]
selected_features_test = test_features_norm[:, [idx-1 for idx in selected_features_indices]]

# Correct IDs format
train_ids = [id[0] if isinstance(id, list) else id for id in train_ids]
valid_ids = [id[0] if isinstance(id, list) else id for id in valid_ids]
test_ids = [id[0] if isinstance(id, list) else id for id in test_ids]

# Initialize and train SVM classifier
classifier = SVC(kernel='linear', decision_function_shape='ovo', probability=True)
classifier.fit(selected_features_train, train_labels)

# Predict on training, validation, and test sets
train_predictions = classifier.predict(selected_features_train)
train_probas = classifier.predict_proba(selected_features_train)[:, 1]
valid_predictions = classifier.predict(selected_features_valid)
test_predictions = classifier.predict(selected_features_test)
valid_probas = classifier.predict_proba(selected_features_valid)[:, 1]
test_probas = classifier.predict_proba(selected_features_test)[:, 1]

# Compute performance metrics
train_acc = accuracy_score(train_labels, train_predictions)
valid_acc = accuracy_score(valid_labels, valid_predictions)
test_acc = accuracy_score(test_labels, test_predictions)

train_auc = auc(*roc_curve(train_labels, train_probas)[:2])
valid_auc = auc(*roc_curve(valid_labels, valid_probas)[:2])
test_auc = auc(*roc_curve(test_labels, test_probas)[:2])

train_f1 = f1_score(train_labels, train_predictions)
valid_f1 = f1_score(valid_labels, valid_predictions)
test_f1 = f1_score(test_labels, test_predictions)

train_recall = recall_score(train_labels, train_predictions)
valid_recall = recall_score(valid_labels, valid_predictions)
test_recall = recall_score(test_labels, test_predictions)

# Plot ROC curves
fig, ax = plt.subplots()
for (label, probas, true_labels) in [('Train', train_probas, train_labels), ('Valid', valid_probas, valid_labels), ('Test', test_probas, test_labels)]:
    fpr, tpr, _ = roc_curve(true_labels, probas)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{label} ROC curve (area = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")
plt.savefig(f'{EXP}_AUC_curve')

# Print performance metrics
print(f'Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}, Train recall: {train_recall:.4f}')
print(f'Valid Acc: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}, Valid F1: {valid_f1:.4f}, Valid recall: {valid_recall:.4f}')
print(f'Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}, Test recall: {test_recall:.4f}')

# Save predictions and probabilities to CSV
train_results_df = pd.DataFrame({
    'patient_ids': train_ids,
    'y_true': np.ravel(train_labels),
    'RA_pred': train_predictions,
    'RA_score': train_probas
})
train_results_df.to_csv(f'{EXP}_train_prob.csv', index=False)

valid_results_df = pd.DataFrame({
    'patient_ids': valid_ids,
    'y_true': np.ravel(valid_labels),
    'RA_pred': valid_predictions,
    'RA_score': valid_probas
})
valid_results_df.to_csv(f'{EXP}_valid_prob.csv', index=False)

test_results_df = pd.DataFrame({
    'patient_ids': test_ids,
    'y_true': np.ravel(test_labels),
    'RA_pred': test_predictions,
    'RA_score': test_probas
})
test_results_df.to_csv(f'{EXP}_test_prob.csv', index=False)