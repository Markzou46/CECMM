import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, roc_curve

# Load training data
df_train = pd.read_csv('data/train.csv')

# Define features (X) and labels (y)
X_train = df_train[['Male0', 'preAFP_bi', 'diameter_bi', 'Age_bi']]
y_train = df_train['y_true']

# Load validation data
df_valid = pd.read_csv('data/valid.csv')

# Define features (X) and labels (y)
X_valid = df_valid[['Male0', 'preAFP_bi', 'diameter_bi', 'Age_bi']]
y_valid = df_valid['y_true']

# Load test data
df_test = pd.read_csv('data/test.csv')

# Define features (X) and labels (y)
X_test = df_test[['Male0', 'preAFP_bi', 'diameter_bi', 'Age_bi']]
y_test = df_test['y_true']

# Initialize machine learning model (SVC in this case)
model = SVC(probability=True)

# Train the model on the training dataset
model.fit(X_train, y_train)

# Define a function for bootstrap confidence intervals
def bootstrap_confidence_interval(metric_func, y_true, y_pred, y_prob=None, n_bootstrap=1000, alpha=0.95):
    rng = np.random.default_rng()
    stats = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        if y_prob is None:
            score = metric_func(y_true[indices], y_pred[indices])
        else:
            score = metric_func(y_true[indices], y_prob[indices])
        stats.append(score)
    lower = np.percentile(stats, (1 - alpha) / 2 * 100)
    upper = np.percentile(stats, (1 + alpha) / 2 * 100)
    return lower, upper

# Sensitivity (recall for positive class)
def sensitivity(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN)

# Specificity (recall for negative class)
def specificity(y_true, y_pred):
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    return TN / (TN + FP)

# Positive Predictive Value
def ppv(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    FP = sum((y_true == 0) & (y_pred == 1))
    return TP / (TP + FP)

# Negative Predictive Value
def npv(y_true, y_pred):
    TN = sum((y_true == 0) & (y_pred == 0))
    FN = sum((y_true == 1) & (y_pred == 0))
    return TN / (TN + FN)

# Function to evaluate and save results for a given dataset
def evaluate_and_save_results(dataset_name, X, y, model, df):
    # Predict classes and probabilities
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Save predicted results
    df[f'y_pred_{dataset_name}'] = y_pred
    df[f'y_prob_{dataset_name}'] = y_prob
    df.to_csv(f'results/{dataset_name}_prob.csv', index=False)

    # Metrics to evaluate
    metrics_dict = {
        'Accuracy': accuracy_score,
        'AUC': roc_auc_score,
        'F1-score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
    }

    # Evaluate metrics and calculate confidence intervals
    metrics_results = {}
    for metric_name, metric_func in metrics_dict.items():
        if metric_name == 'AUC':
            score = metric_func(y, y_prob)
            lower, upper = bootstrap_confidence_interval(metric_func, y, y_prob)
        else:
            score = metric_func(y, y_pred)
            lower, upper = bootstrap_confidence_interval(metric_func, y, y_pred)
        metrics_results[metric_name] = (score, lower, upper)

    # Add Sensitivity, Specificity, PPV, and NPV with confidence intervals
    metrics_results['Sensitivity'] = sensitivity(y, y_pred), *bootstrap_confidence_interval(sensitivity, y, y_pred)
    metrics_results['Specificity'] = specificity(y, y_pred), *bootstrap_confidence_interval(specificity, y, y_pred)
    metrics_results['PPV'] = ppv(y, y_pred), *bootstrap_confidence_interval(ppv, y, y_pred)
    metrics_results['NPV'] = npv(y, y_pred), *bootstrap_confidence_interval(npv, y, y_pred)

    # Print and save the results
    print(f'\n{dataset_name} results:')
    for metric, (score, lower, upper) in metrics_results.items():
        print(f'{dataset_name} {metric}: {score:.4f} ({lower:.4f} - {upper:.4f})')

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'{dataset_name} ROC (AUC = {metrics_results["AUC"][0]:.2f})')

    # Save metrics to CSV
    results_df = pd.DataFrame(metrics_results, index=['Score', '95% CI Lower', '95% CI Upper']).T
    results_df.to_csv(f'results/{dataset_name}_metrics.csv', index=True)

# Evaluate training set
evaluate_and_save_results('train', X_train, y_train, model, df_train)

# Evaluate validation set
evaluate_and_save_results('valid', X_valid, y_valid, model, df_valid)

# Evaluate test set
evaluate_and_save_results('test', X_test, y_test, model, df_test)

# Plot diagonal line for ROC curve
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# Configure plot appearance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the ROC curve plot
plt.savefig('results/roc_curve.png')
plt.show()