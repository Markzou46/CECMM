import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, roc_curve
import numpy as np

# Load data
df_train = pd.read_csv('merged_train.csv')
df_valid = pd.read_csv('merged_valid.csv')
df_test = pd.read_csv('merged_test.csv')

# Features and labels
features = ['RA_score', 'Male0', 'preAFP_bi', 'diameter_bi', 'Age_bi']
X_train, y_train = df_train[features], df_train['y_true']
X_valid, y_valid = df_valid[features], df_valid['y_true']
X_test, y_test = df_test[features], df_test['y_true']

# Model training
model = SVC(probability=True)
model.fit(X_train, y_train)

# Define evaluation function
def bootstrap_confidence_interval(metric_func, y_true, y_pred, y_prob=None, n_bootstrap=1000, alpha=0.95):
    rng = np.random.default_rng()
    stats = [metric_func(y_true[indices := rng.integers(0, len(y_true), len(y_true))],
                         y_prob[indices] if y_prob is not None else y_pred[indices])
             for _ in range(n_bootstrap)]
    return np.percentile(stats, [(1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100])

# Calculate metrics
def evaluate_and_save_results(dataset_name, X, y, model, df):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    df[f'y_pred_{dataset_name}'] = y_pred
    df[f'y_prob_{dataset_name}'] = y_prob
    df.to_csv(f'{dataset_name}_results.csv', index=False)

    metrics_dict = {'Accuracy': accuracy_score, 'AUC': roc_auc_score, 'F1-score': f1_score, 'Recall': recall_score}
    for metric_name, metric_func in metrics_dict.items():
        score = metric_func(y, y_prob if metric_name == 'AUC' else y_pred)
        lower, upper = bootstrap_confidence_interval(metric_func, y, y_pred, y_prob)
        print(f'{dataset_name} {metric_name}: {score:.4f} (95% CI: {lower:.4f}-{upper:.4f})')

    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'{dataset_name} ROC (AUC={roc_auc_score(y, y_prob):.2f})')

# Evaluate results
for name, X, y, df in [('train', X_train, y_train, df_train), ('valid', X_valid, y_valid, df_valid), ('test', X_test, y_test, df_test)]:
    evaluate_and_save_results(name, X, y, model, df)
