
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve, auc,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


df = pd.read_csv('data/pdb_block_group.csv')

features = [
    'Tot_Population_CEN_2010',
    'Tot_Population_ACS_09_13',
    'Males_CEN_2010',
    'Females_CEN_2010',
    'Hispanic_CEN_2010',
    'NH_White_alone_CEN_2010',
    'NH_Blk_alone_CEN_2010',
    'NH_AIAN_alone_CEN_2010',
    'NH_Asian_alone_CEN_2010',
    'NH_NHOPI_alone_CEN_2010',
    'NH_SOR_alone_CEN_2010',
    'Pov_Univ_ACS_09_13',
    'Prs_Blw_Pov_Lev_ACS_09_13',
    'Med_HHD_Inc_BG_ACS_09_13',
    'Med_House_Value_BG_ACS_09_13',
    'LAND_AREA',
    'RURAL_POP_CEN_2010',
    'URBANIZED_AREA_POP_CEN_2010',
    'URBAN_CLUSTER_POP_CEN_2010'
]


for col in ['Med_HHD_Inc_BG_ACS_09_13', 'Med_House_Value_BG_ACS_09_13']:
    if col in df.columns:
        df[col] = df[col].replace(r'[\$,]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')


df = df.dropna(subset=features + ['has_superfund'])
X = df[features].copy()
y = df['has_superfund'].astype(int).copy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)


rf_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),  
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

lr_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=2000, random_state=42))
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_scores(pipeline, X_train, y_train, scoring='roc_auc'):
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return scores

print('CV ROC-AUC RF:', cv_scores(rf_pipeline, X_train, y_train, scoring='roc_auc').mean())
print('CV PR-AUC RF:', cv_scores(rf_pipeline, X_train, y_train, scoring='average_precision').mean())
print('CV ROC-AUC LR:', cv_scores(lr_pipeline, X_train, y_train, scoring='roc_auc').mean())
print('CV PR-AUC LR:', cv_scores(lr_pipeline, X_train, y_train, scoring='average_precision').mean())


rf_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)


y_pred_rf = rf_pipeline.predict(X_test)
y_proba_rf = rf_pipeline.predict_proba(X_test)[:,1]

y_pred_lr = lr_pipeline.predict(X_test)
y_proba_lr = lr_pipeline.predict_proba(X_test)[:,1]


print('\nRandom Forest Test Classification Report:')
print(classification_report(y_test, y_pred_rf, digits=4))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_rf))

print('\nLogistic Regression Test Classification Report:')
print(classification_report(y_test, y_pred_lr, digits=4))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_lr))


roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
pr_precision_rf, pr_recall_rf, _ = precision_recall_curve(y_test, y_proba_rf)
pr_auc_rf = auc(pr_recall_rf, pr_precision_rf)

roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
pr_precision_lr, pr_recall_lr, _ = precision_recall_curve(y_test, y_proba_lr)
pr_auc_lr = auc(pr_recall_lr, pr_precision_lr)

print(f'\nRandom Forest: ROC-AUC={roc_auc_rf:.4f}, PR-AUC={pr_auc_rf:.4f}')
print(f'Logistic Regression: ROC-AUC={roc_auc_lr:.4f}, PR-AUC={pr_auc_lr:.4f}')


RocCurveDisplay.from_estimator(rf_pipeline.named_steps['clf'], 
                               rf_pipeline.named_steps['scaler'].transform(X_test),
                               y_test).plot()

RocCurveDisplay.from_estimator(rf_pipeline, X_test, y_test).plot()
plt.title('ROC curve - Random Forest')
plt.savefig(os.path.join(figures_dir, 'roc_rf_pipeline.png'))
plt.close()

RocCurveDisplay.from_estimator(lr_pipeline, X_test, y_test).plot()
plt.title('ROC curve - Logistic Regression')
plt.savefig(os.path.join(figures_dir, 'roc_lr_pipeline.png'))
plt.close()


PrecisionRecallDisplay(precision=pr_precision_rf, recall=pr_recall_rf).plot()
plt.title('Precision-Recall - Random Forest')
plt.savefig(os.path.join(figures_dir, 'pr_rf.png'))
plt.close()

PrecisionRecallDisplay(precision=pr_precision_lr, recall=pr_recall_lr).plot()
plt.title('Precision-Recall - Logistic Regression')
plt.savefig(os.path.join(figures_dir, 'pr_lr.png'))
plt.close()


print('\nPermutation importance (RF) -- running on test set (this is expensive but informative)...')

X_test_scaled = rf_pipeline.named_steps['scaler'].transform(X_test)
rf_clf = rf_pipeline.named_steps['clf']

p_imp = permutation_importance(rf_clf, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({'feature': features, 'importance_mean': p_imp.importances_mean, 'std': p_imp.importances_std})
print(perm_df.sort_values('importance_mean', ascending=False).head(10))

pd_test = pd.DataFrame({'y_true': y_test, 'y_prob_rf': y_proba_rf})
print('\nProbabilities summary by true class (RF):')
print(pd_test.groupby('y_true')['y_prob_rf'].describe())

#Test comment for git

threshold = 0.3
y_pred_thresh = (y_proba_rf >= threshold).astype(int)
print(f'\nConfusion matrix RF at threshold {threshold}:')
print(confusion_matrix(y_test, y_pred_thresh))
print(classification_report(y_test, y_pred_thresh, digits=4))

