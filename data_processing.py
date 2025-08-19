"""
Data Processing Module for canine-cancer-lab-data-benchmark
============================================================

This module contains function for data preprocessing
including sex_status encoding, MICE imputation, feature engineering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit

# Data preprocessing pipeline
def preprocess_data(X):
    """Comprehensive data preprocessing pipeline"""
    X_processed = X.copy()
    
    # Handle sex_status encoding
    sex_mapping = {
        'Spayed Female': 0,
        'Neutered Male': 1,
        'Intact Male': 2,
        'Intact Female': 3
    }
    X_processed['sex_status'] = X_processed['sex_status'].map(sex_mapping)
    
    # Create clinically relevant ratios

    # Neutrophil-to-Lymphocyte Ratio (NLR)
    if 'CBC_neutrophils' in X_processed.columns and 'CBC_lymphocytes' in X_processed.columns:
        X_processed['NLR'] = X_processed['CBC_neutrophils'] / X_processed['CBC_lymphocytes']
        print("✓ Neutrophil-to-Lymphocyte Ratio (NLR) created")

    # Platelet-to-Lymphocyte Ratio
    if 'CBC_platelets' in X_processed.columns and 'CBC_lymphocytes' in X_processed.columns:
        X_processed['PLR'] = X_processed['CBC_platelets'] / X_processed['CBC_lymphocytes']
        print("✓ Platelet-to-Lymphocyte Ratio (PLR) created")
        
    # Handle infinite values in the new ratios
    print("--- Handling Infinite Values ---")
    infinite_cols = []
    for col in ['NLR', 'PLR']:
        if col in X_processed.columns:
            inf_count = np.isinf(X_processed[col]).sum()
            if inf_count > 0:
                print(f"Found {inf_count} infinite values in {col}")
                X_processed[col] = X_processed[col].replace([np.inf, -np.inf], np.nan)
                infinite_cols.append(col)

    if infinite_cols:
        print(f"✓ Infinite values replaced with NaN in: {infinite_cols}")
        
    # Separate numeric and categorical features
    numeric_features = X_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fit Iterative Imputer (MICE) on training data
    # we only have numerical missing data 
    imputer = IterativeImputer()
    X_processed[numeric_features] = imputer.fit_transform(X_processed[numeric_features])

    # Fit scaler on training data
    scaler = RobustScaler()
    X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])

    print(f"Final feature matrix shape: {X_processed.shape}")

    return X_processed


# Prepare training data
X_raw = train_data.drop(['target', 'subject_id'], axis=1)
y = train_data['target']

# Preprocess training data
X = preprocess_data(X_raw)

# Convert to integer labels
y = y.astype(int)

# -------------------------------------------------------------------------  #

# Initial 80/20 Split using GroupShuffleSplit
# Use GroupShuffleSplit for patient-level cross-validation
# This first split creates our final, held-out test set.
gss1 = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=RANDOM_STATE)
trainval_idx, test_idx = next(gss1.split(X, y, groups=train_data['subject_id']))
X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]
groups_trainval = train_data['subject_id'].iloc[trainval_idx]
groups_test = train_data['subject_id'].iloc[test_idx]

# this second split creates train, validation set
# train/60%, val/20%, test/20%
gss2 = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=RANDOM_STATE)
train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups=groups_trainval))
X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

# Verify splits
train_pos_pct = (y_train.sum() / len(y_train)) * 100
val_pos_pct = (y_train.sum() / len(y_train)) * 100
test_pos_pct = (y_test.sum() / len(y_test)) * 100

print(f"Training set: {len(X_train)} samples ({train_pos_pct:.2f}% positive)")
print(f"Validation set: {len(X_val)} samples ({val_pos_pct:.2f}% positive)")
print(f"Test set: {len(X_test)} samples ({test_pos_pct:.2f}% positive)")

# Bar plot of positive class percentages
plt.figure(figsize=(8, 6))
sets = ['Train', 'Validation', 'Test']
percentages = [train_pos_pct, val_pos_pct, test_pos_pct]
bars = plt.bar(sets, percentages, color=['blue', 'orange', 'green'], alpha=0.7)
plt.ylabel('Percentage of Positive Class (%)')
plt.title('Positive Class Distribution Across Data Splits')
for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{pct:.2f}%', ha='center', va='bottom')
plt.tight_layout()
# plt.savefig(f'{output_dir}/data_splits_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Verification of No Leakage ---
train_groups = np.unique(groups_trainval)
test_groups = np.unique(groups_test)
print(f"Number of groups in Training set: {len(train_groups)}")
print(f"Number of groups in Test set: {len(test_groups)}")

# The intersection should be an empty array
leakage = np.intersect1d(train_groups, test_groups)
print(f"\nOverlap of groups between train and test: {leakage}")
if len(leakage) == 0:
    print("✅ Success: No data leakage between training and test sets.\n")
else:
    print("❌ Failure: Data leakage detected!\n")


