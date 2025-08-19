
"""
Machine Learning Modeling Module for canine-cancer-lab-data-benchmark
========================================================================

This code implements a comprehensive comparative framework to identify the 
best-performing combination of machine learning algorithm, feature set, and 
data balancing technique, with final model evaluation and SHAP interpretation.

It includes:
1. Complete comparative framework (5 models × 3 feature sets × 7 balancing techniques)
2. Proper hyperparameter tuning with MCC optimization
3. Final model evaluation with comprehensive metrics
4. SHAP interpretation and visualization
5. Bootstrap confidence intervals
6. ROC and Precision-Recall curves

Requirements:
- pandas, numpy, scikit-learn, matplotlib, seaborn
- xgboost, lightgbm (for gradient boosting models)
- imbalanced-learn (for resampling techniques)
- shap (for model interpretation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix, 
                           roc_curve, precision_recall_curve, matthews_corrcoef,
                           precision_score, recall_score, f1_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

# Try to import specialized packages (install if needed)
try:
    import xgboost as xgb
    print("XGBoost imported successfully")
except ImportError:
    print("XGBoost not available - using alternatives")
    xgb = None

try:
    import lightgbm as lgb
    print("LightGBM imported successfully")
except ImportError:
    print("LightGBM not available - using alternatives")
    lgb = None

try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    print("Imbalanced-learn imported successfully")
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    print("Imbalanced-learn not available - using class weights only")
    IMBALANCED_LEARN_AVAILABLE = False

try:
    import shap
    print("SHAP imported successfully")
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available - skipping model interpretation")
    SHAP_AVAILABLE = False

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class MLComparativeFramework:
    """
    Comprehensive Machine Learning Comparative Framework
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = []
        self.best_model = None
        self.best_params = None
        self.best_score = -np.inf
        self.feature_selectors = {}
        
    def apply_feature_selection(self, X_train, y_train, X_val=None, X_test=None, method='all', k=20):
        """
        Apply various feature selection methods
        
        Parameters:
        X_train, y_train : Training data
        X_val, X_test : Validation and test data (optional)
        method : str, feature selection method ('univariate', 'rfe', 'manual', 'all')
        k : int, number of features to select
        
        Returns:
        Dictionary with selected datasets for each method
        """
        feature_sets = {}
        
        if method in ['univariate', 'all']:
            print(f"Applying univariate feature selection (k={k})...")
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_uni = selector.fit_transform(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()]
            
            feature_sets['univariate'] = {
                'X_train': pd.DataFrame(X_train_uni, columns=selected_features, index=X_train.index),
                'features': selected_features,
                'selector': selector
            }
            
            if X_val is not None:
                feature_sets['univariate']['X_val'] = pd.DataFrame(
                    selector.transform(X_val), columns=selected_features, index=X_val.index)
            if X_test is not None:
                feature_sets['univariate']['X_test'] = pd.DataFrame(
                    selector.transform(X_test), columns=selected_features, index=X_test.index)
        
        if method in ['rfe', 'all']:
            print(f"Applying RFE feature selection (k={k})...")
            rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            selector = RFE(rf, n_features_to_select=k)
            X_train_rfe = selector.fit_transform(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()]
            
            feature_sets['rfe'] = {
                'X_train': pd.DataFrame(X_train_rfe, columns=selected_features, index=X_train.index),
                'features': selected_features,
                'selector': selector
            }
            
            if X_val is not None:
                feature_sets['rfe']['X_val'] = pd.DataFrame(
                    selector.transform(X_val), columns=selected_features, index=X_val.index)
            if X_test is not None:
                feature_sets['rfe']['X_test'] = pd.DataFrame(
                    selector.transform(X_test), columns=selected_features, index=X_test.index)
        
        if method in ['manual', 'all']:
            print("Applying manual feature selection...")
            # Select top biomarkers based on statistical significance and 
            # clinical relevance (anemia, Thrombocytopenia, changes in WBC, hypercalcemia, hypoglycemia, elevated Liver Enzymes (ALT, ALP, GGT))
            manual_features = [
                'age_at_visit', 'CHEM_Albumin', 'CHEM_Albumin:Globulin Ratio', 
                'CHEM_Magnesium', 'CBC_lymphocytes', 'CHEM_Globulin', 'CHEM_Calcium',
                'CBC_MCHC', 'CHEM_Sodium', 'CBC_Hemoglobin (HGB)', 'CBC_WBC', 
                'CHEM_GGT', 'CBC_platelets', 'CHEM_Glucose', 'CBC_band neutrophils'
            ]
            
            # Filter features that exist in the dataset
            available_features = [f for f in manual_features if f in X_train.columns]
            print(f"Manual features found: {len(available_features)}/{len(manual_features)}")
            
            if available_features:
                feature_sets['manual'] = {
                    'X_train': X_train[available_features],
                    'features': available_features,
                    'selector': None
                }
                
                if X_val is not None:
                    feature_sets['manual']['X_val'] = X_val[available_features]
                if X_test is not None:
                    feature_sets['manual']['X_test'] = X_test[available_features]
        
        self.feature_selectors = feature_sets
        return feature_sets
    
    def setup_models_and_balancers(self, class_weight_dict):
        """Setup base models and balancing techniques"""
        
        # Base models
        base_models = {
            'LogisticRegression': LogisticRegression(
                class_weight=class_weight_dict, 
                random_state=self.random_state, 
                max_iter=1000
            ),
            'RandomForest': RandomForestClassifier(
                class_weight=class_weight_dict, 
                random_state=self.random_state
            ),
            'MLP': MLPClassifier(
                random_state=self.random_state, 
                max_iter=500
            )
        }
        
        # Add gradient boosting models 
        if xgb is not None:
            base_models['XGB'] = xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        if lgb is not None:
            base_models['LGBM'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1
            )
        
        # Balancing techniques
        balancers = {'None': None}
        
        if IMBALANCED_LEARN_AVAILABLE:
            balancers.update({
                # Oversampling Methods
                'SMOTE': SMOTE(random_state=self.random_state),
                'ADASYN': ADASYN(random_state=self.random_state),
                'RandomOverSampler': RandomOverSampler(random_state=self.random_state),
                # Undersampling Methods
                'RandomUnderSampler': RandomUnderSampler(random_state=self.random_state),
                # Hybrid Methods
                'SMOTETomek': SMOTETomek(random_state=self.random_state),
                'SMOTEEnn': SMOTEENN(random_state=self.random_state)
            })
        
        return base_models, balancers
    
    def setup_hyperparameters(self):
        """Setup hyperparameter grids for each model"""
        param_grids = {
            'LogisticRegression': {'C': [0.1, 1, 10, 100]},
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'alpha': [0.001, 0.01, 0.1]
            }
        }
        
        if xgb is not None:
            param_grids['XGB'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        if lgb is not None:
            param_grids['LGBM'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        return param_grids
    
    def run_comparative_analysis(self, X_train, y_train, X_val, y_val):
        """
        Run the complete comparative analysis
        
        Parameters:
        X_train, y_train : Training data
        X_val, y_val : Validation data
        """
        print("=== STARTING COMPARATIVE ANALYSIS ===")
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"Class weights calculated: {class_weight_dict}")
        
        # Apply feature selection
        feature_sets = self.apply_feature_selection(X_train, y_train, X_val)
        
        # Setup models and balancers
        base_models, balancers = self.setup_models_and_balancers(class_weight_dict)
        param_grids = self.setup_hyperparameters()
        
        print(f"Models: {list(base_models.keys())}")
        print(f"Balancers: {list(balancers.keys())}")
        print(f"Feature sets: {list(feature_sets.keys())}")
        
        # Run comparative analysis
        total_combinations = len(base_models) * len(balancers) * len(feature_sets)
        print(f"\nRunning {total_combinations} combinations...")
        
        combination_count = 0
        for fs_name, fs_data in feature_sets.items():
            X_train_fs = fs_data['X_train']
            X_val_fs = fs_data['X_val']
            
            print(f"\n--- Feature Set: {fs_name} ({X_train_fs.shape[1]} features) ---")
            
            for m_name, model in base_models.items():
                for b_name, balancer in balancers.items():
                    combination_count += 1
                    print(f"\n[{combination_count}/{total_combinations}] {m_name} + {b_name} + {fs_name}")
                    
                    try:
                        # Create pipeline
                        if balancer is not None and IMBALANCED_LEARN_AVAILABLE:
                            pipe_steps = [('balancer', balancer), ('classifier', model)]
                            pipe = ImbPipeline(pipe_steps)
                            param_grid = {'classifier__' + k: v for k, v in param_grids[m_name].items()}
                        else:
                            pipe = Pipeline([('classifier', model)])
                            param_grid = {'classifier__' + k: v for k, v in param_grids[m_name].items()}
                        
                        # Hyperparameter tuning with MCC optimization
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                        grid_search = GridSearchCV(
                            pipe, 
                            param_grid, 
                            cv=cv, 
                            scoring='matthews_corrcoef',  # MCC as specified
                            n_jobs=-1, 
                            error_score='raise'
                        )
                        
                        # Fit and evaluate
                        grid_search.fit(X_train_fs, y_train)
                        y_val_pred = grid_search.predict(X_val_fs)
                        y_val_proba = grid_search.predict_proba(X_val_fs)[:, 1]
                        
                        # Calculate validation metrics
                        val_mcc = matthews_corrcoef(y_val, y_val_pred)
                        val_roc_auc = roc_auc_score(y_val, y_val_proba)
                        val_f1 = f1_score(y_val, y_val_pred)
                        val_precision = precision_score(y_val, y_val_pred)
                        val_recall = recall_score(y_val, y_val_pred)
                        
                        # Store results
                        result = {
                            'Model': m_name,
                            'Balancer': b_name,
                            'FeatureSet': fs_name,
                            'Val_MCC': val_mcc,
                            'Val_ROC_AUC': val_roc_auc,
                            'Val_F1': val_f1,
                            'Val_Precision': val_precision,
                            'Val_Recall': val_recall,
                            'Best_Params': grid_search.best_params_,
                            'CV_MCC': grid_search.best_score_,
                            'Model_Object': grid_search.best_estimator_
                        }
                        
                        self.results.append(result)
                        
                        # Update best model if this is better
                        if val_mcc > self.best_score:
                            self.best_score = val_mcc
                            self.best_model = grid_search.best_estimator_
                            self.best_params = result
                        
                        print(f"MCC: {val_mcc:.4f}, ROC-AUC: {val_roc_auc:.4f}")
                        
                    except Exception as e:
                        print(f"Error with {m_name}-{b_name}-{fs_name}: {str(e)}")
                        continue
        
        # Create results DataFrame
        self.results_df = pd.DataFrame(self.results)
        print(f"\n=== COMPARATIVE ANALYSIS COMPLETE ===")
        print(f"Total successful combinations: {len(self.results)}")
        
        return self.results_df
    
    def get_best_model_summary(self):
        """Get summary of the best performing model"""
        if self.best_params is None:
            print("No best model found. Run comparative analysis first.")
            return None
        
        print("\n=== BEST MODEL SUMMARY ===")
        print(f"Model: {self.best_params['Model']}")
        print(f"Balancer: {self.best_params['Balancer']}")
        print(f"Feature Set: {self.best_params['FeatureSet']}")
        print(f"Validation MCC: {self.best_params['Val_MCC']:.4f}")
        print(f"Validation ROC-AUC: {self.best_params['Val_ROC_AUC']:.4f}")
        print(f"Best Parameters: {self.best_params['Best_Params']}")
        
        return self.best_params
    
    def bootstrap_confidence_interval(self, y_true, y_proba, metric_func, n_bootstrap=1000, alpha=0.05):
        """Calculate bootstrap confidence interval for a metric"""
        bootstrap_scores = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true.iloc[indices]
            y_proba_boot = y_proba[indices]
            
            # Calculate metric
            score = metric_func(y_true_boot, y_proba_boot)
            bootstrap_scores.append(score)
        
        # Calculate confidence interval
        lower_bound = np.percentile(bootstrap_scores, 100 * (alpha/2))
        upper_bound = np.percentile(bootstrap_scores, 100 * (1 - alpha/2))
        
        return lower_bound, upper_bound
    
    def final_evaluation(self, X_train, y_train, X_test, y_test, output_dir):
        """
        Perform final model evaluation on test set
        
        Parameters:
        X_train, y_train : Full training data for final training
        X_test, y_test : Test data for final evaluation
        output_dir : str, directory to save results
        """
        if self.best_model is None:
            print("No best model found. Run comparative analysis first.")
            return None
        
        print("\n=== FINAL MODEL EVALUATION ===")
        
        # Get the best feature set
        best_fs_name = self.best_params['FeatureSet']
        best_fs_data = self.feature_selectors[best_fs_name]
        
        # Apply feature selection to full training and test data
        if best_fs_data['selector'] is not None:
            X_train_final = pd.DataFrame(
                best_fs_data['selector'].transform(X_train), 
                columns=best_fs_data['features'],
                index=X_train.index
            )
            X_test_final = pd.DataFrame(
                best_fs_data['selector'].transform(X_test),
                columns=best_fs_data['features'], 
                index=X_test.index
            )
        else:
            # Manual feature selection
            X_train_final = X_train[best_fs_data['features']]
            X_test_final = X_test[best_fs_data['features']]
        
        # Retrain best model on full training data
        print("Retraining best model on full training data...")
        
        # Access the LogisticRegression model instance from the pipeline
        final_model = self.best_model.named_steps['classifier']
        
        # Re-initialize the model to ensure a clean slate, using only its own parameters
        final_model = final_model.__class__(**final_model.get_params())

        final_model.fit(X_train_final, y_train)
        
        # Make predictions on test set
        print("Evaluating on test set...")
        y_test_pred = final_model.predict(X_test_final)
        y_test_proba = final_model.predict_proba(X_test_final)[:, 1]
        
        # Calculate comprehensive metrics
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'mcc': matthews_corrcoef(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
        
        # Bootstrap confidence intervals for ROC-AUC
        print("Calculating bootstrap confidence intervals...")
        ci_lower, ci_upper = self.bootstrap_confidence_interval(
            y_test, y_test_proba, roc_auc_score, n_bootstrap=1000
        )
        test_metrics['roc_auc_ci_lower'] = ci_lower
        test_metrics['roc_auc_ci_upper'] = ci_upper
        
        # Calculate PPV and NPV
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        test_metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        test_metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        test_metrics['sensitivity'] = test_metrics['recall']  # Same as recall
        test_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Print results
        print("\n=== FINAL TEST SET PERFORMANCE ===")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {test_metrics['recall']:.4f}")
        print(f"Specificity: {test_metrics['specificity']:.4f}")
        print(f"F1-Score: {test_metrics['f1']:.4f}")
        print(f"Matthews Correlation Coefficient: {test_metrics['mcc']:.4f}")
        print(f"ROC-AUC: {test_metrics['roc_auc']:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
        print(f"Positive Predictive Value (PPV): {test_metrics['ppv']:.4f}")
        print(f"Negative Predictive Value (NPV): {test_metrics['npv']:.4f}")
        
        # Classification report
        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(y_test, y_test_pred))
        
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='blue', lw=2, 
                label=f'ROC curve (AUC = {test_metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Generate Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
        plt.subplot(1, 2, 2)
        plt.plot(recall_curve, precision_curve, color='red', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        curve_path = f"{output_dir}/roc_pr_curves.png"
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC and Precision-Recall curves saved to: {curve_path}")
        
        # SHAP interpretation (if available)
        if SHAP_AVAILABLE:
            print("\n=== SHAP MODEL INTERPRETATION ===")
            self.shap_interpretation(final_model, X_train_final, X_test_final, output_dir)
        
        return test_metrics, final_model
    
    def shap_interpretation(self, model, X_train, X_test, output_dir):
        """Generate SHAP interpretations"""
        try:
            print("Generating SHAP explanations...")
            
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X_train)
                shap_values = explainer(X_test)
                shap_values_plot = shap_values[:, :, 1]  # Focus on positive class
            else:
                explainer = shap.Explainer(model.predict, X_train)
                shap_values = explainer(X_test)
                shap_values_plot = shap_values
            
            # Global feature importance (summary plot)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X_test, show=False)
            summary_path = f"{output_dir}/shap_summary_plot.png"
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot saved to: {summary_path}")
            
            # Feature importance bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_plot, X_test, plot_type="bar", show=False)
            bar_path = f"{output_dir}/shap_feature_importance.png"
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP feature importance plot saved to: {bar_path}")
            
            # Individual prediction explanation (waterfall plot for first few samples)
            for i in range(min(3, len(X_test))):
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(shap_values_plot[i], show=False)
                waterfall_path = f"{output_dir}/shap_waterfall_sample_{i}.png"
                plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"SHAP waterfall plot for sample {i} saved to: {waterfall_path}")
            
        except Exception as e:
            print(f"Error generating SHAP interpretations: {str(e)}")
    
    def save_results(self, output_dir):
        """Save all results to files"""
        if hasattr(self, 'results_df'):
            results_path = f"{output_dir}/comparative_results.csv"
            self.results_df.to_csv(results_path, index=False)
            print(f"Comparative results saved to: {results_path}")
            
            # Save top 10 results
            top_results = self.results_df.sort_values('Val_MCC', ascending=False).head(10)
            top_path = f"{output_dir}/top_10_results.csv"
            top_results[['Model', 'Balancer', 'FeatureSet', 'Val_MCC', 'Val_ROC_AUC']].to_csv(top_path, index=False)
            print(f"Top 10 results saved to: {top_path}")

# Example usage (requires actual data):
"""
# Initialize framework
framework = MLComparativeFramework(random_state=42)

# Load your data (X_train, y_train, X_val, y_val, X_test, y_test)
# Make sure your data has the clinical features mentioned in the instructions
# Refer to data_processing.py

# Run comparative analysis
results_df = framework.run_comparative_analysis(X_train, y_train, X_val, y_val)

# Get best model summary
best_model = framework.get_best_model_summary()

# Final evaluation
test_metrics, final_model = framework.final_evaluation(
    X_train, y_train, X_test, y_test, output_dir="/path/to/output"
)

# Save results
framework.save_results(output_dir="/path/to/output")
"""

print("Optimized ML Comparative Framework created successfully!")

