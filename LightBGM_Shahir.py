import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
# Added average_precision_score for PR AUC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

def main():
    # ==========================================
    # 1. SETUP & DATA LOADING
    # ==========================================
    dataset_path = r"Meta Model Dataset/Training_For_Meta_Model.csv"
    save_dir = r"saved_models_tausif"
    metrics_file = r"meta_model_metrics.txt"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {dataset_path}")
        return

    # Define Column Groups
    target_col = 'Cardiovascular Disease'
    drop_for_A = ['Cholesterol_Level', 'Diastolic_Blood_Pressure', 'Systolic_Blood_Pressure', 'Glucose_Level', 'id', target_col]
    drop_for_B = ['Smoking_Status', 'Physical_Activity', 'Alcohol_Intake', 'id', target_col]

    X = dataset.drop([target_col], axis=1)
    Y = dataset[target_col]

    column_mappings = {
        'Cholesterol Level': 'Cholesterol_Level',
        'Diastolic Blood Pressure': 'Diastolic_Blood_Pressure',
        'Glucose Level': 'Glucose_Level',
        'Systolic Blood Pressure': 'Systolic_Blood_Pressure',
        'Alcohol Intake': 'Alcohol_Intake',
        'Physical Activity': 'Physical_Activity',
        'Smoking Status': 'Smoking_Status'
    }
    X = X.rename(columns=column_mappings)

    # Split Data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # ==========================================
    # 2. RETRAIN BASE MODELS
    # ==========================================
    print("\n--- Retraining Base Models ---")

    # --- Train Model A (Lifestyle) ---
    print("Training Model A (Lifestyle)...")
    X_train_A = X_train.drop(columns=[c for c in drop_for_A if c in X_train.columns], errors='ignore')
    X_test_A  = X_test.drop(columns=[c for c in drop_for_A if c in X_test.columns], errors='ignore')
    
    model_A = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    model_A.fit(X_train_A, Y_train)
    joblib.dump(model_A, os.path.join(save_dir, "lightgbm_model_A_new.pkl"))

    # --- Train Model B (Health) ---
    print("Training Model B (Health)...")
    X_train_B = X_train.drop(columns=[c for c in drop_for_B if c in X_train.columns], errors='ignore')
    X_test_B  = X_test.drop(columns=[c for c in drop_for_B if c in X_test.columns], errors='ignore')
    
    model_B = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    model_B.fit(X_train_B, Y_train)
    joblib.dump(model_B, os.path.join(save_dir, "lightgbm_model_B_new.pkl"))

    # ==========================================
    # 3. GENERATE META-FEATURES
    # ==========================================
    print("\n--- Generating Meta-Features ---")

    probs_A_train = model_A.predict_proba(X_train_A)[:, 1]
    probs_B_train = model_B.predict_proba(X_train_B)[:, 1]

    probs_A_test = model_A.predict_proba(X_test_A)[:, 1]
    probs_B_test = model_B.predict_proba(X_test_B)[:, 1]

    X_meta_train = np.column_stack((probs_A_train, probs_B_train))
    X_meta_test  = np.column_stack((probs_A_test, probs_B_test))

    # ==========================================
    # 4. TRAIN META MODEL (LightGBM)
    # ==========================================
    print("Training Meta Model (LightGBM)...")

    meta_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        random_state=42,
        verbosity=-1
    )

    meta_model.fit(X_meta_train, Y_train)
    joblib.dump(meta_model, os.path.join(save_dir, "meta_model_lgbm.pkl"))

    # ==========================================
    # 5. FINAL EVALUATION & LOGGING
    # ==========================================
    print("\n--- Final Evaluation on Test Set ---")
    
    final_preds = meta_model.predict(X_meta_test)
    final_probs = meta_model.predict_proba(X_meta_test)[:, 1]

    # Calculate All Metrics
    auc_A = roc_auc_score(Y_test, probs_A_test)
    auc_B = roc_auc_score(Y_test, probs_B_test)
    
    acc_meta = accuracy_score(Y_test, final_preds)
    roc_auc_meta = roc_auc_score(Y_test, final_probs)
    pr_auc_meta = average_precision_score(Y_test, final_probs) # PR AUC
    f1_meta = f1_score(Y_test, final_preds)
    prec_meta = precision_score(Y_test, final_preds)
    rec_meta = recall_score(Y_test, final_preds)

    # Print to Console
    print(f"Base Model A (Lifestyle) ROC AUC: {auc_A:.4f}")
    print(f"Base Model B (Health)    ROC AUC: {auc_B:.4f}")
    print("-" * 40)
    print(f"Meta Model Accuracy:  {acc_meta:.4f}")
    print(f"Meta Model ROC AUC:   {roc_auc_meta:.4f}")
    print(f"Meta Model PR AUC:    {pr_auc_meta:.4f}")
    print(f"Meta Model F1-Score:  {f1_meta:.4f}")
    print(f"Meta Model Precision: {prec_meta:.4f}")
    print(f"Meta Model Recall:    {rec_meta:.4f}")

    # Write to Text File
    try:
        with open(metrics_file, "w") as f:
            f.write("=== Meta Model Evaluation Metrics ===\n")
            f.write(f"Base Model A (Lifestyle) ROC AUC: {auc_A:.4f}\n")
            f.write(f"Base Model B (Health)    ROC AUC: {auc_B:.4f}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Meta Model Accuracy:  {acc_meta:.4f}\n")
            f.write(f"Meta Model ROC AUC:   {roc_auc_meta:.4f}\n")
            f.write(f"Meta Model PR AUC:    {pr_auc_meta:.4f}\n")
            f.write(f"Meta Model F1-Score:  {f1_meta:.4f}\n")
            f.write(f"Meta Model Precision: {prec_meta:.4f}\n")
            f.write(f"Meta Model Recall:    {rec_meta:.4f}\n")
        print(f"\n[Success] All metrics (Accuracy, ROC AUC, PR AUC, F1, Precision, Recall) saved to '{metrics_file}'")
    except Exception as e:
        print(f"\n[Error] Could not save metrics to file: {e}")

if __name__ == "__main__":
    main()