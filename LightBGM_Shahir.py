import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

def load_and_train_base_model(csv_path, target_col, model_name, save_path, exclude_cols=None):
    """
    Loads a dataset, trains a LightGBM model, and saves it.
    exclude_cols: List of extra columns to drop (to ensure compatibility with Meta Dataset).
    """
    print(f"\n--- Processing {model_name} ---")
    print(f"Loading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    # Prepare drop list
    drop_list = [target_col, 'id']
    if exclude_cols:
        drop_list.extend(exclude_cols)
        print(f"[{model_name}] Excluding columns to ensure compatibility: {exclude_cols}")

    # Create Feature Matrix X and Target Y
    X = df.drop(columns=drop_list, errors='ignore')
    Y = df[target_col]
    
    print(f"Training {model_name} on {len(df)} rows using {len(X.columns)} features...")
    print(f"Features used: {X.columns.tolist()}")
    
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X, Y)
    
    joblib.dump(model, save_path)
    print(f"Saved {model_name} to {save_path}")
    
    return model, X.columns.tolist()

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    # Using the filenames you uploaded
    path_lifestyle_data = r"Model A Dataset/lifestyle_dataset.csv"
    path_health_data    = r"Model B Dataset/healthFactors_dataset_with_indicator.csv"
    path_meta_data      = r"Meta Model Dataset/Training_For_Meta_Model.csv"
    
    save_dir = r"saved_models_tausif"
    metrics_file = r"meta_model_metrics.txt"
    target_col = 'Cardiovascular Disease'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ==========================================
    # 2. TRAIN BASE MODELS
    # ==========================================
    try:
        # --- Train Model A (Lifestyle) ---
        # No extra exclusions needed for Lifestyle
        model_A, features_A = load_and_train_base_model(
            path_lifestyle_data, target_col, "Model A (Lifestyle)", 
            os.path.join(save_dir, "lightgbm_model_A_new.pkl")
        )

        # --- Train Model B (Health) ---
        # IMPORTANT: 'Is Minority' is in Health Dataset but NOT in Meta Dataset.
        # We MUST exclude it, otherwise Model B will crash during the Meta step.
        model_B, features_B = load_and_train_base_model(
            path_health_data, target_col, "Model B (Health)", 
            os.path.join(save_dir, "lightgbm_model_B_new.pkl"),
            exclude_cols=['Is Minority'] 
        )
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed during base model training: {e}")
        return

    # ==========================================
    # 3. PREPARE META-MODEL DATASET
    # ==========================================
    print(f"\n--- Preparing Meta Model Dataset ---")
    print(f"Loading meta-training data from: {path_meta_data}")
    
    if not os.path.exists(path_meta_data):
        print(f"Error: Meta dataset not found at {path_meta_data}")
        return

    meta_df = pd.read_csv(path_meta_data)
    
    # NOTE: We REMOVED the renaming logic. 
    # Your files consistently use spaces (e.g. "Alcohol Intake"), so we keep them as is.

    # Split Meta Dataset
    X_meta_raw = meta_df.drop(columns=[target_col, 'id'], errors='ignore')
    Y_meta_raw = meta_df[target_col]

    X_train_split, X_test_split, Y_train, Y_test = train_test_split(
        X_meta_raw, Y_meta_raw, test_size=0.2, random_state=42, stratify=Y_meta_raw
    )

    # ==========================================
    # 4. GENERATE META-FEATURES
    # ==========================================
    print("Generating predictions from Base Models...")

    def get_preds_safe(model, feature_list, data):
        # Strictly select columns. If mismatch occurs, print helpful error.
        missing = [c for c in feature_list if c not in data.columns]
        if missing:
            raise ValueError(f"Meta Dataset is missing columns required by this model: {missing}")
        
        subset = data[feature_list]
        return model.predict_proba(subset)[:, 1]

    try:
        # Get predictions for Train Split
        probs_A_train = get_preds_safe(model_A, features_A, X_train_split)
        probs_B_train = get_preds_safe(model_B, features_B, X_train_split)
        X_meta_train = np.column_stack((probs_A_train, probs_B_train))

        # Get predictions for Test Split
        probs_A_test = get_preds_safe(model_A, features_A, X_test_split)
        probs_B_test = get_preds_safe(model_B, features_B, X_test_split)
        X_meta_test = np.column_stack((probs_A_test, probs_B_test))
        
    except ValueError as e:
        print(f"\n[ERROR] Column Mismatch detected: {e}")
        print("Please check that 'Training_For_Meta_Model.csv' contains all necessary columns.")
        return

    # ==========================================
    # 5. TRAIN META MODEL
    # ==========================================
    print("Training Meta Model (LightGBM)...")

    meta_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,        # Shallow tree since we only have 2 inputs
        num_leaves=7,
        random_state=42,
        verbosity=-1
    )

    meta_model.fit(X_meta_train, Y_train)
    joblib.dump(meta_model, os.path.join(save_dir, "meta_model_lgbm.pkl"))

    # ==========================================
    # 6. EVALUATION
    # ==========================================
    print("\n--- Final Evaluation ---")
    
    final_preds = meta_model.predict(X_meta_test)
    final_probs = meta_model.predict_proba(X_meta_test)[:, 1]

    # Metrics
    auc_A = roc_auc_score(Y_test, probs_A_test)
    auc_B = roc_auc_score(Y_test, probs_B_test)
    
    acc_meta = accuracy_score(Y_test, final_preds)
    roc_auc_meta = roc_auc_score(Y_test, final_probs)
    pr_auc_meta = average_precision_score(Y_test, final_probs)
    f1_meta = f1_score(Y_test, final_preds)
    prec_meta = precision_score(Y_test, final_preds)
    rec_meta = recall_score(Y_test, final_preds)

    # Console Output
    print(f"Base Model A (Lifestyle) ROC AUC: {auc_A:.4f}")
    print(f"Base Model B (Health)    ROC AUC: {auc_B:.4f}")
    print("-" * 40)
    print(f"Meta Model Accuracy:  {acc_meta:.4f}")
    print(f"Meta Model ROC AUC:   {roc_auc_meta:.4f}")
    print(f"Meta Model PR AUC:    {pr_auc_meta:.4f}")
    print(f"Meta Model F1-Score:  {f1_meta:.4f}")
    print(f"Meta Model Precision: {prec_meta:.4f}")
    print(f"Meta Model Recall:    {rec_meta:.4f}")

    # File Output
    try:
        with open(metrics_file, "w") as f:
            f.write("=== Meta Model Evaluation Metrics ===\n")
            f.write(f"Training Source A: {path_lifestyle_data}\n")
            f.write(f"Training Source B: {path_health_data}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Base Model A (Lifestyle) ROC AUC: {auc_A:.4f}\n")
            f.write(f"Base Model B (Health)    ROC AUC: {auc_B:.4f}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Meta Model Accuracy:  {acc_meta:.4f}\n")
            f.write(f"Meta Model ROC AUC:   {roc_auc_meta:.4f}\n")
            f.write(f"Meta Model PR AUC:    {pr_auc_meta:.4f}\n")
            f.write(f"Meta Model F1-Score:  {f1_meta:.4f}\n")
            f.write(f"Meta Model Precision: {prec_meta:.4f}\n")
            f.write(f"Meta Model Recall:    {rec_meta:.4f}\n")
        print(f"\n[Success] Metrics saved to '{metrics_file}'")
    except Exception as e:
        print(f"\n[Error] Could not save metrics file: {e}")

if __name__ == "__main__":
    main()