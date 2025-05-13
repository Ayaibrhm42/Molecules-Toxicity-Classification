import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
warnings.filterwarnings('ignore')


class MoleculeToxicityClassifier:
    
    def __init__(self):
        #Initialize the classifier with required components.
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tabnet_model = None
        self.feature_names = None
        self.class_weights = None
        
    def engineer_toxicity_features(self, df):
        
        #Create domain-specific feature combinations for toxicity prediction.
        #Args:
            #df: DataFrame with original molecular descriptors
        #Returns:
            #DataFrame with engineered features added
        
        df_eng = df.copy()
        
        print("Creating chemical feature combinations...")
        
        # 1. Electronic Reactivity Score
        if all(col in df.columns for col in ['MATS6p', 'SpMax6_Bhi', 'SaasC']):
            df_eng['electronic_reactivity'] = (
                df['MATS6p'] * 0.4 +
                df['SpMax6_Bhi'] * 0.3 +
                df['SaasC'] * 0.3
            )
            print("- Electronic reactivity score created")
        
        # 2. Mass Distribution Score
        if all(col in df.columns for col in ['ATS6m', 'ATS0m', 'GATS3v']):
            df_eng['mass_distribution_score'] = (
                df['ATS6m'] * 0.5 +
                df['ATS0m'] * 0.3 +
                df['GATS3v'] * 0.2
            )
            print("- Mass distribution score created")
        
        # 3. Bioavailability Indicator
        if all(col in df.columns for col in ['MLFER_S', 'maxHCsats', 'nHeteroRing']):
            df_eng['bioavailability_score'] = (
                df['MLFER_S'] * 0.4 +
                df['maxHCsats'] * 0.3 +
                df['nHeteroRing'] * 0.3
            )
            print("- Bioavailability score created")
        
        # 4. Structural Complexity Index
        if all(col in df.columns for col in ['BIC4', 'nHeteroRing', 'GATS3v']):
            df_eng['structural_complexity'] = (
                df['BIC4'] * 0.5 +
                df['nHeteroRing'] * 0.3 +
                df['GATS3v'] * 0.2
            )
            print("- Structural complexity index created")
        
        # 5. Interaction Potential
        if all(col in df.columns for col in ['MATS6p', 'MLFER_S']):
            df_eng['interaction_potential'] = df['MATS6p'] * df['MLFER_S']
            print("- Interaction potential created")
        
        # 6. Heavy Atom Effect
        if all(col in df.columns for col in ['ATS6m', 'SpMax6_Bhi']):
            df_eng['heavy_atom_effect'] = df['ATS6m'] * df['SpMax6_Bhi']
            print("- Heavy atom effect created")
        
        # 7. Normalized Charge Distribution
        if all(col in df.columns for col in ['SaasC', 'ATS0m']):
            df_eng['charge_density'] = df['SaasC'] / (df['ATS0m'] + 1e-10)
            print("- Charge density created")
        
        # 8. Ring System Complexity
        if all(col in df.columns for col in ['nHeteroRing', 'BIC4']):
            df_eng['ring_complexity'] = df['nHeteroRing'] * df['BIC4']
            print("- Ring complexity created")
        
        # 9. Toxicity Risk Score
        if all(col in df.columns for col in ['MATS6p', 'MLFER_S', 'ATS6m', 'nHeteroRing', 'SaasC', 'SpMax6_Bhi']):
            df_eng['toxicity_risk_score'] = (
                df['MATS6p'] * 0.25 +
                df['MLFER_S'] * 0.15 +
                df['ATS6m'] * 0.15 +
                df['nHeteroRing'] * 0.15 +
                df['SaasC'] * 0.15 +
                df['SpMax6_Bhi'] * 0.15
            )
            print("- Toxicity risk score created")
        
        # 10. Create log transforms for skewed features
        skewed_features = ['ATS6m', 'ATS0m', 'BIC4']
        for feature in skewed_features:
            if feature in df.columns:
                df_eng[f'log_{feature}'] = np.log1p(df[feature])
        print("- Log transformations applied")
        
        # 11. Create polynomial features for top predictors
        top_features = ['MATS6p', 'MLFER_S', 'ATS6m', 'nHeteroRing']
        for feature in top_features:
            if feature in df.columns:
                df_eng[f'{feature}_squared'] = df[feature] ** 2
        print("- Polynomial features created")
        
        # 12. Create more interaction terms
        if all(col in df.columns for col in ['MATS6p', 'MLFER_S']):
            df_eng['mats_mlfer_interaction'] = df['MATS6p'] * df['MLFER_S']
            print("- Additional interaction terms created")
        
        new_features = len(df_eng.columns) - len(df.columns)
        print(f"\nTotal new features created: {new_features}")
        
        return df_eng
    
    def load_and_explore_data(self, filepath):
        
        self.df = pd.read_csv(filepath)
        print(f"Dataset shape: {self.df.shape}")
        print("\nClass distribution:")
        print(self.df['Class'].value_counts())
        print(f"\nClass percentages:")
        print(self.df['Class'].value_counts(normalize=True) * 100)
        return self.df
    
    def evaluate_model(self, X, y, show_plots=True, set_name="Validation"):
    
        print("\n" + "="*50)
        print(f"{set_name.upper()} SET EVALUATION")
        print("="*50)
        
        # Get predictions and probabilities
        y_prob = self.tabnet_model.predict_proba(X)
        y_pred = self.tabnet_model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        auc_roc = roc_auc_score(y, y_prob[:, 1]) if len(np.unique(y)) == 2 else None
        
        print(f"{set_name} Set Size: {len(y)} samples")
        print(f"{set_name} Accuracy: {accuracy:.4f}")
        print(f"{set_name} F1-Score: {f1:.4f}")
        if auc_roc:
            print(f"{set_name} AUC-ROC: {auc_roc:.4f}")
        
        if show_plots and auc_roc:
            # ROC curve
            plt.figure(figsize=(10, 8))
            fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc_roc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {set_name} Set')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # Confusion Matrix
            cm = confusion_matrix(y, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title(f'Confusion Matrix - {set_name} Set')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'auc_roc': auc_roc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def analyze_feature_importance(self, plot=True):
       
        if not hasattr(self.tabnet_model, 'feature_importances_'):
            raise ValueError("Model not trained. Train the model first.")
            
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.tabnet_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if plot:
            # Plot top 15 features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Most Important Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            # Print top 10 features
            print("\nTop 10 Most Important Features:")
            for i, row in importance_df.head(10).iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
                
        return importance_df


def run_toxicity_classification(data_filepath, save_model=True):
   
    print("Starting Molecule Toxicity Classification")
    print("=" * 70)
    
    # Initialize classifier
    classifier = MoleculeToxicityClassifier()
    
    # Load data
    df = classifier.load_and_explore_data(data_filepath)
    
    # Apply feature engineering
    print("\nStep 1: Feature Engineering")
    df_engineered = classifier.engineer_toxicity_features(df)
    
    # Preprocess
    X = df_engineered.drop('Class', axis=1)
    y = df_engineered['Class']
    classifier.feature_names = X.columns.tolist()
    y_encoded = classifier.label_encoder.fit_transform(y)
    
    print(f"\nTotal features after engineering: {len(classifier.feature_names)}")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    classifier.class_weights = {i: weight for i, weight in enumerate(class_weights)}
    
    # Scale features
    X_scaled = classifier.scaler.fit_transform(X)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    print(f"\nData split:")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # Model configuration with optimal hyperparameters
    print("\nStep 2: Creating Model with Optimal Parameters")
    # These are the optimal hyperparameters found during experimentation
    optimizer_fn = torch.optim.Adam
    optimizer_params = dict(lr=0.02, weight_decay=0.001)
    scheduler_fn = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_params = dict(T_max=300)
    
    classifier.tabnet_model = TabNetClassifier(
        n_a=20,
        n_d=18, 
        n_steps=9, 
        gamma=1.6, 
        lambda_sparse=1e-05,
        optimizer_fn=optimizer_fn,
        optimizer_params=optimizer_params,
        scheduler_fn=scheduler_fn,
        scheduler_params=scheduler_params,
        mask_type='entmax',
        seed=42,
        verbose=1
    )
    
    # Train model
    print("\nStep 3: Training Model")
    sample_weights = np.array([classifier.class_weights[label] for label in y_train])
    
    classifier.tabnet_model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_name=['train', 'valid'],
        max_epochs=300,
        patience=50,
        batch_size=16,
        virtual_batch_size=16,
        weights=sample_weights,
        drop_last=False
    )
    
    # Evaluate on validation set
    print("\nStep 4: Validation Evaluation")
    val_results = classifier.evaluate_model(X_val, y_val, set_name="Validation")
    
    # Final test evaluation
    print("\nStep 5: Final Test Evaluation")
    test_results = classifier.evaluate_model(X_test, y_test, set_name="Test")
    
    # Analyze feature importance
    print("\nStep 6: Feature Importance Analysis")
    importance_df = classifier.analyze_feature_importance(plot=True)
    
    # Save model if requested
    if save_model:
        try:
            classifier.tabnet_model.save_model('molecule_toxicity_model.zip')
            print("\nModel saved as 'molecule_toxicity_model.zip'")
        except Exception as e:
            print(f"\nWarning: Could not save model: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Validation AUC-ROC: {val_results['auc_roc']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test AUC-ROC: {test_results['auc_roc']:.4f}")
    
    return classifier, test_results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run the classification
    classifier, results = run_toxicity_classification('Molecules_Toxicity_Classificationin.csv')
