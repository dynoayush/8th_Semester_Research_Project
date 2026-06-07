import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df2 = pd.read_csv('Data_Required/Diseases_and_Symptoms_dataset.csv')

print("Dropping Non-Feature Columns...")
df2 = df2.drop(columns=['Medications', 'Recommendations'], errors='ignore')

print("Renaming Target Column...")
df2 = df2.rename(columns={'disease': 'prognosis', 'Disease': 'prognosis', 'diseases': 'prognosis'}, errors='ignore')

print("Normalizing Binary Symptom Columns...")
symptom_cols = [c for c in df2.columns if c != 'prognosis']
df2[symptom_cols] = df2[symptom_cols].apply(
    lambda col: col.map(lambda x: 1 if str(x).strip().lower() in ['1', '1.0', 'yes', 'true'] else 0)
)

print("Handling Missing Values...")
df2[symptom_cols] = df2[symptom_cols].fillna(0)

# REMOVED STEP 4.5 (Disease Grouping) 
# This ensures the model outputs the EXACT disease name so it matches the CSVs!

print("Label Encoding Target...")
le2 = LabelEncoder()
df2['prognosis'] = le2.fit_transform(df2['prognosis'])

X = df2.drop(columns=['prognosis'])
y = df2['prognosis']

print("Training XGBoost...")
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
xgb_model.fit(X, y)

print("Saving models...")
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(le2, 'label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'symptoms_list.pkl')
print("Done! Saved xgb_model.pkl, label_encoder.pkl, symptoms_list.pkl")
