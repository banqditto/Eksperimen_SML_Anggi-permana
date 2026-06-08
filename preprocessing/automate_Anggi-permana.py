import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_raw_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File mentah tidak ditemukan di {file_path}")
        sys.exit(1)
        
    print(f"Memuat data dari: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df, output_dir):
    print("Memulai otomatisasi preprocessing data...")
    
    imputer = SimpleImputer(strategy='median')
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_clean = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_clean['Outcome'] = y_train.values
    
    test_clean = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_clean['Outcome'] = y_test.values
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train_clean.csv')
    test_path = os.path.join(output_dir, 'test_clean.csv')
    
    train_clean.to_csv(train_path, index=False)
    test_clean.to_csv(test_path, index=False)
    
    print(f"Otomatisasi sukses! Berkas disimpan di:\n - {train_path}\n - {test_path}")

if __name__ == "__main__":
    RAW_DATA_PATH = "diabetes_raw/diabetes.csv"
    OUTPUT_PREPROCESSING_DIR = "preprocessing/diabetes_preprocessing"
    
    raw_df = load_raw_data(RAW_DATA_PATH)
    preprocess_data(raw_df, OUTPUT_PREPROCESSING_DIR)
