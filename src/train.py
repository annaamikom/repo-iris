import pandas as pd
import joblib
import mlflow
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

def train():
    # 1. Setup MLflow
    mlflow.set_experiment("Eksperimen RF Iris 2")
    mlflow.sklearn.autolog()

    # 2. Load Data Training (Hasil dari stage prepare)
    if not os.path.exists('data/prepared/train.csv'):
        print(" Data training tidak ditemukan. Jalankan stage prepare dulu!")
        return

    train_df = pd.read_csv('data/prepared/train.csv')
    
    # Memisahkan fitur dan target
    # Asumsi kolom terakhir adalah target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    # 3. Membangun Scikit-Learn Pipeline (Konsep Ideal)
    # Kita bungkus semua transformasi agar "menikah" di dalam model.pkl
    
    # Gabungkan PCA dan KBest sebagai fitur tambahan
    combined_features = FeatureUnion([
        ('pca', PCA(n_components=2)),
        ('kbest', SelectKBest(chi2, k=3))
    ])

    # Pipeline utama
    pipeline = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('features', combined_features),
        ('forest', RandomForestClassifier(n_estimators=7, random_state=42))
    ])

    # 4. Proses Pelatihan
    print(" Memulai pelatihan model...")
    with mlflow.start_run(run_name="DVC_Train_Stage"):
        pipeline.fit(X_train, y_train)
        
        # 5. Simpan Hasil Akhir
        os.makedirs('models', exist_ok=True)
        joblib.dump(pipeline, 'models/model.pkl')
        
    print("Model (bersama Scaler, PCA, & KBest) berhasil disimpan di models/model.pkl")

if __name__ == "__main__":
    train()