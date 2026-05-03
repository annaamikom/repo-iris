import pandas as pd
import joblib
import json
import mlflow
import os
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    # 1. Setup MLflow (Gunakan eksperimen yang sama dengan train.py)
    mlflow.set_experiment("Eksperimen RF Iris")

    # 2. Load Data Test & Model
    if not os.path.exists('data/prepared/test.csv') or not os.path.exists('models/model.pkl'):
        print(" Data test atau Model tidak ditemukan!")
        return

    test_df = pd.read_csv('data/prepared/test.csv')
    model = joblib.load('models/model.pkl')

    # Memisahkan fitur dan target
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # 3. Prediksi
    print("Mengevaluasi model pada data uji...")
    y_pred = model.predict(X_test)
    
    # 4. Hitung Metrik
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']
    
    # 5. Log ke MLflow
    # Kita tidak menggunakan autolog di sini agar bisa mencatat metrik custom secara manual
    with mlflow.start_run(run_name="DVC_Eval_Stage"):
        mlflow.log_metric("test_accuracy", accuracy)
        # Anda bisa menambah log lain seperti f1-score macro
        mlflow.log_metric("f1_macro", report['macro avg']['f1-score'])
        
        print(f"Accuracy: {accuracy:.4f}")

    # 6. Simpan ke JSON untuk DVC Metrics
    # DVC akan menggunakan file ini untuk membandingkan performa antar eksperimen
    metrics = {
        "accuracy": accuracy,
        "f1_score": report['macro avg']['f1-score'],
        "precision": report['macro avg']['precision']
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Metrics berhasil disimpan di metrics.json")

if __name__ == "__main__":
    evaluate()