import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import sys

def prepare(input_path):
    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"File {input_path} tidak ditemukan!")
        return

    df = pd.read_csv(input_path)
    
    # 2. Cleaning Ringan Menghapus: kolom Id 
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # 3. X-y Splitting (Feature & Target)
    #  kolom terakhir adalah target (Species)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 4. Visualisasi (Plotting)
    # Kita simpan plot ke folder 'plots' 
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue=df.columns[-1])
    plt.title('Visualisasi Fitur Iris')
    plt.savefig('plots/data_distribution.png')
    print(" Plot distribusi data disimpan di plots/data_distribution.png")

    # 5. Simpan Data Split
    # Kita simpan hasil split agar tahap 'train' dan 'evaluate' konsisten
    os.makedirs('data/prepared', exist_ok=True)
    
    # Membagi data menjadi train dan test
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=y)
    
    train_df.to_csv('data/prepared/train.csv', index=False)
    test_df.to_csv('data/prepared/test.csv', index=False)
    
    print(f"Data berhasil di-split dan disimpan di folder data/prepared/")

if __name__ == "__main__":
    # Input path diambil dari argumen pertama (misal: data/Iris.csv)
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/Iris.csv'
    prepare(path)