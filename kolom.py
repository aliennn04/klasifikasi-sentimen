import pandas as pd

# Baca file CSV
data = pd.read_csv("whatsapp.csv")

# Tampilkan semua nama kolom
print("=" * 50)
print("DAFTAR KOLOM")
print("=" * 50)
print(data.columns.tolist())

# Tampilkan informasi dataset
print("\n" + "=" * 50)
print("INFO DATASET")
print("=" * 50)
print(data.info())

# Tampilkan 5 baris pertama
print("\n" + "=" * 50)
print("5 BARIS PERTAMA")
print("=" * 50)
print(data.head())

# Tampilkan isi setiap kolom (5 data pertama)
print("\n" + "=" * 50)
print("CONTOH ISI TIAP KOLOM")
print("=" * 50)

for kolom in data.columns:
    print(f"\nKolom: {kolom}")
    print(data[kolom].head())