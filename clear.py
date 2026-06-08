import pandas as pd

# Membaca dataset
df = pd.read_csv("whatsapp.csv")

# Hapus data kosong
df = df.dropna(subset=['content'])
df['content'] = df['content'].astype(str).str.lower()

# Daftar kata positif
kata_positif = [
    'bagus', 'baik', 'mantap', 'cepat', 'mudah',
    'membantu', 'suka', 'keren', 'terbaik',
    'memuaskan', 'berguna', 'hebat'
]

# Daftar kata negatif
kata_negatif = [
    'buruk', 'jelek', 'error', 'lambat', 'gagal',
    'bug', 'rusak', 'kecewa', 'parah',
    'lemot', 'gangguan', 'masalah'
]

def label_sentimen(teks):
    positif = 0
    negatif = 0

    kata = teks.split()

    for k in kata:
        if k in kata_positif:
            positif += 1
        if k in kata_negatif:
            negatif += 1

    if positif > negatif:
        return "Positif"
    elif negatif > positif:
        return "Negatif"
    else:
        return "Netral"

# Pelabelan
df['sentimen'] = df['content'].apply(label_sentimen)

# Jika hanya ingin Positif dan Negatif
df = df[df['sentimen'] != 'Netral']

# Simpan hasil
df.to_csv("dataset_berlabel.csv", index=False)

print(df['sentimen'].value_counts())
print(df[['content', 'sentimen']].head())