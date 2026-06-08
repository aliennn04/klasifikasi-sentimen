import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =========================
# MEMBACA DATASET
# =========================
df = pd.read_csv("whatsapp.csv")

# Hapus data kosong
df = df.dropna(subset=['content'])

# =========================
# 1. CASE FOLDING
# =========================
df['content'] = df['content'].astype(str).str.lower()

# =========================
# 2. CLEANING
# =========================
def cleaning(text):
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Hapus mention dan hashtag
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    # Hapus angka
    text = re.sub(r'\d+', '', text)

    # Hapus tanda baca dan karakter khusus
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df['content'] = df['content'].apply(cleaning)

# =========================
# 3. TOKENIZING
# =========================
def tokenizing(text):
    return text.split()

df['tokens'] = df['content'].apply(tokenizing)

# =========================
# 4. STOPWORD REMOVAL
# =========================
factory_stopword = StopWordRemoverFactory()
stopwords = set(factory_stopword.get_stop_words())

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

df['tokens'] = df['tokens'].apply(remove_stopwords)

# =========================
# 5. STEMMING
# =========================
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

df['tokens'] = df['tokens'].apply(stemming)

# Gabungkan token menjadi kalimat kembali
df['content_bersih'] = df['tokens'].apply(lambda x: ' '.join(x))

# =========================
# KAMUS SENTIMEN
# =========================
kata_positif = [
    'bagus', 'baik', 'mantap', 'cepat', 'mudah',
    'bantu', 'suka', 'keren', 'baik',
    'puas', 'guna', 'hebat', 'terbaik'
]

kata_negatif = [
    'buruk', 'jelek', 'error', 'lambat', 'gagal',
    'bug', 'rusak', 'kecewa', 'parah',
    'lemot', 'ganggu', 'masalah'
]

# =========================
# PELABELAN SENTIMEN
# =========================
def label_sentimen(text):
    positif = 0
    negatif = 0

    words = text.split()

    for word in words:
        if word in kata_positif:
            positif += 1

        if word in kata_negatif:
            negatif += 1

    if positif > negatif:
        return "Positif"
    elif negatif > positif:
        return "Negatif"
    else:
        return "Netral"

df['sentimen'] = df['content_bersih'].apply(label_sentimen)

# Hapus data netral jika hanya ingin 2 kelas
df = df[df['sentimen'] != 'Netral']

# =========================
# SIMPAN HASIL
# =========================
df.to_csv("dataset_berlabel.csv", index=False)

# =========================
# HASIL
# =========================
print("\nDistribusi Sentimen:")
print(df['sentimen'].value_counts())

print("\nContoh Hasil:")
print(df[['content', 'content_bersih', 'sentimen']].head())