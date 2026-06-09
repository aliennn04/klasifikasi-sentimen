import pandas as pd
import re
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =====================================================
# 1. MEMBACA DATASET
# =====================================================

df = pd.read_csv("whatsapp.csv")

# Sesuaikan nama kolom
# Jika dataset Anda menggunakan kolom text
df = df.dropna(subset=['text'])

# =====================================================
# 2. MEMBACA KAMUS NORMALISASI
# =====================================================

with open(
    "combined_slang_words.txt",
    "r",
    encoding="utf-8"
) as f:

    normalisasi_dict = json.load(f)

print("Jumlah kata slang :", len(normalisasi_dict))

# =====================================================
# 3. MEMBACA STOPWORD
# =====================================================

with open(
    "combined_stop_words.txt",
    "r",
    encoding="utf-8"
) as f:

    stopwords = set(
        line.strip()
        for line in f
        if line.strip()
    )

print("Jumlah stopword :", len(stopwords))

# =====================================================
# 4. MEMBACA LEXICON INSET
# =====================================================

positive = pd.read_csv(
    "positive.tsv",
    sep="\t"
)

negative = pd.read_csv(
    "negative.tsv",
    sep="\t"
)

lexicon = pd.concat(
    [positive, negative],
    ignore_index=True
)

lexicon_dict = dict(
    zip(
        lexicon['word'],
        lexicon['weight']
    )
)

print("Jumlah kata positif :", len(positive))
print("Jumlah kata negatif :", len(negative))

# =====================================================
# 5. CASE FOLDING
# =====================================================

df['case_folding'] = (
    df['text']
    .astype(str)
    .str.lower()
)

# =====================================================
# 6. CLEANING
# =====================================================

def cleaning(text):

    # hapus URL
    text = re.sub(
        r'http\S+|www\S+|https\S+',
        '',
        text
    )

    # hapus mention
    text = re.sub(
        r'@\w+',
        '',
        text
    )

    # hapus hashtag
    text = re.sub(
        r'#\w+',
        '',
        text
    )

    # hapus angka
    text = re.sub(
        r'\d+',
        '',
        text
    )

    # hapus simbol
    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text
    )

    # hapus spasi berlebih
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text

df['cleaning'] = (
    df['case_folding']
    .apply(cleaning)
)

# =====================================================
# 7. TOKENIZING
# =====================================================

def tokenizing(text):
    return text.split()

df['tokenizing'] = (
    df['cleaning']
    .apply(tokenizing)
)

# =====================================================
# 8. NORMALISASI
# =====================================================

def normalize(tokens):

    return [
        normalisasi_dict.get(
            word,
            word
        )
        for word in tokens
    ]

df['normalisasi'] = (
    df['tokenizing']
    .apply(normalize)
)

# =====================================================
# 9. STOPWORD REMOVAL
# =====================================================

def remove_stopword(tokens):

    return [
        word
        for word in tokens
        if word not in stopwords
    ]

df['stopword'] = (
    df['normalisasi']
    .apply(remove_stopword)
)

# =====================================================
# 10. STEMMING
# =====================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(tokens):

    return [
        stemmer.stem(word)
        for word in tokens
    ]

df['stemming'] = (
    df['stopword']
    .apply(stemming)
)

# =====================================================
# 11. GABUNGKAN KATA
# =====================================================

df['content_bersih'] = (
    df['stemming']
    .apply(lambda x: ' '.join(x))
)

# =====================================================
# 12. HITUNG SKOR SENTIMEN
# =====================================================

def sentiment_score(text):

    score = 0

    words = text.split()

    for word in words:

        if word in lexicon_dict:
            score += lexicon_dict[word]

    return score

df['score_sentimen'] = (
    df['content_bersih']
    .apply(sentiment_score)
)

# =====================================================
# 13. LABEL SENTIMEN
# =====================================================

def sentiment_label(score):

    if score > 0:
        return "Positif"

    elif score < 0:
        return "Negatif"

    else:
        return "Netral"

df['sentimen'] = (
    df['score_sentimen']
    .apply(sentiment_label)
)

# =====================================================
# 14. DATASET 2 KELAS
# =====================================================

df_2kelas = df[
    df['sentimen'] != 'Netral'
]

# =====================================================
# 15. SIMPAN DATASET
# =====================================================

df.to_csv(
    "dataset_label_lengkap.csv",
    index=False,
    encoding="utf-8-sig"
)

df_2kelas.to_csv(
    "dataset_label_2kelas.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 16. DISTRIBUSI SENTIMEN
# =====================================================

print("\nDistribusi Sentimen:")

print(
    df['sentimen']
    .value_counts()
)

print("\nDistribusi Sentimen 2 Kelas:")

print(
    df_2kelas['sentimen']
    .value_counts()
)

# =====================================================
# 17. CONTOH HASIL
# =====================================================

print("\nContoh Hasil:")

print(
    df[
        [
            'text',
            'content_bersih',
            'score_sentimen',
            'sentimen'
        ]
    ]
    .head(10)
)

# =====================================================
# 18. CEK NORMALISASI
# =====================================================

print("\nContoh Normalisasi:")

for i in range(min(10, len(df))):

    print("\nAsli       :", df['cleaning'].iloc[i])

    print(
        "Normalisasi:",
        " ".join(
            df['normalisasi'].iloc[i]
        )
    )