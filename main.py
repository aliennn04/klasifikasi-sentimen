import pandas as pd
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# =====================================
# BACA DATASET
# =====================================
data = pd.read_csv("whatsapp.csv")

print("=" * 50)
print("DATASET BERHASIL DIBACA")
print("=" * 50)

print("Jumlah data awal :", len(data))

# Ambil 1700 data pertama
data = data.head(1700)

# Ambil kolom yang diperlukan
data = data[["text", "score"]]

# Hapus data kosong
data = data.dropna()

# Ubah text menjadi string
data["text"] = data["text"].astype(str)

# =====================================
# PREPROCESSING
# =====================================
def preprocessing(teks):
    
    # CASE FOLDING
    teks = teks.lower()

    # Hapus URL
    teks = re.sub(r"http\S+|www\S+|https\S+", "", teks)

    # Hapus mention
    teks = re.sub(r"@\w+", "", teks)

    # Hapus hashtag
    teks = re.sub(r"#\w+", "", teks)

    # Hapus emoji / stiker unicode
    teks = re.sub(
        r"[\U00010000-\U0010ffff]",
        "",
        teks,
        flags=re.UNICODE
    )

    # Hapus angka
    teks = re.sub(r"\d+", "", teks)

    # Hapus tanda baca
    teks = re.sub(r"[^\w\s]", " ", teks)

    # Hapus spasi berlebih
    teks = re.sub(r"\s+", " ", teks).strip()

    return teks

data["text"] = data["text"].apply(preprocessing)

# Hapus data kosong setelah cleaning
data = data[data["text"] != ""]

# Hapus ulasan terlalu pendek
data = data[data["text"].str.len() > 2]

# =====================================
# HAPUS DATA NETRAL (score = 3)
# =====================================
data = data[data["score"] != 3]

print("Jumlah data setelah preprocessing :", len(data))

# =====================================
# LABEL OTOMATIS (2 KELAS)
# =====================================
def label_sentimen(score):
    if score >= 4:
        return "positif"
    else:
        return "negatif"

data["label"] = data["score"].apply(label_sentimen)

print("\nDistribusi Label:")
print(data["label"].value_counts())

# =====================================
# FITUR DAN TARGET
# =====================================
X = data["text"]
y = data["label"]

# =====================================
# TF-IDF
# =====================================
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(X)

# =====================================
# SPLIT DATA
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# NAIVE BAYES
# =====================================
nb = MultinomialNB()

nb.fit(X_train, y_train)

pred_nb = nb.predict(X_test)

acc_nb = float(accuracy_score(y_test, pred_nb))
prec_nb = float(precision_score(y_test, pred_nb, average="weighted"))
rec_nb = float(recall_score(y_test, pred_nb, average="weighted"))
f1_nb = float(f1_score(y_test, pred_nb, average="weighted"))

print("\n")
print("=" * 50)
print("HASIL NAIVE BAYES")
print("=" * 50)

print(f"Accuracy : {acc_nb:.4f}")
print(f"Precision: {prec_nb:.4f}")
print(f"Recall   : {rec_nb:.4f}")
print(f"F1-Score : {f1_nb:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, pred_nb))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_nb))

# =====================================
# RANDOM FOREST
# =====================================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

acc_rf = float(accuracy_score(y_test, pred_rf))
prec_rf = float(precision_score(y_test, pred_rf, average="weighted"))
rec_rf = float(recall_score(y_test, pred_rf, average="weighted"))
f1_rf = float(f1_score(y_test, pred_rf, average="weighted"))

print("\n")
print("=" * 50)
print("HASIL RANDOM FOREST")
print("=" * 50)

print(f"Accuracy : {acc_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall   : {rec_rf:.4f}")
print(f"F1-Score : {f1_rf:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, pred_rf))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_rf))

# =====================================
# TABEL PERBANDINGAN
# =====================================
hasil = pd.DataFrame({
    "Metode": ["Naive Bayes", "Random Forest"],
    "Accuracy": [acc_nb, acc_rf],
    "Precision": [prec_nb, prec_rf],
    "Recall": [rec_nb, rec_rf],
    "F1-Score": [f1_nb, f1_rf]
})

print("\n")
print("=" * 50)
print("PERBANDINGAN HASIL")
print("=" * 50)
print(hasil)

# =====================================
# GRAFIK ACCURACY
# =====================================
model = ["Naive Bayes", "Random Forest"]
accuracy = [acc_nb, acc_rf]

plt.figure(figsize=(8, 5))

bars = plt.bar(model, accuracy)

plt.title("Perbandingan Accuracy Naive Bayes dan Random Forest")
plt.ylabel("Accuracy")
plt.xlabel("Metode")
plt.ylim(0, 1)

for bar in bars:
    yval = float(bar.get_height())

    plt.text(
        float(bar.get_x()) + float(bar.get_width()) / 2,
        yval + 0.01,
        f"{yval:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()

# =====================================
# SIMPAN HASIL PERBANDINGAN
# =====================================

hasil.to_csv(
    "hasil_perbandingan.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================
# WORD CLOUD SEMUA ULASAN
# =====================================

print("\n" + "=" * 50)
print("WORD CLOUD")
print("=" * 50)

text_semua = " ".join(data["text"].astype(str))

wc_semua = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=200
).generate(text_semua)

plt.figure(figsize=(12,6))
plt.imshow(wc_semua, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud Seluruh Ulasan WhatsApp")
plt.tight_layout()
plt.show()

wc_semua.to_file("wordcloud_semua.png")

# =====================================
# WORD CLOUD POSITIF
# =====================================

text_pos = " ".join(
    data[data["label"] == "positif"]["text"]
    .astype(str)
)

wc_pos = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=200
).generate(text_pos)

plt.figure(figsize=(12,6))
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud Sentimen Positif")
plt.tight_layout()
plt.show()

wc_pos.to_file("wordcloud_positif.png")

# =====================================
# WORD CLOUD NEGATIF
# =====================================

text_neg = " ".join(
    data[data["label"] == "negatif"]["text"]
    .astype(str)
)

wc_neg = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=200
).generate(text_neg)

plt.figure(figsize=(12,6))
plt.imshow(wc_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud Sentimen Negatif")
plt.tight_layout()
plt.show()

wc_neg.to_file("wordcloud_negatif.png")

print("\nWord Cloud berhasil disimpan:")
print("wordcloud_semua.png")
print("wordcloud_positif.png")
print("wordcloud_negatif.png")

# =====================================
# CONFUSION MATRIX VISUAL NB
# =====================================

cm_nb = confusion_matrix(y_test, pred_nb)

plt.figure(figsize=(6,5))
plt.imshow(cm_nb)

plt.title("Confusion Matrix Naive Bayes")
plt.colorbar()

plt.xticks([0,1], ["Negatif","Positif"])
plt.yticks([0,1], ["Negatif","Positif"])

for i in range(cm_nb.shape[0]):
    for j in range(cm_nb.shape[1]):
        plt.text(
            j,
            i,
            cm_nb[i,j],
            ha="center",
            va="center"
        )

plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.show()

# =====================================
# CONFUSION MATRIX VISUAL RF
# =====================================

cm_rf = confusion_matrix(y_test, pred_rf)

plt.figure(figsize=(6,5))
plt.imshow(cm_rf)

plt.title("Confusion Matrix Random Forest")
plt.colorbar()

plt.xticks([0,1], ["Negatif","Positif"])
plt.yticks([0,1], ["Negatif","Positif"])

for i in range(cm_rf.shape[0]):
    for j in range(cm_rf.shape[1]):
        plt.text(
            j,
            i,
            cm_rf[i,j],
            ha="center",
            va="center"
        )

plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.show()

# =====================================
# GRAFIK PERBANDINGAN SEMUA METRIK
# =====================================

hasil_plot = hasil.set_index("Metode")

hasil_plot.plot(
    kind="bar",
    figsize=(10,6)
)

plt.title(
    "Perbandingan Naive Bayes dan Random Forest"
)

plt.ylabel("Nilai")
plt.xlabel("Metode")

plt.ylim(0,1)

plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

print("TEST WORD CLOUD")