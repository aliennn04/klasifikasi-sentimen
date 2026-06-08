import pandas as pd
import matplotlib.pyplot as plt
import re

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