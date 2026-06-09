import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =====================================
# BACA DATASET HASIL PREPROCESSING
# =====================================

df = pd.read_csv("whatsapp.csv")

print(df["label"].value_counts())

# =====================================
# WORD CLOUD POSITIF
# =====================================

text_pos = " ".join(
    df[df["label"] == "positif"]["text"]
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
plt.show()

# =====================================
# SIMPAN
# =====================================

wc_pos.to_file("wordcloud_positif.png")

# =====================================
# WORD CLOUD NEGATIF
# =====================================

text_neg = " ".join(
    df[df["label"] == "negatif"]["text"]
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
plt.show()

wc_neg.to_file("wordcloud_negatif.png")

print("\nWord Cloud berhasil disimpan")