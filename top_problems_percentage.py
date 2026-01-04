import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict, Counter

# -----------------------------
# Setup
# -----------------------------
nltk.download("stopwords")

EN_STOP = set(stopwords.words("english"))
AR_STOP = set([
    "مش","مو","ما","في","على","من","عن","كان","كانت","جدا","جداً","مره","مرة"
])

translator = GoogleTranslator(source="auto", target="en")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(
    r"C:\Users\Kat\Downloads\talabat_restaurants_final.csv",
    encoding="utf-8"
)

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned_text"] = df["cleaned_text"].apply(clean_text)

# -----------------------------
# Translate Arabic → English
# -----------------------------
def translate_if_needed(text):
    try:
        if re.search(r"[\u0600-\u06FF]", text):
            return translator.translate(text)
        return text
    except:
        return text

df["en_text"] = df["cleaned_text"].apply(translate_if_needed)

# -----------------------------
# Split into sentences
# -----------------------------
def split_sentences(text):
    return [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 3]

df["sentences"] = df["en_text"].apply(split_sentences)

# -----------------------------
# Define semantic categories
# -----------------------------
CATEGORIES = {
    "Food Temperature": [
        "food was cold", "not hot", "arrived cold", "temperature bad"
    ],
    "Taste & Flavor": [
        "bad taste", "not tasty", "delicious", "flavorless"
    ],
    "Cooking Quality": [
        "undercooked", "raw", "burnt", "dry food", "not cooked well"
    ],
    "Missing / Wrong Items": [
        "missing item", "order incomplete", "wrong order"
    ],
    "Portion Size": [
        "small portion", "quantity too small"
    ],
    "Packaging": [
        "bad packaging", "spilled", "leaking"
    ],
    "Delivery Speed": [
        "late delivery", "fast delivery", "on time"
    ],
    "Service": [
        "bad service", "good service", "rude staff", "polite staff"
    ],
    "Price / Value": [
        "expensive", "not worth", "good price"
    ],
    "Cleanliness": [
        "dirty", "not clean", "clean restaurant"
    ]
}

# Encode category examples
category_embeddings = {}
for cat, examples in CATEGORIES.items():
    category_embeddings[cat] = model.encode(examples, convert_to_tensor=True)

# -----------------------------
# Assign sentence to category
# -----------------------------
def classify_sentence(sentence):
    sent_emb = model.encode(sentence, convert_to_tensor=True)
    best_cat = None
    best_score = 0

    for cat, emb in category_embeddings.items():
        score = util.cos_sim(sent_emb, emb).max().item()
        if score > best_score:
            best_score = score
            best_cat = cat

    return best_cat if best_score >= 0.55 else None

# -----------------------------
# Collect categories per restaurant
# -----------------------------
restaurant_results = {}

for restaurant, group in df.groupby("restaurant_name"):
    neg_counts = Counter()
    pos_counts = Counter()

    total_neg = 0
    total_pos = 0

    for _, row in group.iterrows():
        sentiment = row["final_sentiment"]
        for sent in row["sentences"]:
            category = classify_sentence(sent)
            if not category:
                continue

            if sentiment == "negative":
                neg_counts[category] += 1
                total_neg += 1
            elif sentiment == "positive":
                pos_counts[category] += 1
                total_pos += 1

    # Convert to percentages
    neg_percent = {
        k: round((v / total_neg) * 100, 1)
        for k, v in neg_counts.items()
    } if total_neg > 0 else {}

    pos_percent = {
        k: round((v / total_pos) * 100, 1)
        for k, v in pos_counts.items()
    } if total_pos > 0 else {}

    restaurant_results[restaurant] = {
        "top_problems": ", ".join(
            [f"{k} ({v}%)" for k, v in sorted(neg_percent.items(), key=lambda x: -x[1])[:5]]
        ),
        "top_positive_features": ", ".join(
            [f"{k} ({v}%)" for k, v in sorted(pos_percent.items(), key=lambda x: -x[1])[:5]]
        )
    }

# -----------------------------
# Save final results
# -----------------------------
final_df = pd.DataFrame.from_dict(restaurant_results, orient="index").reset_index()
final_df.rename(columns={"index": "restaurant_name"}, inplace=True)

final_df.to_csv(
    r"C:\Users\Kat\Downloads\talabat_restaurant_insights_percent.csv",
    index=False,
    encoding="utf-8"
)

print("Final NLP-based restaurant insights saved successfully")