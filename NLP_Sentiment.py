import pandas as pd
import re
import torch
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
 
# ------------------------
# 1. Load data
# ------------------------
df = pd.read_csv(r"C:\Users\Kat\Downloads\talabat_15_restaurants_reviews.csv", encoding='utf-8')
 
# ------------------------
# 2. Preprocessing
# ------------------------
def preprocess(text):
    text = str(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text
 
df['cleaned_text'] = df['review_text'].apply(preprocess)
 
# ------------------------
# 3. Load model
# ------------------------
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
model.eval()
 
labels = ['1_star', '2_star', '3_star', '4_star', '5_star']
 
# ------------------------
# 4. Prediction function
# ------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = np.argmax(probs)
    stars = idx + 1  # Convert 0–4 to 1–5
    return stars, probs[idx]
 
df[['model_star', 'model_confidence']] = df['cleaned_text'].apply(lambda x: pd.Series(predict_sentiment(x)))
 
# ------------------------
# 5. Combine rating with model
# ------------------------
# Map 1–5 stars to sentiment score (-1 to +1)
rating_map = {1: -1, 2: -0.5, 3: 0, 4: 0.5, 5: 1}
df['rating_score'] = df['rating'].map(rating_map)
df['model_score'] = df['model_star'].map(rating_map) * df['model_confidence']
 
# Weighted final score
df['final_score'] = 0.7 * df['model_score'] + 0.3 * df['rating_score']
 
def map_final_sentiment(score):
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    else:
        return "neutral"
 
df['final_sentiment'] = df['final_score'].apply(map_final_sentiment)
 
# ------------------------
# 6. Extract top words using attention weights
# ------------------------
def get_top_attention_words(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions  # List of attention layers
 
    attn = attentions[-1][0].mean(dim=0)  # last layer, average heads
    cls_attention = attn[0].cpu().numpy()  # CLS token attention
 
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    token_scores = list(zip(tokens, cls_attention))
    token_scores = [t for t in token_scores if t[0] not in tokenizer.all_special_tokens]
    token_scores = sorted(token_scores, key=lambda x: x[1], reverse=True)
    top_tokens = [t for t, s in token_scores[:5]]
    return " ".join(top_tokens)
 
df['top_words'] = df['cleaned_text'].apply(get_top_attention_words)
 
# ------------------------
# 7. Save final CSV
# ------------------------
df.to_csv(r"C:\Users\Kat\Documents\talabat_restaurants_final.csv", index=False, encoding='utf-8')
print(" CSV saved with sentiment")