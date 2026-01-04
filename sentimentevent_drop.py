import pandas as pd

# ===============================
# 1. Load raw reviews data
# ===============================
df = pd.read_csv(
    r"C:\Users\Kat\Downloads\talabat_restaurants_final.csv",
    encoding="utf-8"
)

# ===============================
# 2. Convert review_date to datetime
# ===============================
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
df = df.dropna(subset=["review_date"])

# ===============================
# 3. Create Year-Month column
# ===============================
df["year_month"] = df["review_date"].dt.to_period("M").astype(str)

# ===============================
# 4. Create sentiment flags
# ===============================
df["is_positive"] = (df["final_sentiment"] == "positive").astype(int)
df["is_negative"] = (df["final_sentiment"] == "negative").astype(int)
df["is_neutral"]  = (df["final_sentiment"] == "neutral").astype(int)

# ===============================
# 5. Aggregate monthly timeline metrics
# ===============================
timeline_df = (
    df.groupby(["restaurant_name", "year_month"])
      .agg(
          review_count=("final_sentiment", "count"),
          avg_final_score=("final_score", "mean"),
          positive_ratio=("is_positive", "mean"),
          negative_ratio=("is_negative", "mean"),
          neutral_ratio=("is_neutral", "mean")
      )
      .reset_index()
)

# ===============================
# 6. Apply minimum review threshold
# ===============================
MIN_REVIEWS = 10
timeline_df = timeline_df[timeline_df["review_count"] >= MIN_REVIEWS]

# ===============================
# 7. Convert ratios to percentages
# ===============================
timeline_df["positive_ratio"] = (timeline_df["positive_ratio"] * 100).round(1)
timeline_df["negative_ratio"] = (timeline_df["negative_ratio"] * 100).round(1)
timeline_df["neutral_ratio"]  = (timeline_df["neutral_ratio"] * 100).round(1)
timeline_df["avg_final_score"] = timeline_df["avg_final_score"].round(2)

# ===============================
# 8. Sort for proper timeline order
# ===============================
timeline_df = timeline_df.sort_values(
    by=["restaurant_name", "year_month"]
)

# ===============================
# 9. Calculate month-to-month sentiment change
# ===============================
timeline_df["positive_change"] = (
    timeline_df.groupby("restaurant_name")["positive_ratio"]
    .diff()
)

timeline_df["negative_change"] = (
    timeline_df.groupby("restaurant_name")["negative_ratio"]
    .diff()
)

# ===============================
# 10. Define spike/drop thresholds
# ===============================
DROP_THRESHOLD = -15   # -15% or worse
SPIKE_THRESHOLD = 15   # +15% or better

def classify_change(x):
    if pd.isna(x):
        return "Stable"
    elif x <= DROP_THRESHOLD:
        return "Sentiment Drop"
    elif x >= SPIKE_THRESHOLD:
        return "Sentiment Spike"
    else:
        return "Stable"

# ===============================
# 11. Classify sentiment events
# ===============================
timeline_df["sentiment_event"] = timeline_df["positive_change"].apply(classify_change)

# ===============================
# 12. Save final output
# ===============================
timeline_df.to_csv(
    r"C:\Users\Kat\Downloads\restaurant_Timeline_analysis_drop.csv",
    index=False,
    encoding="utf-8"
)

print(" Timeline analysis + sentiment spike/drop detection completed")
print(" File saved: restaurant_sentiment_events.csv")