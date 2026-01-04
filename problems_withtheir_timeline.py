import pandas as pd
from transformers import pipeline

# ===============================
# 1. Load original sentiment file
# ===============================
df = pd.read_csv(
    r"C:\Users\Kat\Downloads\talabat_restaurants_final.csv",
    encoding="utf-8"
)

# ===============================
# 2. Prepare dates
# ===============================
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
df = df.dropna(subset=["review_date"])
df["year_month"] = df["review_date"].dt.to_period("M").astype(str)

# ===============================
# 3. Keep ONLY negative reviews
# ===============================
df = df[df["final_sentiment"] == "Negative"]

# ===============================
# 4. Define candidate problem categories
# ===============================
problem_categories = [
    "Food Temperature",
    "Taste & Flavor",
    "Service",
    "Portion Size",
    "Missing / Wrong Items",
    "Cleanliness",
    "Delivery Speed",
    "Packaging",
    "Price / Value",
    "Incorrect Billing",
    "Other"
]

# ===============================
# 5. Load zero-shot classifier
# ===============================
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ===============================
# 6. Detect problems for each review
# ===============================
def detect_problems(text, categories, threshold=0.2):
    if not isinstance(text, str) or text.strip() == "":
        return []
    result = classifier(text, candidate_labels=categories, multi_label=True)
    # Only keep labels above threshold
    detected = [label for label, score in zip(result['labels'], result['scores']) if score >= threshold]
    return detected

# Apply detection
df["detected_problem"] = df["review_text"].apply(lambda x: detect_problems(x, problem_categories))
df = df.explode("detected_problem")
df = df[df["detected_problem"].notna() & (df["detected_problem"] != "")]

# ===============================
# 7. Aggregate problems monthly
# ===============================
problem_timeline = (
    df.groupby(["restaurant_name", "year_month", "detected_problem"])
      .size()
      .reset_index(name="problem_count")
)

# ===============================
# 8. Convert to percentage
# ===============================
total_per_month = (
    problem_timeline.groupby(["restaurant_name", "year_month"])["problem_count"]
    .transform("sum")
)

problem_timeline["problem_percentage"] = (
    (problem_timeline["problem_count"] / total_per_month) * 100
).round(1)

# ===============================
# 9. Sort cleanly
# ===============================
problem_timeline = problem_timeline.sort_values(
    by=["restaurant_name", "year_month", "problem_percentage"],
    ascending=[True, True, False]
)

# ===============================
# 10. Save output
# ===============================
problem_timeline.to_csv(
    r"C:\Users\Kat\Downloads\restaurant_problem_timeline_monthly.csv",
    index=False,
    encoding="utf-8"
)

print(" Problem timeline analysis completed")
print(" File saved: restaurant_problem_timeline_monthly.csv")