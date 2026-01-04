import requests
import csv
import time
import random
import os

csv_file = os.path.join(os.getcwd(), "talabat_15_restaurants_reviews.csv")

# Map restaurant names to branch IDs (extracted from your URLs)
restaurants = {
    "99 Grill": 624149,
    "Pizza Hut": 600352,
    "Ninja Sushi": 734783,
    "Wazzup Dog": 628339,
    "Between Buns": 40177,
    "Dominos Pizza": 9778,
    "Shawerma Reem": 650859,
    "Shawarmaati": 47347,
    "Mr Hotdog": 661804,
    "Xn Shawerma": 736547,
    "Burger Maker": 683243,
    "Ibra Sandwich": 760475,
    "Crispy Chicken": 662822,
    "Buffalo Wings Rings": 638830,
    "Chicken Kingdom": 730696
}

all_reviews = []

for name, branch_id in restaurants.items():
    print(f"Scraping reviews for {name} (branch {branch_id})...")
    
    page = 1
    page_size = 50  # number of reviews per request
    total_pages = 1  # will update from first request

    while page <= total_pages:
        url = f"https://www.talabat.com/nextFeedbackApi/branches/{branch_id}/reviews/{page}/{page_size}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "accept": "application/json, text/plain, /"
        }
        response = requests.get(url, headers=headers)
        data = response.json()

        # Update total pages from API response
        total_pages = data.get("totalPages", 1)

        # Extract reviews
        for review in data.get("details", []):
            all_reviews.append({
                "restaurant_name": name,
                "review_date": review.get("date"),
                "rating": review.get("rate"),
                "review_text": review.get("review")
            })

        page += 1
        time.sleep(random.uniform(0.5, 1.5))  

# Save CSV
csv_file = os.path.join(os.getcwd(), "talabat_15_restaurants_reviews.csv")
keys = all_reviews[0].keys()
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(all_reviews)

print(f"\n Successfully saved {len(all_reviews)} reviews")
print(f"File location: {csv_file}")