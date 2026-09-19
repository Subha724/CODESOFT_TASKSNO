"""
generate_data.py
-----------------
Generates a realistic synthetic RETAIL SALES dataset used throughout the
Data Visualization Dashboard project (Task 3).

Run:
    python3 generate_data.py

Output:
    ../data/sales_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible results every time the script is run
np.random.seed(42)

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"

# ---------------------------------------------------------------------------
# Reference lists
# ---------------------------------------------------------------------------
regions = ["North", "South", "East", "West", "Central"]

categories_products = {
    "Electronics": ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Tablet"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sneakers", "Dress"],
    "Home & Kitchen": ["Blender", "Cookware Set", "Vacuum Cleaner", "Air Fryer", "Lamp"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Running Shoes", "Bicycle", "Tennis Racket"],
    "Beauty": ["Perfume", "Skincare Kit", "Hair Dryer", "Makeup Kit", "Shampoo"],
}

payment_methods = ["Credit Card", "Debit Card", "UPI", "Cash", "Wallet"]

# Base price ranges per category, used to keep numbers realistic
price_ranges = {
    "Electronics": (60, 1200),
    "Clothing": (10, 120),
    "Home & Kitchen": (20, 300),
    "Sports": (15, 400),
    "Beauty": (8, 150),
}

N = 2000  # number of orders

rows = []
date_range = pd.date_range(start="2024-01-01", end="2025-08-31", freq="D")

for i in range(N):
    category = np.random.choice(list(categories_products.keys()),
                                 p=[0.28, 0.22, 0.18, 0.17, 0.15])
    product = np.random.choice(categories_products[category])
    region = np.random.choice(regions, p=[0.24, 0.20, 0.19, 0.22, 0.15])
    order_date = np.random.choice(date_range)

    low, high = price_ranges[category]
    unit_price = np.round(np.random.uniform(low, high), 2)
    quantity = np.random.randint(1, 8)

    # Slight seasonal boost around Nov/Dec (holiday sales) and mid-year sale (June)
    month = pd.Timestamp(order_date).month
    seasonal_multiplier = 1.0
    if month in (11, 12):
        seasonal_multiplier = 1.35
    elif month == 6:
        seasonal_multiplier = 1.15

    sales = np.round(unit_price * quantity * seasonal_multiplier, 2)
    cost = np.round(sales * np.random.uniform(0.55, 0.8), 2)
    profit = np.round(sales - cost, 2)

    discount_pct = np.random.choice([0, 5, 10, 15, 20], p=[0.4, 0.25, 0.2, 0.1, 0.05])
    rating = np.clip(np.round(np.random.normal(4.2, 0.6), 1), 1.0, 5.0)
    customer_age = int(np.clip(np.random.normal(35, 12), 18, 70))
    payment = np.random.choice(payment_methods, p=[0.35, 0.25, 0.2, 0.1, 0.1])

    rows.append({
        "OrderID": f"ORD{10000 + i}",
        "OrderDate": pd.Timestamp(order_date).strftime("%Y-%m-%d"),
        "Region": region,
        "Category": category,
        "Product": product,
        "UnitPrice": unit_price,
        "Quantity": quantity,
        "DiscountPct": discount_pct,
        "Sales": sales,
        "Cost": cost,
        "Profit": profit,
        "CustomerAge": customer_age,
        "Rating": rating,
        "PaymentMethod": payment,
    })

df = pd.DataFrame(rows)
df.sort_values("OrderDate", inplace=True)
df.reset_index(drop=True, inplace=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"Generated {len(df)} rows -> {OUT_PATH}")
print(df.head())
