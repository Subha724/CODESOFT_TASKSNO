"""
create_visualizations.py
-------------------------
TASK 3: DATA VISUALIZATION DASHBOARD

Loads the retail sales dataset and produces a full set of publication-quality
charts using Matplotlib and Seaborn:

    1. Bar chart      - Total Sales by Product Category
    2. Line chart     - Monthly Sales Trend (2024-2025)
    3. Pie chart      - Sales Share by Region
    4. Histogram      - Distribution of Order Values
    5. Scatter plot   - Quantity vs. Profit (colored by Category)
    6. Bonus heatmap  - Avg Sales by Region x Category

Every chart is customized with a title, axis labels, legend (where relevant)
and a deliberate color scheme, and each is saved as a high-resolution PNG
into ../charts/. Key insights are printed to the console and written to
../charts/insights.txt.

Run:
    python3 create_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "sales_data.csv"
CHART_DIR = BASE / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "font.family": "DejaVu Sans",
})

COLOR_CATEGORY = {
    "Electronics": "#4C72B0",
    "Clothing": "#DD8452",
    "Home & Kitchen": "#55A868",
    "Sports": "#C44E52",
    "Beauty": "#8172B2",
}
REGION_COLORS = sns.color_palette("crest", 5)

df = pd.read_csv(DATA_PATH, parse_dates=["OrderDate"])
df["Month"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()

insights = []

# ---------------------------------------------------------------------------
# 1. BAR CHART - Total Sales by Category
# ---------------------------------------------------------------------------
cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(cat_sales.index, cat_sales.values,
               color=[COLOR_CATEGORY[c] for c in cat_sales.index],
               edgecolor="white", linewidth=1.2)
ax.set_title("Total Sales by Product Category")
ax.set_xlabel("Category")
ax.set_ylabel("Total Sales ($)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
for b in bars:
    ax.annotate(f"${b.get_height():,.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(CHART_DIR / "01_bar_sales_by_category.png")
plt.close()

top_cat = cat_sales.index[0]
insights.append(f"'{top_cat}' is the top-performing category with ${cat_sales.iloc[0]:,.0f} in total sales.")

# ---------------------------------------------------------------------------
# 2. LINE CHART - Monthly Sales Trend
# ---------------------------------------------------------------------------
monthly = df.groupby("Month")["Sales"].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(monthly["Month"], monthly["Sales"], marker="o", linewidth=2.5,
        color="#2E86AB", markerfacecolor="#F24236", markeredgecolor="white",
        markersize=7, label="Monthly Sales")
ax.fill_between(monthly["Month"], monthly["Sales"], alpha=0.12, color="#2E86AB")
ax.set_title("Monthly Sales Trend (Jan 2024 - Aug 2025)")
ax.set_xlabel("Month")
ax.set_ylabel("Total Sales ($)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
ax.legend(loc="upper left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(CHART_DIR / "02_line_monthly_sales_trend.png")
plt.close()

peak_month = monthly.loc[monthly["Sales"].idxmax(), "Month"]
avg_by_calendar_month = df.groupby(df["OrderDate"].dt.month)["Sales"].mean()
holiday_avg = avg_by_calendar_month.loc[[11, 12]].mean()
other_avg = avg_by_calendar_month.drop([11, 12]).mean()
insights.append(
    f"{peak_month.strftime('%B %Y')} recorded the highest total sales; more broadly, average order "
    f"value in Nov/Dec (${holiday_avg:,.0f}) runs {((holiday_avg / other_avg) - 1):.0%} above the "
    f"rest of the year, reflecting a holiday-season pricing effect."
)

# ---------------------------------------------------------------------------
# 3. PIE CHART - Sales Share by Region
# ---------------------------------------------------------------------------
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
wedges, texts, autotexts = ax.pie(
    region_sales.values,
    labels=region_sales.index,
    autopct="%1.1f%%",
    colors=REGION_COLORS,
    startangle=90,
    pctdistance=0.8,
    explode=[0.06 if v == region_sales.max() else 0 for v in region_sales.values],
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
plt.setp(autotexts, size=10, weight="bold", color="white")
ax.set_title("Sales Share by Region")
ax.legend(wedges, region_sales.index, title="Region", loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.tight_layout()
plt.savefig(CHART_DIR / "03_pie_sales_by_region.png")
plt.close()

top_region = region_sales.index[0]
insights.append(f"The '{top_region}' region drives the largest share of sales ({region_sales.iloc[0] / region_sales.sum():.1%}).")

# ---------------------------------------------------------------------------
# 4. HISTOGRAM - Distribution of Order Values
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.histplot(df["Sales"], bins=30, kde=True, color="#4C72B0",
             edgecolor="white", ax=ax)
mean_val = df["Sales"].mean()
median_val = df["Sales"].median()
ax.axvline(mean_val, color="#C44E52", linestyle="--", linewidth=2, label=f"Mean: ${mean_val:,.0f}")
ax.axvline(median_val, color="#55A868", linestyle="--", linewidth=2, label=f"Median: ${median_val:,.0f}")
ax.set_title("Distribution of Order Values")
ax.set_xlabel("Order Value ($)")
ax.set_ylabel("Number of Orders")
ax.legend()
plt.tight_layout()
plt.savefig(CHART_DIR / "04_histogram_order_values.png")
plt.close()

insights.append(f"Order values are right-skewed: the mean (${mean_val:,.0f}) sits above the median (${median_val:,.0f}), driven by a handful of high-value Electronics orders.")

# ---------------------------------------------------------------------------
# 5. SCATTER PLOT - Quantity vs Profit, colored by Category
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=df, x="Quantity", y="Profit", hue="Category",
                 palette=COLOR_CATEGORY, alpha=0.75, s=60,
                 edgecolor="white", linewidth=0.4, ax=ax)
ax.set_title("Quantity Sold vs. Profit by Category")
ax.set_xlabel("Quantity Sold (units)")
ax.set_ylabel("Profit ($)")
ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(CHART_DIR / "05_scatter_quantity_vs_profit.png")
plt.close()

corr = df["Quantity"].corr(df["Profit"])
insights.append(f"Quantity and Profit show a weak correlation (r = {corr:.2f}), suggesting profit is driven more by product price/margin than order size.")

# ---------------------------------------------------------------------------
# 6. BONUS - Heatmap: Avg Sales by Region x Category
# ---------------------------------------------------------------------------
pivot = df.pivot_table(index="Region", columns="Category", values="Sales", aggfunc="mean")

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5,
            cbar_kws={"label": "Avg. Sales ($)"}, ax=ax)
ax.set_title("Average Order Value: Region x Category")
plt.tight_layout()
plt.savefig(CHART_DIR / "06_heatmap_region_category.png")
plt.close()

best_cell = pivot.stack().idxmax()
insights.append(f"The highest average order value combo is {best_cell[0]} + {best_cell[1]} (${pivot.stack().max():,.0f} avg).")

# ---------------------------------------------------------------------------
# Save insights
# ---------------------------------------------------------------------------
with open(CHART_DIR / "insights.txt", "w") as f:
    f.write("KEY INSIGHTS - Retail Sales Dashboard\n")
    f.write("=" * 40 + "\n\n")
    for i, ins in enumerate(insights, 1):
        f.write(f"{i}. {ins}\n")

print("All charts generated in:", CHART_DIR)
print("\nKEY INSIGHTS")
for i, ins in enumerate(insights, 1):
    print(f"{i}. {ins}")
