import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("sales.csv")

print("Original Dataset")
print(df.head())

# -------------------------
# Data Cleaning
# -------------------------

# Remove rows having missing values
df_clean = df.dropna()

# Convert Date column
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

# Create Revenue column
df_clean['Revenue'] = df_clean['Quantity'] * df_clean['Unit_Price']

print("\nCleaned Dataset")
print(df_clean.head())

# -------------------------
# Summary Statistics
# -------------------------

print("\nSummary Statistics")
print(df_clean.describe())

# -------------------------
# Revenue by Product Category
# -------------------------

category_revenue = df_clean.groupby(
    "Product_Category"
)["Revenue"].sum()

print("\nRevenue by Category")
print(category_revenue)

# -------------------------
# Top 10 Customers
# -------------------------

top_customers = df_clean.groupby(
    "Customer_Name"
)["Revenue"].sum().sort_values(ascending=False).head(10)

print("\nTop 10 Customers")
print(top_customers)

# -------------------------
# Monthly Sales
# -------------------------

df_clean["Month"] = df_clean["Date"].dt.strftime("%b")

monthly_sales = df_clean.groupby(
    "Month"
)["Revenue"].sum()

# -------------------------
# Plot 1
# -------------------------

plt.figure(figsize=(10,5))
monthly_sales.plot(marker='o')

plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)

plt.show()

# -------------------------
# Plot 2
# -------------------------

plt.figure(figsize=(7,5))
category_revenue.plot(kind='bar')

plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Revenue")

plt.show()