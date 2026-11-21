from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "house_prices.csv")

print("Dataset preview:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

df = df.drop_duplicates()
df = df.dropna()

# EDA: correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Housing Features Correlation")
plt.tight_layout()
plt.savefig(ROOT / "correlation_heatmap.png")
plt.close()

# Features and target
X = df[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))

sample_house = pd.DataFrame([{
    "Area": 1500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Age": 5
}])
predicted_price = model.predict(sample_house)[0]
print("Predicted price for sample house:", round(predicted_price, 2))
print("\nHeatmap saved as correlation_heatmap.png")
