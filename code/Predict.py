import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ──────────────────────────────────────────────
# Race Time Prediction – CS 362 Final Project
# ──────────────────────────────────────────────

# Convert MM:SS to decimal minutes
def to_decimal(time_str):
    parts = str(time_str).split(":")
    return int(parts[0]) + int(parts[1]) / 60

# Convert decimal minutes back to MM:SS
def to_mmss(decimal):
    minutes = int(decimal)
    seconds = round((decimal - minutes) * 60)
    return f"{minutes}:{seconds:02d}"

# Load dataset and convert all MM:SS columns to decimal
data = pd.read_csv("data.csv")
data["avg_pace"]  = data["avg_pace"].apply(to_decimal)
data["last_race"] = data["last_race"].apply(to_decimal)
data["next_race"] = data["next_race"].apply(to_decimal)

print("=" * 50)
print("   Race Time Prediction Model – CS 362")
print("=" * 50)

# Features and target
X = data[["mileage", "avg_pace", "last_race"]]
y = data["next_race"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Show what the model learned
print("\n🧠 What the Model Learned:")
print(f"   Mileage weight:    {model.coef_[0]:.4f}")
print(f"   Avg pace weight:   {model.coef_[1]:.4f}")
print(f"   Last race weight:  {model.coef_[2]:.4f}")
print(f"   Intercept:         {model.intercept_:.4f}")

# Model accuracy
train_preds = model.predict(X)
mae = mean_absolute_error(y, train_preds)
r2  = r2_score(y, train_preds)
print(f"\n📊 Model Accuracy:")
print(f"   Mean Absolute Error: {mae:.3f} minutes")
print(f"   R² Score:            {r2:.3f}  (1.0 = perfect)")

# Try your own runner
print("\n" + "=" * 50)
print("   Try Your Own Runner")
print("=" * 50)
print("   Enter your training stats in MM:SS format\n")

try:
    mileage   = float(input("   Weekly mileage (miles):          "))
    avg_pace  = to_decimal(input("   Avg training pace (MM:SS):       "))
    last_race = to_decimal(input("   Last race time (MM:SS):          "))

    new_data = pd.DataFrame([[mileage, avg_pace, last_race]],
                            columns=["mileage", "avg_pace", "last_race"])
    result = model.predict(new_data)[0]
    print(f"\n   ➜  Predicted Race Time: {to_mmss(result)}")

except Exception as e:
    print(f"   Error: {e}")

print("\n✅ Done.\n")
