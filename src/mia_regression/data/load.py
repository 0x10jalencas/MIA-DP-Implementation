import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/raw/insurance.csv")
df = df.dropna(subset=["age", "bmi", "children", "smoker", "region", "charges"])
df = df.drop_duplicates()
df["age"] = df["age"].astype(float)
df["children"] = df["children"].astype(int)
df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

X = df.drop("charges", axis=1)
y = df["charges"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_target, X_shadow, y_target, y_shadow = train_test_split(X_scaled, y, test_size=0.6, random_state=42)
X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(
    X_target, y_target, test_size=0.3, random_state=42
)

print(f"Target train: {X_t_train.shape}, Target test: {X_t_test.shape}")
print(f"Shadow pool: {X_shadow.shape}")
