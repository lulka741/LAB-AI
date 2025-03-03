import pandas as pd

df = pd.read_csv("titanic.csv")
print(df.head())

print(df.isnull().sum())

df.fillna({"Age": df["Age"].median()}, inplace=True)
df.fillna({"Fare": df["Fare"].median()}, inplace=True)

print("\n",df.isnull().sum())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

df.drop(columns=["Cabin", "Ticket"], inplace=True)

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True, dtype=int)

print(df.head())

df.to_csv("processed_titanic.csv", index=False)