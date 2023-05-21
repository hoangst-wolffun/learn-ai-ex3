import pandas as pd

data = pd.read_csv("csgo.csv")

print(data.describe())

data = data.drop(["day", "month", "year", "wait_time_s"], axis=1)

corr = data.corr(numeric_only=True)
print(corr)