import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


data = pd.read_csv("csgo.csv")

print(data.info())

data2 = data.drop(["date", "day", "month", "year", "wait_time_s"], axis=1)

corr = data.corr(numeric_only=True)
print(corr)

target = "result"
x = data2.drop(target, axis=1)
y = data2[target]
print(y.unique())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

ordinal_result = ["Lost", "Tie", "Win"]
ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Tie")),
        ("encoder", OrdinalEncoder(categories=[ordinal_result])),
])
ordinal_feature = ["result"]
#
# result_test = ordinal_transformer.fit_transform(y_train)

numeric_features = ["match_time_s", "team_a_rounds", "team_b_rounds", "ping", "kills", "assists", "deaths", "mvps", "hs_percent", "points"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())]
)

nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(sparse=False)),

])
nominal_feature = ["map"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        # ("ordinal", ordinal_transformer, ordinal_feature),
        ("nominal", nominal_transformer, nominal_feature),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("clf", RandomForestClassifier(max_depth=2, random_state=0))]
)

clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

for i, j in zip(y_test, y_predict):
    print("Actual {}. Predict {}".format(i, j))

print(classification_report(y_test, y_predict))