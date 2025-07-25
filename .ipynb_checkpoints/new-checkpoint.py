import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

one = pd.read_csv("PCOS_data.csv")
two = pd.read_csv("CLEAN- PCOS SURVEY SPREADSHEET.csv")

pd.set_option('display.max_rows', None)  # or 1000

def one_preprocess(df):
    df.rename(columns={" Age (yrs)": "age", "BMI": "bmi", "Cycle(R/I)": "cycle", "PCOS (Y/N)": "detection",
                       "Cycle length(days)": "days", "Skin darkening (Y/N)": "skin_darkening",
                       "Weight gain(Y/N)": "weight_gain", "hair growth(Y/N)": "hair_growth",
                       "Hair loss(Y/N)": "hair_loss", "Pimples(Y/N)": "pimples", "Fast food (Y/N)": "fast_food",
                       "Reg.Exercise(Y/N)": "exercise"}, inplace=True)
    df = df[["age", "bmi", "cycle", "days", "skin_darkening", "weight_gain", "hair_growth","hair_loss", "pimples", "fast_food", "exercise", "detection"]].copy()
    df.cycle = df.cycle.map({2:1, 4:2})
    df.cycle = df.cycle.astype('Int64')
    df.fast_food = df.fast_food.astype('Int64')
    return df

def two_preprocess(df):
    df["bmi"] = round((df["weight"]) / ((df["height"]/100) ** 2), 1)
    df = df.drop(df[df.bmi > 50].index)
    df = df[["age", "bmi", "cycle", "days", "skin_darkening", "weight_gain", "hair_growth","hair_loss", "pimples", "fast_food", "exercise", "detection"]].copy()
    return df

one = one_preprocess(one)
two = two_preprocess(two)

merged = pd.concat([one, two], ignore_index=True)
merged.dropna(inplace=True)

x = merged.drop(columns="detection")
y = merged["detection"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, shuffle=True, random_state=8)
dtrain_clf = xgb.DMatrix(x_train, y_train, enable_categorical=True)
dtest_clf = xgb.DMatrix(x_test, y_test, enable_categorical=True)
evals = [(dtrain_clf, "train"), (dtest_clf, "validation")]

params = {"objective": "binary:logistic"}
n = 1000

results = xgb.XGBClassifier(
   params, dtrain_clf,
   num_boost_round=n,
   nfold=5,
#    verbose_eval=50,
   metrics=["logloss", "auc"]
)

print(results)
