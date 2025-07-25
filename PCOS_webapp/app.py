import json

import catboost
import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model components
# model = joblib.load('model/best_model.pkl'
model = catboost.CatBoostClassifier()
model.load_model("../best_model")

# scaler = joblib.load("../scaler.pkl")
with open("../selected_features.json") as f:
    selected_features = json.load(f)["selected_features"]

df = pd.read_pickle("../unscaled_X.pkl")
df = df[selected_features]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Extract input from form
            user_input = [
                float(request.form.get(feature)) for feature in selected_features
            ]
            user_df = pd.DataFrame([user_input], columns=selected_features)

            combined = pd.concat([df, user_df], ignore_index=True)

            scaler = StandardScaler()
            combined_scaled = scaler.fit_transform(combined)
            combined = pd.DataFrame(combined_scaled, columns=combined.columns)

            print(combined.tail(1))
            print(model.classes_)

            probability = model.predict(
                combined.tail(1), prediction_type="Probability"
            )[:, 1][0]
            print(probability)
            result = "PCOS Detected" if probability > 0.5 else "No PCOS"

            return render_template(
                "pcos-home.html", prediction=result, prob=f"{probability*100:.2f}%"
            )
        except Exception as e:
            return render_template(
                "pcos-home.html",
                error=f"Invalid input. Please check your values. {e}",
                model_features=selected_features,
            )

    return render_template("pcos-home.html", model_features=selected_features)
    # return render_template("pcos-home.html")


@app.route("/about")
def about():
    return render_template("about-us.html")


if __name__ == "__main__":
    app.run(debug=True)
