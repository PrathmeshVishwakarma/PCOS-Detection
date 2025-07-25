from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# # Load model components
# model = joblib.load('model/best_model.pkl')
# scaler = joblib.load('model/scaler.pkl')
# selected_features = joblib.load('model/selected_features.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract input from form
            user_input = [float(request.form.get(feature)) for feature in selected_features]
            user_df = pd.DataFrame([user_input], columns=selected_features)

            # Scale and predict
            user_scaled = scaler.transform(user_df)
            probability = model.predict_proba(user_scaled)[0][1]  # Probability of PCOS
            result = "PCOS Detected" if probability > 0.5 else "No PCOS"

            return render_template('pcos-home.html', prediction=result, prob=f"{probability*100:.2f}%")
        except:
            return render_template(
                'pcos-home.html',
                error="Invalid input. Please check your values.",
                model_features=selected_features
            )

    # return render_template('pcos-home.html', model_features=selected_features)
    return render_template("pcos-home.html")

if __name__ == '__main__':
    app.run(debug=True)


