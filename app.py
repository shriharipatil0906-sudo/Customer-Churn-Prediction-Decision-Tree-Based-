from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    account_length = float(request.form["account_length"])
    international_plan = int(request.form["international_plan"])
    voice_mail_plan = int(request.form["voice_mail_plan"])
    vmail_messages = float(request.form["vmail_messages"])

    day_minutes = float(request.form["day_minutes"])
    day_calls = float(request.form["day_calls"])
    day_charge = float(request.form["day_charge"])

    eve_minutes = float(request.form["eve_minutes"])
    eve_calls = float(request.form["eve_calls"])
    eve_charge = float(request.form["eve_charge"])

    night_minutes = float(request.form["night_minutes"])
    night_calls = float(request.form["night_calls"])
    night_charge = float(request.form["night_charge"])

    intl_minutes = float(request.form["intl_minutes"])
    intl_calls = float(request.form["intl_calls"])
    intl_charge = float(request.form["intl_charge"])

    customer_service_calls = float(
        request.form["customer_service_calls"]
    )

    input_data = np.array([[
        account_length,
        international_plan,
        voice_mail_plan,
        vmail_messages,
        day_minutes,
        day_calls,
        day_charge,
        eve_minutes,
        eve_calls,
        eve_charge,
        night_minutes,
        night_calls,
        night_charge,
        intl_minutes,
        intl_calls,
        intl_charge,
        customer_service_calls
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Customer is likely to Churn"
    else:
        result = "Customer is likely to Stay"

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)