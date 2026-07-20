

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib
matplotlib.use("Agg")   # draw images without opening a window
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "That page/route was not found (404). Check the URL."}), 404


@app.errorhandler(500)
def handle_500(e):
    return jsonify({"error": "Server error: " + str(e)}), 500


@app.errorhandler(Exception)
def handle_any_error(e):
    return jsonify({"error": str(e)}), 500


def figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    plt.close(fig)
    return "data:image/png;base64," + image_base64


def train_model(payload):
    y_values = np.array(payload["y_values"])
    feature_names = []
    feature_data = {}

    for feature in payload["features"]:
        name = feature["name"]
        values = feature["values"]
        feature_names.append(name)
        feature_data[name] = values

    x_data = pd.DataFrame(feature_data)

    model = LinearRegression()
    model.fit(x_data, y_values)
    predictions = model.predict(x_data)

    return model, x_data, y_values, predictions, feature_names


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        payload = request.get_json()
        y_name = payload["y_name"]

        model, x_data, y_values, predictions, feature_names = train_model(payload)

        mse = mean_squared_error(y_values, predictions)
        mae = mean_absolute_error(y_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_values, predictions)
        intercept = float(model.intercept_)
        coefficients = [float(v) for v in model.coef_]

        equation_terms = []
        for i in range(len(coefficients)):
            term = "(" + str(round(coefficients[i], 4)) + " * " + feature_names[i] + ")"
            equation_terms.append(term)
        equation_str = y_name + " = " + str(round(intercept, 4)) + " + " + " + ".join(equation_terms)

        slopes = []
        for i in range(len(feature_names)):
            slopes.append({
                "name": feature_names[i],
                "value": round(coefficients[i], 4)
            })

        return jsonify({
            "equation": equation_str,
            "intercept": round(intercept, 4),
            "slopes": slopes,
            "mae": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/visualize", methods=["POST"])
def visualize():
    try:
        payload = request.get_json()
        y_name = payload["y_name"]

        model, x_data, y_values, predictions, feature_names = train_model(payload)

        mse = mean_squared_error(y_values, predictions)
        mae = mean_absolute_error(y_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_values, predictions)

        fig1 = plt.figure(figsize=(7, 5))
        plt.scatter(y_values, predictions, color="#1b6982", edgecolor="#065f46",
                    linewidth=1, s=50, alpha=0.85)

        min_val = min(float(np.min(y_values)), float(np.min(predictions)))
        max_val = max(float(np.max(y_values)), float(np.max(predictions)))
        plt.plot([min_val, max_val], [min_val, max_val], color="#647486",
                  linestyle="--", linewidth=1.5, label="Perfect prediction line")

        plt.xlabel("Actual " + y_name)
        plt.ylabel("Predicted " + y_name)
        plt.title("Model Prediction Accuracy")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)

        image1 = figure_to_base64(fig1)

        metric_names = ["R2 Score", "RMSE", "MAE", "MSE"]
        metric_values = [float(r2), float(rmse), float(mae), float(mse)]
        colors = ["#4f46e5", "#3b82f6", "#60a5fa", "#93c5fd"]

        fig2 = plt.figure(figsize=(7, 5))
        bars = plt.bar(metric_names, metric_values, color=colors)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                str(round(height, 4)),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center", va="bottom", fontweight="bold"
            )

        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.title("Model Evaluation Metrics")
        plt.grid(axis="y", linestyle=":", alpha=0.6)

        image2 = figure_to_base64(fig2)

        return jsonify({
            "image1": image1,
            "image2": image2
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        new_values = payload["new_values"]   # one number per feature, in order

        model, x_data, y_values, predictions, feature_names = train_model(payload)

        sample = pd.DataFrame([new_values], columns=feature_names)
        predicted_value = float(model.predict(sample)[0])

        return jsonify({
            "prediction": round(predicted_value, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
