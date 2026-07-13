

from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use("Agg")   # lets matplotlib draw images without opening a window
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
def figure_to_base64(fig):
    buffer = io.BytesIO()               # a temporary "file" kept in memory
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)                      # go back to the start of the buffer
    image_bytes = buffer.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    plt.close(fig)                      # free up memory
    return "data:image/png;base64," + image_base64
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/visualize_regression", methods=["POST"])
def visualize_regression():

    # Step 1: Read the data sent from the webpage (JSON format)
    data = request.get_json()

    x = data["x"]
    y = data["y"]
    m = data["m"]
    c = data["c"]
    mse = data["mse"]
    mae = data["mae"]
    rmse = data["rmse"]
    r_squared = data["r2"]

    n = len(x)

    # Step 2: Calculate the predicted Y values using Y = mX + c
    y_pred = []
    for i in range(n):
        predicted_value = (m * x[i]) + c
        y_pred.append(predicted_value)

    # ================================================
    # DIAGRAM 1: Regression Line Plot
    # (identical to the main notebook code)
    # ================================================
    fig1 = plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="blue", label="Actual points")
    plt.plot(x, y_pred, color="red", label="Regression line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Simple Linear Regression")
    plt.legend()

    metrics_text = (
        "Y = " + str(round(m, 4)) + "X + " + str(round(c, 4)) + "\n"
        "MAE  = " + str(round(mae, 4)) + "\n"
        "MSE  = " + str(round(mse, 4)) + "\n"
        "RMSE = " + str(round(rmse, 4)) + "\n"
        "R-squared = " + str(round(r_squared, 4))
    )

    plt.gca().text(
        0.02, 0.98, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    image1 = figure_to_base64(fig1)

    # ================================================
    # DIAGRAM 2: Model Metrics Dot Chart
    # (identical to the main notebook code)
    # ================================================
    metric_names = ["MSE", "MAE", "RMSE", "R2"]
    metric_values = [mse, mae, rmse, r_squared]

    fig2 = plt.figure(figsize=(7, 5))
    plt.scatter(metric_values, metric_names, color="steelblue", s=80, zorder=3)

    for i in range(len(metric_names)):
        name = metric_names[i]
        value = metric_values[i]
        plt.hlines(y=name, xmin=min(metric_values) - 0.05, xmax=value,
                   color="lightgray", linewidth=1, zorder=1)
        plt.text(value, name, "  " + str(round(value, 4)), va="center", fontsize=10)

    plt.title("Model Metrics")
    plt.xlabel("Value")
    plt.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()

    image2 = figure_to_base64(fig2)

    # Step 3: Send both images back to the webpage as JSON
    return jsonify({
        "image1": image1,
        "image2": image2
    })
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
