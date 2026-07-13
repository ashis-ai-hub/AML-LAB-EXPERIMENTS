import matplotlib.pyplot as plt

raw_x = input("Enter X values separated by space: ")
raw_y = input("Enter Y values separated by space: ")

x = []
for value in raw_x.split():
    x.append(float(value))

y = []
for value in raw_y.split():
    y.append(float(value))

n = len(x)

sum_x = 0
sum_y = 0
for i in range(n):
    sum_x = sum_x + x[i]
    sum_y = sum_y + y[i]

mean_x = sum_x / n
mean_y = sum_y / n

numerator = 0
denominator = 0
for i in range(n):
    diff_x = x[i] - mean_x
    diff_y = y[i] - mean_y
    numerator = numerator + (diff_x * diff_y)
    denominator = denominator + (diff_x * diff_x)

m = numerator / denominator
c = mean_y - (m * mean_x)

print("\nSlope (m):", round(m, 4))
print("Intercept (c):", round(c, 4))
print("Equation: Y =", round(m, 4), "* X +", round(c, 4))

y_pred = []
for i in range(n):
    predicted_value = (m * x[i]) + c
    y_pred.append(predicted_value)

errors = []
for i in range(n):
    error_value = y[i] - y_pred[i]
    errors.append(error_value)

sum_abs_error = 0
for e in errors:
    sum_abs_error = sum_abs_error + abs(e)
mae = sum_abs_error / n

sum_squared_error = 0
for e in errors:
    sum_squared_error = sum_squared_error + (e * e)
mse = sum_squared_error / n

rmse = mse ** 0.5

ss_total = 0
for value in y:
    ss_total = ss_total + (value - mean_y) ** 2

ss_residual = sum_squared_error
r_squared = 1 - (ss_residual / ss_total)

print("\nMAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R-squared:", round(r_squared, 4))

plt.figure(figsize=(7, 5))
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

plt.show()

metric_names = ["MSE", "MAE", "RMSE", "R2"]
metric_values = [mse, mae, rmse, r_squared]

plt.figure(figsize=(7, 5))
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
plt.show()
