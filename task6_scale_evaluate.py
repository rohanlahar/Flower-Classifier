"""
Project 2: Flower Classifier
Task 6: Improve the Model with Feature Scaling and Evaluation
-----------------------------------------------------------
Goal: Scale the 4 measurements to a common 0-1 range, retrain the
k-NN classifier, and compare its accuracy against the unscaled
baseline from Task 5. Then look deeper with a classification
report and confusion matrix, and visualize the results.

Why scaling matters: k-NN decides "closeness" using distance.
Petal length ranges roughly 1-7 cm while sepal width ranges roughly
2-4.4 cm, so without scaling, the feature with the larger range can
dominate the distance calculation. MinMaxScaler rescales every
feature to 0-1 so each one contributes fairly.

Requires: 'iris_train.csv', 'iris_test.csv', 'species_names.csv',
and 'unscaled_accuracy.txt' (all created by Task 4 / Task 5) must be
in the same folder.
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------
# Step 1: Load the same train/test split used in Task 4 & 5
# ---------------------------------------------------------
train_df = pd.read_csv("iris_train.csv")
test_df = pd.read_csv("iris_test.csv")
species_map = pd.read_csv("species_names.csv")

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target_names = species_map["species_name"].tolist()

X_train = train_df[feature_cols]
y_train = train_df["species_id"]
X_test = test_df[feature_cols]
y_test = test_df["species_id"]

with open("unscaled_accuracy.txt") as f:
    unscaled_accuracy = float(f.read())

print(f"Baseline (unscaled) accuracy from Task 5: {unscaled_accuracy * 100:.2f}%")

# ---------------------------------------------------------
# Step 2: Scale features to a 0-1 range
# ---------------------------------------------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use the SAME scaling as training

# ---------------------------------------------------------
# Step 3: Train a new k-NN model on the scaled features
# ---------------------------------------------------------
k = 3
knn_scaled = KNeighborsClassifier(n_neighbors=k)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_scaled.predict(X_test_scaled)
scaled_accuracy = knn_scaled.score(X_test_scaled, y_test)

print(f"New (scaled) accuracy: {scaled_accuracy * 100:.2f}%")

improvement = (scaled_accuracy - unscaled_accuracy) * 100
if improvement > 0:
    print(f"Scaling improved accuracy by {improvement:.2f} percentage points.")
elif improvement < 0:
    print(f"Scaling decreased accuracy by {abs(improvement):.2f} percentage points "
          f"(iris features are already on similar scales, so this can happen).")
else:
    print("Scaling made no difference for this split.")

# ---------------------------------------------------------
# Step 4: Detailed evaluation - classification report
# ---------------------------------------------------------
print("\nClassification Report (scaled model):")
print(classification_report(y_test, y_pred_scaled, target_names=target_names))

# ---------------------------------------------------------
# Step 5: Confusion matrix - which species get confused?
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_scaled)
print("Confusion Matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=target_names, columns=target_names))

# ---------------------------------------------------------
# Step 6: Save the scaled model, scaler, and results
# ---------------------------------------------------------
joblib.dump(knn_scaled, "knn_scaled_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

summary = pd.DataFrame({
    "Model": ["Unscaled k-NN", "Scaled k-NN"],
    "Accuracy (%)": [round(unscaled_accuracy * 100, 2), round(scaled_accuracy * 100, 2)]
})
summary.to_csv("accuracy_comparison.csv", index=False)
print("\nAccuracy comparison saved to 'accuracy_comparison.csv'.")

# ---------------------------------------------------------
# Step 7: Visualize - accuracy comparison + confusion matrix
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: bar chart comparing unscaled vs scaled accuracy
bars = axes[0].bar(
    ["Unscaled k-NN", "Scaled k-NN"],
    [unscaled_accuracy * 100, scaled_accuracy * 100],
    color=["#DD8452", "#4C72B0"]
)
axes[0].set_ylim(0, 105)
axes[0].set_ylabel("Test accuracy (%)")
axes[0].set_title("Accuracy: Unscaled vs Scaled Features")
for bar in bars:
    height = bar.get_height()
    axes[0].annotate(f"{height:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 4), textcoords="offset points", ha="center")
axes[0].grid(axis="y", alpha=0.3)

# Right: confusion matrix for the scaled model
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title("Confusion Matrix (Scaled k-NN)")

plt.tight_layout()
plot_path = "knn_evaluation_plot.png"
plt.savefig(plot_path, dpi=150)
print(f"Evaluation plot saved to '{plot_path}'.")
