"""
Project 2: Flower Classifier
Task 5: Train a k-Nearest Neighbors (k-NN) Classifier
-----------------------------------------------------------
Goal: Load the train/test split from Task 4 and train a k-NN
classifier that predicts a flower's species from its measurements.

How k-NN works: for a new flower, it measures the distance to every
flower in the training set, finds the "k" closest ones, and predicts
the species that appears most often among those neighbors.

Requires: 'iris_train.csv', 'iris_test.csv', and 'species_names.csv'
(all created by task4_prepare_data.py) must be in the same folder.

This script saves the trained (unscaled) model and its predictions
so Task 6 can compare against a scaled version without retraining
from scratch.
"""

import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------
# Step 1: Load the train/test split produced by Task 4
# ---------------------------------------------------------
train_df = pd.read_csv("iris_train.csv")
test_df = pd.read_csv("iris_test.csv")
species_map = pd.read_csv("species_names.csv")

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_train = train_df[feature_cols]
y_train = train_df["species_id"]
X_test = test_df[feature_cols]
y_test = test_df["species_id"]

print(f"Loaded {len(X_train)} training flowers and {len(X_test)} test flowers.")

# ---------------------------------------------------------
# Step 2: Create and train the k-NN classifier (k = 3)
# ---------------------------------------------------------
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 3: Predict on the test set and check accuracy
# ---------------------------------------------------------
y_pred = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)

print(f"\nk-NN model trained with k={k} neighbors.")
print(f"Test accuracy (unscaled features): {accuracy * 100:.2f}%")

# ---------------------------------------------------------
# Step 4: Show a few example predictions
# ---------------------------------------------------------
name_lookup = dict(zip(species_map["species_id"], species_map["species_name"]))

preview = test_df[feature_cols].copy()
preview["Actual"] = test_df["species_id"].map(name_lookup)
preview["Predicted"] = pd.Series(y_pred).map(name_lookup)
preview["Correct"] = preview["Actual"] == preview["Predicted"]

print("\nSample predictions (first 10 test flowers):")
print(preview.head(10).to_string(index=False))

# ---------------------------------------------------------
# Step 5: Save the model and predictions for Task 6
# ---------------------------------------------------------
joblib.dump(knn, "knn_unscaled_model.pkl")

results_df = test_df[feature_cols + ["species_id"]].copy()
results_df["predicted_id"] = y_pred
results_df.to_csv("knn_unscaled_predictions.csv", index=False)

with open("unscaled_accuracy.txt", "w") as f:
    f.write(str(accuracy))

print("\nModel saved to 'knn_unscaled_model.pkl'.")
print("Predictions saved to 'knn_unscaled_predictions.csv'.")
print("Task 6 will use this as a baseline to compare against scaled features.")
