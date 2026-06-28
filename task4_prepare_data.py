"""
Project 2: Flower Classifier
Task 4: Prepare the Data (Split into Train/Test)
-----------------------------------------------------------
Goal: Load the Iris dataset, explore it, separate features (measurements)
from labels (species), and split into training/testing sets.

The Iris dataset has 150 flowers, each with 4 measurements:
sepal length, sepal width, petal length, petal width (all in cm),
and a species label: 0 = setosa, 1 = versicolor, 2 = virginica.

This script saves the train/test split to CSV files so Task 5 and
Task 6 can reuse the exact same split (important — using a different
split each time would make accuracy comparisons across tasks unfair).
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Step 1: Load the Iris dataset
# ---------------------------------------------------------
iris = load_iris()

# Build a readable DataFrame to explore (features + species name)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species_id"] = iris.target
df["species_name"] = df["species_id"].map(
    {i: name for i, name in enumerate(iris.target_names)}
)

# ---------------------------------------------------------
# Step 2: Explore the data
# ---------------------------------------------------------
print("First 5 rows:")
print(df.head())

print("\nDataset shape (rows, columns):", df.shape)

print("\nHow many flowers per species:")
print(df["species_name"].value_counts())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nFeature statistics:")
print(df.describe())

# ---------------------------------------------------------
# Step 3: Separate features (X) from labels (y)
# ---------------------------------------------------------
X = iris.data          # 4 measurements per flower
y = iris.target        # species id (0, 1, or 2)

# ---------------------------------------------------------
# Step 4: Split into training (70%) and testing (30%) sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

print(f"\nTotal samples: {len(X)}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ---------------------------------------------------------
# Step 5: Save the split so Task 5 and Task 6 can reuse it
# ---------------------------------------------------------
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

train_df = pd.DataFrame(X_train, columns=feature_cols)
train_df["species_id"] = y_train
train_df.to_csv("iris_train.csv", index=False)

test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df["species_id"] = y_test
test_df.to_csv("iris_test.csv", index=False)

# Save species names too (0/1/2 -> setosa/versicolor/virginica)
species_map = pd.DataFrame({
    "species_id": list(range(len(iris.target_names))),
    "species_name": iris.target_names
})
species_map.to_csv("species_names.csv", index=False)

print("\nSaved 'iris_train.csv', 'iris_test.csv', and 'species_names.csv'.")
print("Task 5 will load these to train the k-NN classifier.")
