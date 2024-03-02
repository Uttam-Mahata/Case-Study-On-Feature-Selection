import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv("wine_dataset.csv")  # Update with your dataset file path
X = data.drop(columns=['class'])  # Features
y = data['class']  # Target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier()
}

# Define evaluation metrics
metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F-score": f1_score
}

# Perform feature selection and evaluate classifiers
print("Feature Selection Method: Stepwise Forward Selection")
for classifier_name, classifier in classifiers.items():
    print(f"\nClassifier: {classifier_name}")

    # Initialize Sequential Feature Selector
    sfs = SequentialFeatureSelector(classifier, direction='forward', scoring='accuracy', cv=5)

    # Perform feature selection
    sfs.fit(X_train, y_train)

    # Get selected features
    selected_features = list(X.columns[sfs.get_support(indices=True)])
    print(f"Selected Features: {selected_features}")

    # Train the classifier with selected features
    classifier.fit(X_train[selected_features], y_train)

    # Make predictions
    y_pred = classifier.predict(X_test[selected_features])

    # Evaluate performance
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_test, y_pred)
        print(f"{metric_name}: {score:.4f}")

    print("\n==============================\n")
