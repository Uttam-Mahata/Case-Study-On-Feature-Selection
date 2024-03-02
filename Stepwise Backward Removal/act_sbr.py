import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Load the dataset
data = pd.read_csv('act_dataset.csv')

# Split features and target variable
X = data.drop('cid', axis=1)
y = data['cid']

# Step 2: Apply Stepwise Backward Removal feature selection algorithm
estimator = SVC(kernel="linear")
selector = RFE(estimator)
selector = selector.fit(X, y)
X_reduced = X.loc[:, selector.support_]

# Step 3: Save the reduced dataset to a new CSV file
X_reduced.to_csv('act_reduced_dataset_sbr.csv', index=False)

# Step 4: Train various classifiers on both original and reduced datasets
classifiers = {
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier()
}

results = []

# Step 5: Evaluate the performance of each classifier
for name, clf in classifiers.items():
    # Original dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_orig = accuracy_score(y_test, y_pred)
    precision_orig = precision_score(y_test, y_pred)
    recall_orig = recall_score(y_test, y_pred)
    f1_orig = f1_score(y_test, y_pred)

    # Reduced dataset
    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    clf.fit(X_train_reduced, y_train_reduced)
    y_pred_reduced = clf.predict(X_test_reduced)
    accuracy_reduced = accuracy_score(y_test_reduced, y_pred_reduced)
    precision_reduced = precision_score(y_test_reduced, y_pred_reduced)
    recall_reduced = recall_score(y_test_reduced, y_pred_reduced)
    f1_reduced = f1_score(y_test_reduced, y_pred_reduced)

    results.append({
        'Classifier': name,
        'Accuracy_Original_Dataset': accuracy_orig,
        'Accuracy_Reduced_Dataset': accuracy_reduced,
        'Precision_Original_Dataset': precision_orig,
        'Precision_Reduced_Dataset': precision_reduced,
        'Recall_Original_Dataset': recall_orig,
        'Recall_Reduced_Dataset': recall_reduced,
        'F1_Score_Original_Dataset': f1_orig,
        'F1_Score_Reduced_Dataset': f1_reduced
    })
#print features with indices
print(selector.support_)
# Step 6: Organize the results in a tabular format
results_df = pd.DataFrame(results)
print(results_df)
