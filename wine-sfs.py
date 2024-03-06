import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import seaborn as sns

# Load the dataset from CSV
data = pd.read_csv('wine_dataset.csv')

# Apply Stepwise Forward Selection algorithm
X = data.drop(columns=['class'])  # Adjust 'target_column' with your target variable
y = data['class']
sfs = SequentialFeatureSelector(RandomForestClassifier(), n_features_to_select='auto', direction='forward', n_jobs=-1)
sfs.fit(X, y)
selected_features = X.columns[sfs.get_support()]

# Save the reduced dataset to a new CSV file
reduced_data = data[selected_features]
reduced_data.to_csv('reduced-sfs_wine_dataset.csv', index=False)

# Split the datasets into training and testing sets
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(reduced_data, y, test_size=0.2, stratify=y, random_state=42)

# Define classifiers and their respective parameter grids for GridSearchCV
classifiers = {
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
    },
    'SVM': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    }
}

# Evaluate the performance of classifiers
results = []
for fsa, X_train, X_test, y_train, y_test in [('Original', X_train_orig, X_test_orig, y_train_orig, y_test_orig),
                                               ('Reduced', X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced)]:
    for clf_name, clf in classifiers.items():
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', GridSearchCV(clf['model'], clf['params'], cv=skf, scoring=make_scorer(accuracy_score)))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append([fsa, clf_name, accuracy, precision, recall, f1])
        dump(pipeline, f'{fsa}_{clf_name}_model.joblib')

# Compile the results into a tabular format
results_df = pd.DataFrame(results, columns=['FSA', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

#save the results to a CSV file
results_df.to_csv('results.csv', index=False)

# Visualize the results in graph
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
ax = sns.barplot(x="Classifier", y="Accuracy", hue="FSA", data=results_df)
ax.set_title('Classifier Performance Comparison')
plt.show()
