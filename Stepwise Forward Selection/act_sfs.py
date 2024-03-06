import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Load the dataset from CSV
data = pd.read_csv('act_dataset.csv')

#Select the continuous feature for normalization
# continuous_features = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd820']
# #Normalize the continuous features
# for feature in continuous_features:
#     data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()



# Step 2: Apply Stepwise Forward Selection algorithm
X = data.drop(columns=['cid'])  # Adjust 'target_column' with your target variable
y = data['cid']
sfs = SequentialFeatureSelector(RandomForestClassifier(), n_features_to_select='auto',scoring='accuracy', direction='forward', n_jobs=-1)
sfs.fit(X, y)
selected_features = X.columns[sfs.get_support()]

# Step 3: Save the reduced dataset to a new CSV file
reduced_data = data[selected_features]
reduced_data.to_csv('reduced-sfs_act_dataset.csv', index=False)

# Step 4: Split the datasets into training and testing sets
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(reduced_data, y, test_size=0.2, random_state=42)

# Step 5: Train various classifiers on both datasets
classifiers = {
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# Step 6: Evaluate the performance of classifiers
results = []
for fsa, X_train, X_test, y_train, y_test in [('Original', X_train_orig, X_test_orig, y_train_orig, y_test_orig),
                                               ('Reduced', X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced)]:
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append([fsa, clf_name, accuracy, precision, recall, f1])

#Print selected features with index
# print(selected_features)
# Step 7: Compile the results into a tabular format
results_df = pd.DataFrame(results, columns=['FSA', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
print(results_df)


# Step 8: Visualize the results in graph
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
ax = sns.barplot(x="Classifier", y="Accuracy", hue="FSA", data=results_df)
ax.set_title('Classifier Performance Comparison')
plt.show()


