import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

warnings.filterwarnings("ignore")

data = {
    'CustormerID': range(1, 101),
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 10,
    'MonthlyCharge': [20, 30, 40, 50, 60, 70, 80, 90, 100, 110] * 10,
    'customerserviceCalls': [1, 2, 3, 4, 0, 1, 2, 3, 4, 0] * 10,
    'Churn': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'] * 10
}
df = pd.DataFrame(data)
# Define the target variable y and features X
X = df[['age', 'MonthlyCharge', 'customerserviceCalls']]
y = df['Churn']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()
# Train the classifier
clf.fit(X_train, y_train)
# Make predictions
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')
# Visualize the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Churn', 'Churn'])
plt.title('Decision Tree for predicting Customer Churn')
plt.show()


