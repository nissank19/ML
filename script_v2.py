import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CLASS_COLUMN_INDEX = 60
TEST_SPLIT_FACTOR = 0.2

# Read data from CSV
sonar_data = pd.read_csv("data/sonar_data.csv", header=None)
sonar_data.groupby(CLASS_COLUMN_INDEX).mean()

# Extract feature (x) and class / label (y) vectors
x = sonar_data.drop(columns=CLASS_COLUMN_INDEX, axis=1)
y = sonar_data[CLASS_COLUMN_INDEX]

# Split data into testing and training sets
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=TEST_SPLIT_FACTOR, stratify=y, random_state=1
)

# Initialize and fit logistic regression model on data
model = LogisticRegression()
model.fit(X_train, Y_train)

# Get training accuracy
y_train = model.predict(X_train)
training_accuracy = accuracy_score(y_train, Y_train)

# Get testing accuracy
y_test = model.predict(X_test)
testing_accuracy = accuracy_score(y_test, Y_test)

# Output model statistics
print(f"Training Accuracy: {training_accuracy*100:.2f}%")
print(f"Testing Accuracy: {testing_accuracy*100:.2f}%")
print(f"Training Examples: {x.shape[0]} examples.")
print(f"Feature Count: {x.shape[1]} features.")

# Plot and save model diagram (only first feature)
X_plot = X_train[0].sort_values()
plt.scatter(X_plot, Y_train, color="blue", alpha=0.3)
X_pred = model.predict_proba(X_train)[:, 0]
X_pred.sort()
plt.plot(X_plot, X_pred, color="red", alpha=0.5)
plt.savefig("plot.png")
