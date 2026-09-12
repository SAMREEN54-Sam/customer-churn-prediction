import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("data/customer_churn.csv")

# Remove unnecessary customer ID column if it exists
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# Convert TotalCharges to numeric if it exists
if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

# Remove rows with missing values
data = data.dropna()

# Convert Yes/No columns into numbers
for column in data.columns:
    if data[column].dtype == "object":
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])

# Separate features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", accuracy)