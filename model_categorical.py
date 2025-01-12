import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# Load data
print("Loading data...")
data = pd.read_csv('updated_final.csv')
print("Data loaded successfully!")

# Check column names
print("Available columns:", data.columns)

# Features and target
features = ['sizex', 'sizey', 'total_size', 'goalx', 'goaly', 'startx', 'starty', 'loop', 'pattern']
target = 'Best-Effi'

# Encode target column (categorical to numeric)
print("Encoding target column...")
encoder = LabelEncoder()
data[target] = encoder.fit_transform(data[target])

# Encode 'pattern' column
pattern_encoder = LabelEncoder().fit(data['pattern'])
data['pattern'] = pattern_encoder.transform(data['pattern'])

# Save the label encoder for target and pattern
dump(encoder, 'label_encoder.joblib')
dump(pattern_encoder, 'pattern_encoder.joblib')

# Save mapping for reverse lookup
print("Class mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# Features and target selection
X = data[features]
y = data[target]

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete!")

# Feature scaling
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
dump(scaler, 'scaler.joblib')
print("Feature scaling complete!")

# Model training
print("Training model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete!")

# Save the trained model
print("Saving the model...")
dump(model, 'best_efficiency_classifier.joblib')
print("Model saved successfully!")

# Model evaluation
print("Evaluating model...")
y_pred = model.predict(X_test_scaled)

# Decode predictions back to original class labels
predicted_classes = encoder.inverse_transform(y_pred)
true_classes = encoder.inverse_transform(y_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, zero_division=1))

# Example prediction with new data
print("Making a sample prediction...")
sample_input = [[10, 10, 100, 9, 9, 0, 0, 1, 3]]  # Example data point
sample_scaled = scaler.transform(sample_input)
sample_prediction = model.predict(sample_scaled)
sample_prediction_class = encoder.inverse_transform(sample_prediction)
print(f"Predicted algorithm: {sample_prediction_class[0]}")