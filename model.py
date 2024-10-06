import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import warnings

# Ignore deprecated warnings
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Display the first few rows
print(data.head())

# Descriptive statistics
print(data.describe())

# Check for any missing values
print(data.isna().sum())

# Check for duplicates
print(data.duplicated().sum())

# Plot distribution of Outcome (diabetes/no diabetes)
plt.figure(figsize=(12,6))
sns.countplot(x="Outcome", data=data)
plt.show()

# Boxplot for each feature
plt.figure(figsize=(12,12))
for i, col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=data[col])
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(data=data, hue='Outcome')
plt.show()

# Histogram of features
plt.figure(figsize=(12,12))
for i, col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
    plt.subplot(3, 3, i+1)
    sns.histplot(x=data[col], kde=True)
plt.show()

# Heatmap to show correlation between features
plt.figure(figsize=(12,12))
sns.heatmap(data.corr(), vmin=-1.0, center=0, cmap='RdBu_r', annot=True)
plt.show()

# Split the dataset into features (X) and target (y)
X = data.drop(columns="Outcome")
y = data["Outcome"]

# Standardize the feature values using StandardScaler
sc_X = StandardScaler()
X_scaled = pd.DataFrame(sc_X.fit_transform(X), columns=X.columns)

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=2)

# Initialize and train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Evaluate the model on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Training Data Accuracy:', training_data_accuracy)

# Evaluate the model on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Test Data Accuracy:', test_data_accuracy)

# Predict on new data (example input)
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the new data using the scaler
std_data = sc_X.transform(input_data_reshaped)
print("Standardized Input:", std_data)

# Predict the outcome (diabetic or not) for the new input
prediction = classifier.predict(std_data)
print("Prediction:", prediction)
if prediction[0] == 0:
    print('No Diabetic')
else:
    print('Diabetic')

# Save the trained model and the scaler using pickle
pickle.dump(classifier, open('classification-model.pkl', 'wb'))
pickle.dump(sc_X, open('scaler.pkl', 'wb'))

# Verify by loading the model and making a new prediction
loaded_model = pickle.load(open('classification-model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Use the loaded model and scaler to predict on new data
std_data = loaded_scaler.transform(input_data_reshaped)
prediction = loaded_model.predict(std_data)
print("Loaded Model Prediction:", prediction)
