import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from skimage.feature import hog
from skimage import exposure

# Load dataset MNIST
digits = datasets.load_digits()

# HOG Feature Extraction
def extract_hog_features(images):
    features = []
    for image in images:
        fd, hog_image = hog(image.reshape((8, 8)), orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=True)
        features.append(fd)
    return np.array(features)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Feature extraction using HOG
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# Standardize features
scaler = StandardScaler().fit(X_train_hog)
X_train_hog = scaler.transform(X_train_hog)
X_test_hog = scaler.transform(X_test_hog)

# Train SVM
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_hog, y_train)
# Predict
y_pred = svm_model.predict(X_test_hog)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Display results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)