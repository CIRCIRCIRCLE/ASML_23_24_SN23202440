from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def knn_classification(x_train, y_train, x_test, y_test, n_neighbors):
    
    # Fit KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(x_train, y_train.ravel())
    predictions = knn_classifier.predict(x_test)

    # Evaluate the performance
    accuracy = accuracy_score(y_test.ravel(), predictions)
    classification_rep = classification_report(y_test.ravel(), predictions)
    confusion_mat = confusion_matrix(y_test.ravel(), predictions)

    # Display results
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('\nClassification Report:\n', classification_rep)
    print('\nConfusion Matrix:\n', confusion_mat)

    return accuracy, classification_rep, confusion_mat

def knn_error_rates(x_train, y_train, x_val, y_val, k_values):
    train_errors = []
    val_errors = []

    for k in k_values:
        # Train KNN model
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(x_train, y_train.ravel())

        # Calculate error rates
        train_preds = knn_classifier.predict(x_train)
        val_preds = knn_classifier.predict(x_val)

        train_error = 1 - accuracy_score(y_train.ravel(), train_preds)
        val_error = 1 - accuracy_score(y_val.ravel(), val_preds)

        # Append errors to lists
        train_errors.append(train_error)
        val_errors.append(val_error)

    return train_errors, val_errors

def plot_knn_error_rates(x_train, y_train, x_val, y_val, k_values):

    train_errors, val_errors = knn_error_rates(x_train, y_train, x_val, y_val, k_values)

    plt.figure(figsize=(10, 6))
    
    smooth_k_values = np.linspace(min(k_values), max(k_values), 100)

    # Interpolate errors for the smooth curve
    smooth_train_errors = np.interp(smooth_k_values, k_values, train_errors)
    smooth_val_errors = np.interp(smooth_k_values, k_values, val_errors)

    plt.plot(smooth_k_values, smooth_train_errors, label='Training Error',  color='#03608C')
    plt.plot(smooth_k_values, smooth_val_errors, label='Validation Error', color='#9f1f31')

    plt.title('Training and Validation Error Rates for Different K Values')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(False)
    plt.show()    