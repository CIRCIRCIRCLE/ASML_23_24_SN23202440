from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def get_n_components_for_explained_variance(x_train_flat, threshold_percent=90):

    pca = PCA().fit(x_train_flat)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    threshold_value = threshold_percent / 100.0
    n_components_threshold = np.argmax(cumulative_variance >= threshold_value) + 1

    print(f'Number of components for {threshold_percent}% explained variance: {n_components_threshold}')

    return n_components_threshold


def svm_classifier_with_pca(x_train, y_train, C, n_components_pca):
    # Create a pipeline with PCA and SVM
    pipeline = Pipeline([
        ('pca', PCA(n_components=n_components_pca)),
        ('svm', SVC())
    ])

    # Define the parameter grid for SVM
    param_grid = {
        'svm__C': C,
        'svm__gamma': ['scale']
    }

    # Creating the SVM classifier with GridSearchCV
    clf = GridSearchCV(pipeline, param_grid, verbose=3, cv=2)
    clf.fit(x_train.reshape(x_train.shape[0], -1), y_train)

    print("Best parameters found by grid search:")
    print(clf.best_params_)
    print("\nBest estimator found by grid search:")
    print(clf.best_estimator_)

    return clf

def test_images(clf, x_test, y_test):
    predictions = clf.predict(x_test)
    
    # Evaluate the performance
    accuracy = accuracy_score(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)
    confusion_mat = confusion_matrix(y_test, predictions)

    # Display results
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('\nClassification Report:\n', classification_rep)
    print('\nConfusion Matrix:\n', confusion_mat)

    return predictions, accuracy, classification_rep, confusion_mat


