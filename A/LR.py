from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def logistic_regression_classification(x_train, y_train, x_test, y_test):
    
    # Initialize Logistic Regression classifier with L2 regularization and the 'lbfgs' solver
    logistic_reg_classifier = LogisticRegression(solver='lbfgs', penalty='l2')
    
    # Fit LR classifier
    logistic_reg_classifier.fit(x_train, y_train)
    predictions = logistic_reg_classifier.predict(x_test)

    # Evaluate the performance
    accuracy = accuracy_score(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)
    confusion_mat = confusion_matrix(y_test, predictions)

    # Display results
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('\nClassification Report:\n', classification_rep)
    print('\nConfusion Matrix:\n', confusion_mat)

    return accuracy, classification_rep, confusion_mat