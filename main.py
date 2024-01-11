from A.load_data import load_data
from A.preprocessing import zero_centered_normalization, data_flatten, data_norm_cnn, standard_normalization, one_hot_encoding
from A.visualization import visualize_tsne_A, visualize_tsne_B, plot_confusion_matrix
from A.LR import logistic_regression_classification
from A.KNN import knn_classification, plot_knn_error_rates
from A.CNN import create_and_compile_model, apply_data_augmentation, train_model_with_augmentation, plot_training_history
from A.predictionA import display_images_in_layout
from B.SVM import get_n_components_for_explained_variance, svm_classifier_with_pca, test_images
from B.CNN_B import create_and_compile_modelB, data_augmentation, train_model_with_augmentationB
from B.predictionB import display_predictions
import numpy as np

def A_pneumonia_detection():
    # load data
    x_train, x_test, x_val, y_train, y_test, y_val = load_data('../datasets/pneumoniamnist.npz')
    img_size = 28
    class_names = ["Normal", "Pneumonia"]

    # preprocessing
    x_train1, x_test1, x_val1 = zero_centered_normalization(x_train, x_test, x_val)
    x_train_flat, x_test_flat, x_val_flat = data_flatten(x_train1, x_test1, x_val1)
    x_train2, x_test2, x_val2 = data_norm_cnn(x_train1, x_test1, x_val1, img_size)

    # data distribution overview
    # This may take 10s, if you don't want to check just tap in N
    overview = input('Do you want to check the distribution overview? Y/N:  ')
    if overview == 'Y':
        visualize_tsne_A(x_train, y_train, class_names)

    # choose the method
    method = input("Choose a method (LR or KNN or CNN): ")
    
    if method == 'LR':
        # call logistic regression
        predicted_labels, accuracy_lr, classification_rep_lr, confusion_mat_lr = logistic_regression_classification(x_train_flat, y_train, x_test_flat, y_test)
        plot_confusion_matrix(confusion_mat_lr, class_names, "Confusion Matrix (Logistic Regression)")
        display_images_in_layout(x_test1, y_test, predicted_labels, class_names, num_columns=5, num_imgs=20)

    elif method == 'KNN':
        # choose the k value based on error rates
        k_values = [k for k in range(1, 12)]
        plot_knn_error_rates(x_train_flat, y_train, x_val_flat, y_val, k_values)
        # based on the fig, manually select the value
        # close the fig, continue to the next step
        k_value = int(input("Enter the number of neighbors (K): "))   # 8 performs best
        predicted_labels, accuracy_knn, classification_rep_knn, confusion_mat_knn = knn_classification(x_train_flat, y_train, x_test_flat, y_test, k_value)
        plot_confusion_matrix(confusion_mat_knn, class_names, "Confusion Matrix (K Nearest Neigbors)")
        display_images_in_layout(x_test1, y_test, predicted_labels, class_names, num_columns=5, num_imgs=20)

    elif method == 'CNN':
        #call CNN
        model = create_and_compile_model()
        augmented_data = apply_data_augmentation(x_train2, y_train)
        history = train_model_with_augmentation(model, augmented_data, x_val2, y_val)
        predictions = model.predict(x_test2)
        predicted_labels = np.round(predictions).astype(int)        

        print('Test result--------------------------------------------------')
        print("Test loss - " , model.evaluate(x_test2,y_test)[0])
        print("Test Accuracy - " , model.evaluate(x_test2,y_test)[1]*100 , "%")

        print('training analysis(shown in fig)------------------------------------------------')
        plot_training_history(history)
        display_images_in_layout(x_test2, y_test, predicted_labels, class_names, num_columns=5, num_imgs=20)
    else:
        print("Invalid method. Please choose LR or KNN.")

def B_path_multi_classification():
    # load data
    x_train, x_test, x_val, y_train, y_test, y_val = load_data('../datasets/pathmnist.npz')
    img_size = 28
    class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

    # preprocessing
    x_train1, x_test1, x_val1 = standard_normalization(x_train, x_test, x_val)
    x_train_flat, x_test_flat, x_val_flat = data_flatten(x_train1, x_test1, x_val1)
    x_train2, x_test2, x_val2 = data_norm_cnn(x_train1, x_test1, x_val1, img_size)
    y_train_one_hot, y_test_one_hot, y_val_one_hot = one_hot_encoding(y_train, y_test, y_val)

    # data distribution overview
    # This may take 8 min!!!!, if you don't want to check just tap in N
    print('Warning: the distribution overview here may take 8min to run, skip it tap: N')
    overview = input('Do you want to check the distribution overview? Y/N:  ')
    if overview == 'Y':
        visualize_tsne_B(x_train, y_train, class_names)

    #choose the method
    method = input("Choose a method (SVM or CNN): ")

    if method == 'SVM':
        ## The training process takes a very long time, this cell may take 40 min!!!!!!!!!
        ## C=10 is the optimal parameter(details can be shown in jupyter notebook)
        n_components_90_percent = get_n_components_for_explained_variance(x_train_flat, threshold_percent=90)
        clf_pca = svm_classifier_with_pca(x_train_flat, y_train.ravel(), C = [10], n_components_pca = n_components_90_percent)
        predicted_labels, accuracy, classification_rep_svm, confusion_mat_svm = test_images(clf_pca, x_test_flat, y_test.ravel())
        plot_confusion_matrix(confusion_mat_svm, class_names, "Confusion Matrix (SVM)")
        display_predictions(x_test1, y_test, predicted_labels, class_names, 8, 40)

    elif method == 'CNN':
        # about 30mins to run
        model = create_and_compile_modelB()
        augmented_data = data_augmentation(x_train1, y_train_one_hot)
        history = train_model_with_augmentationB(model, augmented_data, x_val1, y_val_one_hot)
        history = train_model_with_augmentationB(model, augmented_data, x_val1, y_val_one_hot)

        predictions = model.predict(x_test1)
        predicted_labels = np.argmax(predictions, axis=1)  # Convert one-hot encoded predictions to categorical
        display_predictions(x_test1, y_test, predicted_labels, class_names, 8, 40)
        

        print('Test result--------------------------------------------------')
        print("Test loss - " , model.evaluate(x_test1,y_test_one_hot)[0])
        print("Test Accuracy - " , model.evaluate(x_test1,y_test_one_hot)[1]*100 , "%")

        print('training analysis(shown in fig)------------------------------------------------')
        plot_training_history(history)
    else:
        print('Invaild method. Please choose SVM or CNN')
        

def main():
    #choose a task
    T = input('Choose a Task (A or B):')
    if T == 'A':
        A_pneumonia_detection()
    elif T == 'B':
        B_path_multi_classification()
    else:
        print('Invalid. Please choose A or B')
    

if __name__ == "__main__":
    main()
