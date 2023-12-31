import numpy as np

def load_data(file_path):
    data = np.load(file_path)

    x_train = data['train_images']
    x_test = data['test_images']
    x_val = data['val_images']

    y_train = data['train_labels']
    y_test = data['test_labels']
    y_val = data['val_labels']

    return x_train, x_test, x_val, y_train, y_test, y_val

'''def label_img():
    img_size = 28
    class_names = ["Normal", "Pneumonia"]
'''

