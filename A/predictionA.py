import numpy as np
import matplotlib.pyplot as plt
def display_images_in_layout(images, original_labels, predicted_labels, class_names, num_columns=5, num_imgs=20):
    
    num_rows = int(num_imgs / num_columns)
    random_idx = np.random.choice(len(images), num_imgs, replace=False)

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 3 * num_rows))

    for i in range(len(random_idx)):
        row, col = divmod(i, num_columns)
        # Display image
        ax = axs[row, col]
        ax.imshow(images[random_idx[i]], cmap='gray')
        
        # Map numeric labels to class names
        original_label = class_names[int(original_labels[random_idx[i]])]
        predicted_label = class_names[int(predicted_labels[random_idx[i]])]
        title_color = 'green' if original_label == predicted_label else 'red'
        
        
        ax.set_title(f"Original: {original_label}\nPredicted: {predicted_label}", color = title_color)
        ax.axis('off')

    plt.tight_layout()
    plt.title('Prediction of Pneumonia on test set')
    plt.show()