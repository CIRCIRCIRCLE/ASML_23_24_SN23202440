import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

def visualize_tsne_A(image_data, labels, class_names):
    image_data_flatten = image_data.reshape(image_data.shape[0], -1)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(image_data_flatten)

    tsne_df = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
    
    # Convert NumPy array to Pandas Series, flatten, and map numeric labels to class names
    tsne_df['Label'] = pd.Series(labels.flatten()).map({0: class_names[0], 1: class_names[1]})
    
    # Define custom colors for each class
    color_dict = {class_names[0]: '#9f1f31', class_names[1]: '#03608C'}

    # Create a scatter plot using seaborn with custom colors
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Label', data=tsne_df, palette=color_dict, alpha=0.7)

    plt.title("t-SNE Visualization of Image Clustering")
    plt.show()

def visualize_tsne_B(image_data, labels, class_names):
    image_data_flatten = image_data.reshape(image_data.shape[0], -1)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(image_data_flatten)

    tsne_df = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['Label'] = labels.astype(str)  # Convert labels to string for better handling in seaborn
    
    # Create a scatter plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Label', data=tsne_df, palette='tab10', alpha=0.7)

    plt.title("t-SNE Visualization of Image Clustering")
    plt.show()



def plot_confusion_matrix(confusion_mat, class_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(title)
    plt.show()

