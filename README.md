# ASML_23_24_SN23202440 Classification Project
## Description
```
Domain        :Machine Learning, Computer Vision  
Sub-Domain    :Supervised Learning, unsupervised Learning, Image Processing, Image Classification, Medical Image
Techniques    :Logistic Regression(LR), K Nearest Neighbors(KNN), Support Vector Machine(SVM), Convolutional Neural Networks(CNN)  
```   
The project contains 2 parts:      
`Task A: Binary Classification for Pneumonia Detection`    
- In the Pneumonia Detection task, the objective is to classify an image onto “Normal” or “Pneumonia”. And the employed dataset contains 4,708/524/624 images for Training/Validation/Test, with each image measuring 28x28 pixels, compressed the classic MINIST dataset.   
- I implemented 3 algorithms to compare the performance of the classification, including `Logistic Regression(LR)`, `K Nearest Neighbors(KNN)` and `Convolutional Neural Networks(CNN)`. Introducing the `Gaussian Blur` technique to emphasize the features.   
    
`Task B: Multi-classification for Colorectal tissue classification`  
- This task involves categorizing colorectal tissue into 9 classes. To make the data more manageable, the original images, initially sized at 3 × 224 × 224, have been resized to 3 × 28 × 28. The employed dataset contains 89,996 / 10,004 / 7,180 images for Training / Validation / Test.   
- This task aimed at categorizing the homogeneous tissue regions into 9 tissue classes. I used 2 algorithms to do the classification which are `Support Vector Machines(SVM)` and `CNN`. Introducing `Principle Components Analysis (PCA)` to reduce redundant features.   

## Dataset Details
The dataset is from medmnist which can be found here: `https://medmnist.com/`, I used the .npz version which can be downloaded here: `https://zenodo.org/records/6496656`

## Tools/Libraries
```
Languages    : Python
Tools/IDE    : VS Code, Jupyter Notebook
Libraries    : NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, Tensorflow, Keras, Random, PIL, Collections, visualkeras
```

## File explanation  
- The full codes and results are displayed in the jupyter notebook (A: 'Pneumonia Detection.ipynb', B: 'Colorectal tissue classification.ipynb')  
- `main.py` is the entrance of the project containing two functions `A_pneumonia_detection` and `B_path_multi_classification` corresponding to 2 tasks. Run the 'main.py' and enter the function you want to call in the command line. Each function has default parameters which can be changed in the corresponding functions.
  ```
  Choose a Task (A or B):
    if you choose `A`: Choose a method (LR or KNN or CNN)
    if you choose `B`: Choose a method (SVM or CNN)  ##for Task B it may run a long time: approximate 40min for SVM, 20min for CNN
  ```
- A file contains load_data.py, preprocessing.py, visualization.py which will also be used in B task. And also contains ML functions: LR.py, KNN.py, CNN.py which will be called in the main function.
- B file contains SVM.py, CNN_B.py which will be called in the main function.

## Performance
### Task A:
|Method|Precision|Recall|Accuracy|
|------|---------|------|--------|
|LR| 79.75| 97.95| 83.17|
|KNN| 80.85| 97.44| 83.97|
|CNN| 90.78| 95.90| 91.19|

### Task B:
<img src="/figs/B_RES.png?raw=true" width="500" />

## Model Predictions
### Task A:
<img src="/figs/A_PRED.png?raw=true" width="800" />

### Task B:
<img src="/figs/B_PRED.png?raw=true" width="800" />

## Confusion Matrix
### Task A:
<p float="center">
  <img src="/figs/A_conf_LR.png?raw=true" width="300" />
  <img src="/figs/A_CONF_KNN.png?raw=true" width="300" /> 
  <img src="/figs/A_CONF_CNN.png?raw=true" width="300" />
</p>

### Task B:
<p float="center">
  <img src="/figs/B_CONF_SVM.png?raw=true" width="400" />
  <img src="/figs/B_CONF_CNN.png?raw=true" width="400" /> 
</p>    

