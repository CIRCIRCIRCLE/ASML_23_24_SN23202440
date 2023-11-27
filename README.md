# ASML1-final-assignment

## Data set
__Dataset description__: https://medmnist.com/   
__Reference Repository:__ https://github.com/MedMNIST/MedMNIST 

## Dataset1 - PneumoniaMNIST
- These images are specifically categorized with binary labels indicating "Normal" (no pneumonia) or "Pneumonia" (presence of pneumonia).   
- `Binary classification tasks`, focusing on the automated `diagnosis of pneumonia` from standardized medical imagery.
- This dataset contains 4,708 / 524 / 624 images for Training / Validation / Test. It is going to be used for task A.    


## Dataset 2 - PathMNIST  
- This dataset includes images of nine different types of tissues, making it suitable for a `multi-class classification` task.
- `Predict survival` from colorectal cancer histology slides.
- This dataset contains 89,996 / 10,004 / 7,180 images for Training / Validation / Test. It is going to be used for task B.  

## Task:
- A: Binary classification task (using PneumoniaMNIST dataset). The objective is to classify an image onto "Normal" (no pneumonia) or "Pneumonia" (presence of pneumonia)
- B: Multi-class classification task (using PathMNIST dataset): The objective is to classi-fy an image onto 9 different types of tissues.


## Report requests: 
- report training, validation, and testing errors / accuracies, along with describe any hyper-parameter tunice process
- using several models to compare
- less than `8 pages` including the reference.
- name format: Report_AMLS_23-24 _SN12345678.pdf
- file format:
  - A
  - B
  - Datasets
    - pneumonialMNIST
    - PathMNIST
  - main.py
  - readme.md
- Readme format:   
  o	a brief description of the organization of your project;  
  o	the role of each file;  
  o	the packages required to run your code (e.g. numpy, scipy, etc.).


## Assessment: 
The assessment will predominantly concentrate on how you articulate about the choice of models, how you develop/train/validate these models, and how you report/discuss/analyse the results.  
