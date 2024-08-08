# Project 1 - Supervised Learning

## Introduction

This project explores three supervised learning algorithms: K-Nearest Neighbors (KNN), Neural Networks (NN), and Support Vector Machines (SVM). The objective is to compare their performance on two datasets: the Diabetes dataset and the White Wine Quality dataset from UC Irvine, which can also be found on Kaggle:
- [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)

## Project Structure

There are 3 folders in this repository:
- **image-all**: Contains all images used in the analysis.
- **Diabetes**: Contains the dataset and all Jupyter Notebooks used for the Diabetes dataset analysis.
- **Wine Quality White**: Contains the dataset and all Jupyter Notebooks used for the White Wine Quality dataset analysis.

Each dataset has 3 different notebooks corresponding to each model/algorithm, which can be run separately. It might be necessary to modify the `read_csv` code path to the correct path of the dataset.

## Requirements

The project is done in Python 3.8 with the following libraries:
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `warnings`
- `sklearn`
- `optuna`

## Data and Methodology

### Diabetes Dataset
The Diabetes dataset from the National Institute of Diabetes and Digestive Kidney Diseases consists of 768 instances with 8 features. The target variable is binary (diabetes or no diabetes). Features include:
- Number of times pregnant
- Glucose concentration
- Blood pressure
- Triceps skin fold thickness
- Insulin
- BMI
- Diabetes pedigree function
- Age

The data is scaled and balanced using SMOTE due to the imbalance in the classes (35% with diabetes, 65% without).

### Wine Quality Dataset
The White Wine Quality dataset contains physicochemical test results for white wine samples. The target variable is the wine quality score, which has been transformed into three classes: low, medium, and high quality.

## Models and Results

### K-Nearest Neighbors (KNN)
- **Diabetes Dataset**: Applied SMOTE to balance the classes. Optimal performance was achieved with k=15 using Euclidean distance.
- **White Wine Quality Dataset**: Optimal performance was achieved with k=30 using Manhattan distance. Oversampling was not necessary.

### Neural Networks (NN)
- **Diabetes Dataset**: Best performance with 2 layers, 64 hidden units, ReLU activation function, learning rate of 0.04, and 20 epochs. SMOTE improved performance significantly.
- **White Wine Quality Dataset**: Best performance with 2 layers, 200 hidden units, ReLU activation function, learning rate of 0.02-0.03, and 25 epochs.

### Support Vector Machines (SVM)
- **Diabetes Dataset**: SVM with RBF kernel (C=1, gamma=0.08) provided the best results, especially after applying SMOTE.
- **White Wine Quality Dataset**: SVM with RBF kernel (C=1, gamma=0.3) provided the best results. SMOTE should be considered for further improvement.

## Conclusion

SVM with the RBF kernel emerged as the best model for both datasets, demonstrating superior prediction accuracy and generalization ability. Despite its computational cost and sensitivity to hyperparameter tuning, SVM outperformed the other models. Applying SMOTE further improved the performance on imbalanced datasets.

## Resources

- [Scikit-learn API Reference](https://scikit-learn.org)
- [Wine Quality Dataset on UCI](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)

## Author

Feedback and question, please send it to Ted Pham at trungpham89@gmail.com