# **Model Evaluation and Result Saving**

## **Overview**
This project evaluates machine learning models using standard classification metrics, including Accuracy, Precision, Recall, and F1-Score. It supports both scikit-learn models (such as Logistic Regression and Random Forest) and Keras models (e.g., Neural Networks). After evaluation, the results are printed to the console and saved in CSV format for easy comparison.

## **Technologies Used**
- **Python**: Programming language used for the project.
- **pandas**: Data manipulation and saving results in a DataFrame format.
- **scikit-learn**: For machine learning models and evaluation metrics.
- **tensorflow / Keras**: For evaluating Keras-based neural network models.
- **numpy**: For numerical operations and model predictions.
- **joblib**: For loading pre-trained scikit-learn models.
- **os**: For handling file system operations like directory creation and checking.

## **Project Structure**
The project follows this folder structure:
/project_folder ├── /data │ └── train.csv # Training data (e.g., Titanic dataset) ├── /models │ ├── logistic_regression.pkl # Trained Logistic Regression model │ ├── random_forest.pkl # Trained Random Forest model (optional) │ ├── neural_network.h5 # Trained Keras Neural Network model (optional) ├── /results │ └── logistic_regression_evaluation_results.csv # CSV file for evaluation results ├── preprocess.py # Data preprocessing script ├── model_evaluation.py # Main script for evaluation and result saving ├── train.py # Script to train and save models └── README.md # Project documentation


### **Explanation of Folders and Files:**
- **/data**: Contains the training dataset (e.g., `train.csv`).
- **/models**: Contains the pre-trained machine learning models (Logistic Regression, Random Forest, Neural Network).
- **/results**: Stores the evaluation results (in CSV format) for each model.
- **preprocess.py**: Preprocessing script for cleaning, encoding, and splitting the data.
- **model_evaluation.py**: Script that evaluates models and saves the results.
- **train.py**: Script used to train and save machine learning models.
- **README.md**: This documentation file.

## **How to Use**

### **Step 1: Prepare the Data**
Ensure the `train.csv` file is available in the `/data` folder. The file should contain both the features and target variable (`Survived` in the Titanic dataset example).

### **Step 2: Train Models**
Before running the evaluation script, you need to train your machine learning models (Logistic Regression, Random Forest, Neural Network). Use the `train.py` script to train and save models:

```bash
python train.py
```
Ensure the trained models are saved in the /models folder (e.g., logistic_regression.pkl, random_forest.pkl, neural_network.h5).

### **Step 3: Run the Evaluation**
Once the models are saved, run the evaluation script to evaluate the models and save the results:

```bash
python model_evaluation.py
```
### **Step 4: Check the Results**

After the evaluation, the results will be saved in the /results folder as CSV files. You can open the evaluation_results.csv file to view the performance metrics for each model.

### **Evaluation Metrics**
The following metrics are used to evaluate the models:

**Accuracy**: Proportion of correctly predicted instances.<br>
**Precision**: Proportion of positive predictions that are actually positive.<brr>
**Recall**: Proportion of actual positive instances that are identified.<br>
**F1-Score**: Harmonic mean of Precision and Recall, providing a balanced measure.<br>

### Example Output ###

### Evaluating Logistic Regression

Accuracy: 0.8100558659217877<br>
Precision: 0.8091625969233472<br>
Recall: 0.8100558659217877<br>
F1 Score: 0.8091929317716401<br>
Model: Logistic Regression


### Evaluating Random Forest
Metrics for Random Forest:
Accuracy: 0.8212290502793296<br>
Precision: 0.8204500025626569<br>
Recall: 0.8212290502793296<br>
F1 Score: 0.8204168769615436<br>
Model: Random Forest

### Evaluating Neural Network
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
Metrics for Neural Network:<br>
Accuracy: 0.7094972067039106<br>
Precision: 0.7409543667206816<br>
Recall: 0.7094972067039106<br>
F1 Score: 0.6789406165942479<br>
Model: Neural Network
Results saved to the 'results' folder.

