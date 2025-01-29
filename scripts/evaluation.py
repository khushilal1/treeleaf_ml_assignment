# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np
# import os


# def evaluate_model(model, X_test, y_test):
#     """
#     Evaluates a trained model using accuracy, precision, recall, and F1-score.
#     Handles both sklearn models (Logistic Regression, Random Forest) and Keras models (Neural Network).
#     Returns the evaluation metrics as a dictionary.
#     """
#     # Make predictions with the model
#     y_pred = model.predict(X_test)

#     # Handle model output types for neural networks (binary or multi-class classification)
#     if isinstance(model, tf.keras.Model):
#         if y_pred.shape[1] == 1:  # Binary classification (one output node)
#             y_pred = (y_pred > 0.5).astype(int)
#         else:  # Multi-class classification
#             y_pred = np.argmax(y_pred, axis=1)
#     else:
#         # For sklearn models, handle the predictions and probabilities
#         if hasattr(model, "predict_proba"):
#             y_pred = (y_pred > 0.5).astype(int)  # For binary classification
#         else:
#             y_pred = np.argmax(y_pred, axis=1) if len(
#                 y_pred.shape) > 1 else y_pred

#     # Calculate the evaluation metrics
#     metrics = {
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Precision": precision_score(y_test, y_pred, average='weighted'),
#         "Recall": recall_score(y_test, y_pred, average='weighted'),
#         "F1 Score": f1_score(y_test, y_pred, average='weighted')
#     }

#     return metrics


# if __name__ == "__main__":
#     from preprocess import preprocess_data
#     from sklearn.model_selection import train_test_split

#     # Preprocess data
#     train_data = preprocess_data("../data/train.csv")
#     X = train_data.drop("Survived", axis=1)
#     y = train_data["Survived"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42)

#     # Dictionary to hold models
#     models = {
#         "Logistic Regression": joblib.load("../models/logistic_regression.pkl"),
#         "Random Forest": joblib.load("../models/random_forest.pkl"),
#         # Keras model
#         "Neural Network": load_model("../models/neural_network.h5")
#     }

#     # Ensure the 'results' directory exists
#     if not os.path.exists('../results'):
#         os.makedirs('../results')


# all_metrics = []
# # Evaluate each model and collect results
# for name, model in models.items():
#     print(f"Evaluating {name}")
#     metrics = evaluate_model(model, X_test, y_test)
#     metrics["Model"] = name  # Add model name to the result

#     # Print the metrics to console
#     print(f"Metrics for {name}:")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value}")

#         # Append metrics to the list
#     all_metrics.append(metrics)
#  # Save all results in a single CSV file
# results_df = pd.DataFrame(all_metrics)
# results_df.to_csv('../results/model_evaluation_results.csv', index=False)

# print("All results saved to 'model_evaluation_results.csv' in the 'results' folder.")
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model using accuracy, precision, recall, and F1-score.
    Handles both sklearn models (Logistic Regression, Random Forest) and Keras models (Neural Network).
    Returns the evaluation metrics as a dictionary.
    """
    # Make predictions with the model
    y_pred = model.predict(X_test)

    # Handle model output types for neural networks (binary or multi-class classification)
    if isinstance(model, tf.keras.Model):
        if y_pred.shape[1] == 1:  # Binary classification (one output node)
            y_pred = (y_pred > 0.5).astype(int)
        else:  # Multi-class classification
            y_pred = np.argmax(y_pred, axis=1)
    else:
        # For sklearn models, handle the predictions and probabilities
        if hasattr(model, "predict_proba"):
            y_pred = (y_pred > 0.5).astype(int)  # For binary classification
        else:
            y_pred = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred

    # Calculate the evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    }

    return metrics


def main():
    from preprocess import preprocess_data
    from sklearn.model_selection import train_test_split

    # Preprocess data
    train_data = preprocess_data("../data/train.csv")
    X = train_data.drop("Survived", axis=1)
    y = train_data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Dictionary to hold models
    models = {
        "Logistic Regression": joblib.load("../models/logistic_regression.pkl"),
        "Random Forest": joblib.load("../models/random_forest.pkl"),
        # Keras model
        "Neural Network": load_model("../models/neural_network.h5")
    }

    # Ensure the 'results' directory exists
    if not os.path.exists('../results'):
        os.makedirs('../results')

    # List to store evaluation results for all models
    all_metrics = []

    # Evaluate each model and collect results
    for name, model in models.items():
        print(f"Evaluating {name}")
        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name  # Add model name to the result

        # Print the metrics to console
        print(f"Metrics for {name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        # Append metrics to the list
        all_metrics.append(metrics)

    # Save all results in a single CSV file
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv('../results/model_evaluation_results.csv', index=False)

    print("All results saved to 'model_evaluation_results.csv' in the 'results' folder.")


if __name__ == "__main__":
    main()
