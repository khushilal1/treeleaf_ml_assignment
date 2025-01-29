from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split


#for logistics regression
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, "../models/logistic_regression.pkl")





#for random forest
def train_random_forest(X_train, y_train):
    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "../models/random_forest.pkl")

#for random forest
def train_neural_network(X_train, y_train):
    # Build the Neural Network model
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),  # Dropout layer to prevent overfitting
        # Hidden layer
        Dense(64, activation='relu'),
        Dropout(0.2),
        # Output layer (adjust the number of neurons as per the number of classes)
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Save the model
    model.save('../models/neural_network.h5')


if __name__ == "__main__":

    train_data = preprocess_data("../data/train.csv")
    X = train_data.drop("Survived", axis=1)
    y = train_data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    train_logistic_regression(X_train, y_train)
    train_random_forest(X_train, y_train)
    train_neural_network(X_train, y_train)
    
