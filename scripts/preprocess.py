import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)
    
    # Drop unnecessary columns
    data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    
    # Fill missing values
    data["Age"] = data["Age"].fillna(data["Age"].median())  # Reassign instead of inplace
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])  # Reassign instead of inplace

    # Convert categorical to numerical
    data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)

    return data

if __name__ == "__main__":
    train_data = preprocess_data("../data/train.csv")
    X = train_data.drop("Survived", axis=1)
    y = train_data["Survived"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
