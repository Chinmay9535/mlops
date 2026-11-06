import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

def train():
    # Load diabetes dataset
    X, y = load_diabetes(return_X_y=True)
    
    # Train a simple Ridge model
    model = Ridge()
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'diabetes_model.joblib')
    print("Model trained and saved to diabetes_model.joblib")

if __name__ == '__main__':
    train()
