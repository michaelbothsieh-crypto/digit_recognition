import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.ndimage import shift
import joblib
import os
import requests

def load_mnist():
    path = "mnist.npz"
    if not os.path.exists(path):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"Downloading MNIST from {url}...")
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)
    
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        
    return (x_train, y_train), (x_test, y_test)

def augment_data(X, y):
    print("Augmenting data...")
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        img = X[i].reshape(28, 28)
        
        # Original
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Shift Up
        X_aug.append(shift(img, [-1, 0], mode='constant', cval=0).flatten())
        y_aug.append(y[i])
        
        # Shift Down
        X_aug.append(shift(img, [1, 0], mode='constant', cval=0).flatten())
        y_aug.append(y[i])
        
        # Shift Left
        X_aug.append(shift(img, [0, -1], mode='constant', cval=0).flatten())
        y_aug.append(y[i])
        
        # Shift Right
        X_aug.append(shift(img, [0, 1], mode='constant', cval=0).flatten())
        y_aug.append(y[i])
        
    return np.array(X_aug), np.array(y_aug)

def train_model():
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Flatten images (28x28 -> 784)
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Augment training data (only a subset to save time/memory if needed, but let's try full)
    # To keep it fast, let's augment a smaller subset or just do it.
    # 60k * 5 = 300k samples. Might be slow for MLPClassifier.
    # Let's augment only 10% of data or just rely on the fact that we have 60k.
    # Actually, let's just augment the whole thing.
    x_train_aug, y_train_aug = augment_data(x_train, y_train)
    
    print(f"Training with {len(x_train_aug)} samples...")

    print("Training MLPClassifier...")
    # MLP with 2 hidden layers
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=20, alpha=1e-4,
                        solver='adam', verbose=10, random_state=1,
                        learning_rate_init=0.001) # Changed solver to adam, larger layers

    mlp.fit(x_train_aug, y_train_aug)

    print(f"Training set score: {mlp.score(x_train_aug, y_train_aug)}")
    print(f"Test set score: {mlp.score(x_test, y_test)}")

    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), 'mnist_model.pkl')
    joblib.dump(mlp, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
