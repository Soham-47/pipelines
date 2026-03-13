import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X, y = load_iris(return_X_y=True)
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    z = np.clip(z, -500, 500)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class ANN:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2./layers[i]))
            self.biases.append(np.zeros((1, layers[i+1])))

    def forward(self, x):
        self.z = []     
        self.a = [x]   
        for i in range(len(self.weights)-1):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            a = relu(z)
            self.z.append(z)
            self.a.append(a)
        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        a = softmax(z)
        self.z.append(z)
        self.a.append(a)
        return a

    def backward(self, y, lr):
        m = y.shape[0]
        dz = self.a[-1] - y
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * relu_deriv(self.z[i-1])

    def fit(self, X, y, epochs=1000, lr=0.01, batch_size=16):
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                self.forward(X_batch)
                self.backward(y_batch, lr)
            
            if epoch % 100 == 0:
                y_pred = self.forward(X)
                loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

model = ANN(layers=[4, 16, 8, 3])
model.fit(X_train, y_train, epochs=1000, lr=0.01, batch_size=16)
preds = model.predict(X_test)
true = np.argmax(y_test, axis=1)
print("Accuracy:", np.mean(preds == true))