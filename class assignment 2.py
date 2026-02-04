import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_neural_network(X, y, gate_name, epochs=10000, learning_rate=0.1, hidden_size=2):
    """
    Train a neural network for logic gates
    """
    np.random.seed(1)
    
    # Initialize weights and biases
    W1 = np.random.rand(2, hidden_size)
    W2 = np.random.rand(hidden_size, 1)
    b1 = np.zeros((1, hidden_size))
    b2 = np.zeros((1, 1))
    
    print(f"\n{'='*50}")
    print(f"Training {gate_name} Gate")
    print(f"{'='*50}")
    
    # Training loop
    for epoch in range(epochs):
        # Forward Propagation
        # Hidden layer
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        # Output layer
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)
        
        # Calculate error
        error = y - a2
        
        # Backpropagation
        d_output = error * sigmoid_derivative(a2)
        error_hidden_layer = d_output.dot(W2.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(a1)
        
        # Update weights and biases
        W2 += a1.T.dot(d_output) * learning_rate
        b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        W1 += X.T.dot(d_hidden_layer) * learning_rate
        b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
        
        # Print progress
        if epoch % 2000 == 0:
            loss = np.mean(error**2)
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    # Final predictions
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    predictions = sigmoid(z2)
    
    print(f"\nFinal Results for {gate_name} Gate:")
    print("Input\t\tExpected\tPredicted\tRounded")
    for i in range(len(X)):
        print(f"{X[i]}\t{y[i][0]}\t\t{predictions[i][0]:.4f}\t\t{round(predictions[i][0])}")
    
    return W1, W2, b1, b2

# Input data (same for all gates)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# AND Gate
y_and = np.array([[0], [0], [0], [1]])
print("\n" + "="*50)
print("AND GATE")
print("="*50)
train_neural_network(X, y_and, "AND", epochs=5000, hidden_size=2)

# OR Gate
y_or = np.array([[0], [1], [1], [1]])
print("\n" + "="*50)
print("OR GATE")
print("="*50)
train_neural_network(X, y_or, "OR", epochs=5000, hidden_size=2)

# NOR Gate
y_nor = np.array([[1], [0], [0], [0]])
print("\n" + "="*50)
print("NOR GATE")
print("="*50)
train_neural_network(X, y_nor, "NOR", epochs=5000, hidden_size=2)

# XOR Gate
y_xor = np.array([[0], [1], [1], [0]])
print("\n" + "="*50)
print("XOR GATE")
print("="*50)
train_neural_network(X, y_xor, "XOR", epochs=10000, hidden_size=2)