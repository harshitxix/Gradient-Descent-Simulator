# Multi-Layer Perceptron (Neural Network)

# Single Perceptron function
def check(x1, x2, x3, w1, w2, w3, theta):
    """Single perceptron: output = 1 if (w1*x1 + w2*x2 + w3*x3) >= theta"""
    total = x1*w1 + x2*w2 + x3*w3
    return 1 if total >= theta else 0

def learn(inputs, outputs):
    """Train a single perceptron to learn linear patterns (AND, OR)"""
    w1, w2, w3, theta = 0.5, 0.5, 0.5, 0.5
    learning_rate = 0.1
    
    # Train for up to 100 epochs
    epoch = 0
    while epoch < 100:
        for inp, out in zip(inputs, outputs):
            x1, x2, x3 = inp
            
            # Make prediction
            guess = check(x1, x2, x3, w1, w2, w3, theta)
            
            # If wrong, update weights
            if guess != out:
                error = out - guess
                w1 += learning_rate * error * x1
                w2 += learning_rate * error * x2
                w3 += learning_rate * error * x3
                theta -= learning_rate * error
        epoch += 1
    
    return w1, w2, w3, theta

def show(inputs, outputs, w1, w2, w3, theta, name):
    """Display results with accuracy"""
    print(f"\n{name}")
    correct = 0
    
    for inp, expected in zip(inputs, outputs):
        x1, x2, x3 = inp
        result = check(x1, x2, x3, w1, w2, w3, theta)
        match = "✓" if result == expected else "✗"
        
        if result == expected:
            correct += 1
        
        print(f"{x1}{x2}{x3} | Expected: {expected} Got: {result} {match}")
    
    accuracy = (correct / len(inputs)) * 100
    print(f"Accuracy: {correct}/{len(inputs)} ({accuracy:.0f}%)")

# ========================================
# NEURAL NETWORK (Multi-Layer Perceptron)
# ========================================

def activate(x):
    """Activation function: returns 1 if x >= 0, else 0"""
    return 1 if x >= 0 else 0

def forward(x1, x2, x3, w_hidden, w_output, b_hidden, b_output):
    """
    Forward pass through 3-layer neural network
    Layer 1 (Input): 3 neurons
    Layer 2 (Hidden): 2 neurons  
    Layer 3 (Output): 1 neuron
    """
    # Hidden layer computations
    h1 = activate(x1*w_hidden[0][0] + x2*w_hidden[0][1] + x3*w_hidden[0][2] + b_hidden[0])
    h2 = activate(x1*w_hidden[1][0] + x2*w_hidden[1][1] + x3*w_hidden[1][2] + b_hidden[1])
    
    # Output layer computation
    output = activate(h1*w_output[0] + h2*w_output[1] + b_output)
    return output

def train_mlp(inputs, outputs):
    """Train multi-layer perceptron to learn non-linear patterns (like XOR)"""
    # Initialize weights and biases
    w_hidden = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    w_output = [0.5, 0.5]
    b_hidden = [0.5, 0.5]
    b_output = 0.5
    learning_rate = 0.1
    
    # Train for up to 200 epochs
    epoch = 0
    while epoch < 200:
        for inp, expected in zip(inputs, outputs):
            x1, x2, x3 = inp
            
            # Forward pass - compute hidden layer
            h1 = activate(x1*w_hidden[0][0] + x2*w_hidden[0][1] + x3*w_hidden[0][2] + b_hidden[0])
            h2 = activate(x1*w_hidden[1][0] + x2*w_hidden[1][1] + x3*w_hidden[1][2] + b_hidden[1])
            
            # Forward pass - compute output
            output = activate(h1*w_output[0] + h2*w_output[1] + b_output)
            
            error = expected - output
            
            # Update output layer if wrong
            if error != 0:
                w_output[0] += learning_rate * error * h1
                w_output[1] += learning_rate * error * h2
                b_output -= learning_rate * error
                
                # Update hidden layer
                for i in range(2):
                    w_hidden[i][0] += learning_rate * error * x1
                    w_hidden[i][1] += learning_rate * error * x2
                    w_hidden[i][2] += learning_rate * error * x3
                    b_hidden[i] -= learning_rate * error
        
        epoch += 1
    
    return w_hidden, w_output, b_hidden, b_output

def show_mlp(inputs, outputs, w_hidden, w_output, b_hidden, b_output, name):
    """Display neural network results with accuracy"""
    print(f"\n{name}")
    correct = 0
    
    for inp, expected in zip(inputs, outputs):
        x1, x2, x3 = inp
        result = forward(x1, x2, x3, w_hidden, w_output, b_hidden, b_output)
        match = "✓" if result == expected else "✗"
        
        if result == expected:
            correct += 1
        
        print(f"{x1}{x2}{x3} | Expected: {expected} Got: {result} {match}")
    
    accuracy = (correct / len(inputs)) * 100
    print(f"Accuracy: {correct}/{len(inputs)} ({accuracy:.0f}%)")

# 8 Inputs: 000, 001, 010, 011, 100, 101, 110, 111
inputs = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]

print("\n" + "="*50)
print("SINGLE LAYER PERCEPTRON (Can learn linear gates)")
print("="*50)

# Test OR gate - Single perceptron works
print("\n>>> Training OR gate...")
outputs_or = [0, 1, 1, 1, 1, 1, 1, 1]
w1, w2, w3, t = learn(inputs, outputs_or)
show(inputs, outputs_or, w1, w2, w3, t, "OR GATE - 100% Learning")

# Test AND gate - Single perceptron works
print("\n>>> Training AND gate...")
outputs_and = [0, 0, 0, 0, 0, 0, 0, 1]
w1, w2, w3, t = learn(inputs, outputs_and)
show(inputs, outputs_and, w1, w2, w3, t, "AND GATE - 100% Learning")

print("\n" + "="*50)
print("MULTI-LAYER NEURAL NETWORK (Can learn non-linear gates)")
print("="*50)

# Test XOR gate - Only neural network can learn
print("\n>>> Training XOR gate with neural network...")
outputs_xor = [0, 1, 1, 0, 1, 0, 0, 1]
w_h, w_o, b_h, b_o = train_mlp(inputs, outputs_xor)
show_mlp(inputs, outputs_xor, w_h, w_o, b_h, b_o, "XOR GATE - Neural Network Learning")
