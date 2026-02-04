def check(x1, x2, w1, w2, theta):
    total = x1*w1 + x2*w2
    if total >= theta:
        return 1
    else:
        return 0

def learn(inputs, outputs):
    w1 = 0.6
    w2 = 0.5
    theta = 0.7
    
    for _ in range(100):
        for i in range(4):
            x1, x2 = inputs[i]
            correct = outputs[i]
            guess = check(x1, x2, w1, w2, theta)
            
            if guess != correct:
                error = correct - guess
                w1 = w1 + 0.1 * error * x1
                w2 = w2 + 0.1 * error * x2
                theta = theta - 0.1 * error
    
    return w1, w2, theta

def show(inputs, outputs, w1, w2, theta, name):
    print(f"\n{name}: w1={w1:.1f}, w2={w2:.1f}, theta={theta:.1f}")
    for i in range(4):
        x1, x2 = inputs[i]
        result = check(x1, x2, w1, w2, theta)
        print(f"{x1},{x2} -> {result}")

# Inputs: 00, 01, 10, 11
inputs = [[0,0], [0,1], [1,0], [1,1]]

# AND
w1, w2, t = learn(inputs, [0,0,0,1])
show(inputs, [0,0,0,1], w1, w2, t, "AND")

# OR
w1, w2, t = learn(inputs, [0,1,1,1])
show(inputs, [0,1,1,1], w1, w2, t, "OR")

# NOR
w1, w2, t = learn(inputs, [1,0,0,0])
show(inputs, [1,0,0,0], w1, w2, t, "NOR")