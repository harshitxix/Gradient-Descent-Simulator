# Forward and Backward Propagation in a Simple Neural Network

inputA = 0.35
inputB = 0.9
weight1 = 0.1
weight2 = 0.8
weight3 = 0.4
weight4 = 0.6
weight5 = 0.3
weight6 = 0.9
target_output = 1.0
learning_rate = 0.1


def easy_init():
    return {
        "inputA": inputA,
        "inputB": inputB,
        "weights": [weight1, weight2, weight3, weight4, weight5, weight6],
        "target": target_output,
        "lr": learning_rate,
    }

def forward_propogation(inputA, inputB, weight1, weight2, weight3, weight4, weight5, weight6, target_output):

    hidden1 = (inputA * weight1) + (inputB * weight2)
    hidden2 = (inputA * weight3) + (inputB * weight4)
    hidden3 = (hidden1 * weight5) + (hidden2 * weight6)

    print("Hidden Layer 1")
    print("Neuron 1: ", hidden1)
    print("Neuron 2: ", hidden2)

    print("Hidden Layer 2")
    print("Neuron 3: ", hidden3)

    output = hidden3
    error = target_output - output
    squared_error = 0.5 * (error ** 2)

    ##print("Output Layer")
    print("Output Neuron: ", output)
    print("Output Error: ", error)
    ##print("Squared Error: ", squared_error)

    return hidden1, hidden2, hidden3, output, error, squared_error

#backward propagation

def backward_propogation(inputA, inputB, hidden1, hidden2, hidden3, output, error, weights, learning_rate):
    w1, w2, w3, w4, w5, w6 = weights

    dE_dout = -error

    dE_dw5 = dE_dout * hidden1
    dE_dw6 = dE_dout * hidden2

    dE_dhidden1 = dE_dout * w5
    dE_dhidden2 = dE_dout * w6
    dE_dw1 = dE_dhidden1 * inputA
    dE_dw2 = dE_dhidden1 * inputB
    dE_dw3 = dE_dhidden2 * inputA
    dE_dw4 = dE_dhidden2 * inputB

    w1_new = w1 - learning_rate * dE_dw1
    w2_new = w2 - learning_rate * dE_dw2
    w3_new = w3 - learning_rate * dE_dw3
    w4_new = w4 - learning_rate * dE_dw4
    w5_new = w5 - learning_rate * dE_dw5
    w6_new = w6 - learning_rate * dE_dw6

    print("Backward Propagation")
    print("Updated Weights:")
    print("w1: ", w1_new)
    print("w2: ", w2_new)
    print("w3: ", w3_new)
    print("w4: ", w4_new)
    print("w5: ", w5_new)
    print("w6: ", w6_new)

    return (w1_new, w2_new, w3_new, w4_new, w5_new, w6_new)


if __name__ == "__main__":
    w1, w2, w3, w4, w5, w6 = weight1, weight2, weight3, weight4, weight5, weight6

    for epoch in range(10):
        print(f"\n===== Epoch {epoch + 1} =====")
        
        hidden1, hidden2, hidden3, output, error, squared_error = forward_propogation(
            inputA, inputB, w1, w2, w3, w4, w5, w6, target_output
        )
        
        print(f"Epoch {epoch + 1} Error: {error}")

        w1, w2, w3, w4, w5, w6 = backward_propogation(
            inputA, inputB, hidden1, hidden2, hidden3, output, error,
            [w1, w2, w3, w4, w5, w6], learning_rate
        )