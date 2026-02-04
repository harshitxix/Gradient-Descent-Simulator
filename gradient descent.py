#step1: define error function
#step2: compute gradient
#step3: update rules(heart of gradient descent)
#step4: repeat until convergence
#for one parameter: New parameter = Old parameter - learning rate * gradient
import numpy as np
import matplotlib.pyplot as plt
X= np.linspace(0,10,50)
Y= 2*X + np.random.randn(50)*1.5 #add some noise

#step2: Initialize parameters
w= np.random.randn() #random weight
b= np.random.randn() #random bias
learning_rate= 0.01
epochs= 1000 #no of iterations
#step3: Training using gradient descent
n= len(X)
lines = []
for epoch in range(epochs):
    Y_pred= w*X + b
    error= Y_pred - Y
    cost= (1/n)*np.sum(error**2) #mean squared error

    #compute gradients
    dw= (2/n)*np.dot(error, X)
    db= (2/n)*np.sum(error)

    #update parameters
    w= w - learning_rate * dw
    b= b - learning_rate * db

    lines.append((w, b))

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Cost: {cost}, w: {w}, b: {b}')

#step4: Plotting results
plt.scatter(X, Y, color='blue', label='Data points')
for w_i, b_i in lines:
    plt.plot(X, w_i*X + b_i, color='orange', alpha=0.05)
plt.plot(X, w*X + b, color='red', label='Fitted line')
plt.title('Linear Regression using Gradient Descent')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

print(f'Final parameters: w = {w}, b = {b}')