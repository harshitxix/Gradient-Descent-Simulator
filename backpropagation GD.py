import pandas as pd

X=[[2,]]

y=[0,0,0,1,1,1]

w1= 0.1; w2= -0.2
w3=0.4; w4=0.3




# Error
error= Y[i]- o
total_error += error**2

#Backpropagation
delta_o = error * o * (1 - o)

 # Hidden layer grdients
delta_h1= delta_o * w7 * h1 * (1 - h1)
delta_h2= delta_o * w8 * h2 * (1 - h2)

#update weights
w1=w1 + learning_rate * delta_h1 * X1
w3=w3 + learning_rate * delta_h1 * X2
w5=w5 + learning_rate * delta_h1 * X3
