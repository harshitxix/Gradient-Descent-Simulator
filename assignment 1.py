# Perceptron for AND, OR, NOR Gates with Training

# -------- INPUT --------
gate = input("Enter gate (AND / OR / NOR): ").upper()
x1 = int(input("Enter x1 (0 or 1): "))
x2 = int(input("Enter x2 (0 or 1): "))

learning_rate = 1

# -------- INITIALIZE WEIGHTS, THRESHOLD & TARGET --------
if gate == "AND":
    w1 = 1
    w2 = 1
    threshold = 2
    target = x1 & x2

elif gate == "OR":
    w1 = 1
    w2 = 1
    threshold = 1
    target = x1 | x2

else:  # NOR
    w1 = -1
    w2 = -1
    threshold = -1
    target = int(not (x1 | x2))

# -------- FORWARD PASS --------
net = (x1 * w1) + (x2 * w2)

if net >= threshold:
    output = 1
else:
    output = 0

print("Initial Output:", output)
print("Target:", target)

# -------- TRAINING (WEIGHT UPDATE) --------
error = target - output

w1 = w1 + learning_rate * error * x1
w2 = w2 + learning_rate * error * x2

# -------- OUTPUT AFTER TRAINING --------
net = (x1 * w1) + (x2 * w2)

if net >= threshold:
    output = 1
else:
    output = 0

print("Updated w1:", w1)
print("Updated w2:", w2)
print("Final Output:", output)