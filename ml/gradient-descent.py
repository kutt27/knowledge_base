import numpy as np
import matplotlib.pyplot as plt 

# example data point
X = np.array([1,2,3,4,5], dtype=np.float32)
Y = np.array([3,6,9,12,15], dype=np.float32)

w = 0.0 
n_iters = 20
learning_rate = 0.01
history = []

for epoch in range(n_iters):
    y_pred = w * X
    loss = np.mean((y_pred - Y)**2)
    history.append(loss)
    gradient = np.mean(2 * X * (y_pred - Y))
    w = w - learning_rate * gradient
    print(f'Epoch {epoch+1}: w = {w:.3f}, Loss = {loss:.8f}')

print(f'Prediction after training: f(5) = {w*5:.3f} (Target was 15)')
plt.plot(history)
plt.title("Gradient Descent Convergence")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
