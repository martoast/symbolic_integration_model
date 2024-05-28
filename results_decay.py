import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# End-of-epoch losses
end_losses = [
    7.58,  # Epoch 1
    6.22,  # Epoch 2
    5.18,  # Epoch 3
    4.93,  # Epoch 4
    3.85,  # Epoch 5
    3.24,  # Epoch 6
    3.63,  # Epoch 7
    2.18,  # Epoch 8
    3.48,  # Epoch 9
    1.68   # Epoch 10
]

# Number of epochs
epochs = np.arange(1, len(end_losses) + 1)

# Exponential decay function
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# Fit the exponential decay model to the data
params, covariance = curve_fit(exp_decay, epochs, end_losses, p0=(10, 0.1, 1))

# Predict losses for future epochs
future_epochs = np.arange(1, 101)
predicted_losses = exp_decay(future_epochs, *params)

# Plot the original data and the fitted curve
plt.figure(figsize=(10, 6))
plt.plot(epochs, end_losses, 'o', label='End of Epoch Loss')
plt.plot(future_epochs, predicted_losses, '-', label='Fitted Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs. Loss with Exponential Decay Fit')
plt.legend()
plt.grid(True)
plt.show()

# Predict the number of epochs to reach near-zero loss
target_loss = 0.01
predicted_epoch = (np.log((target_loss - params[2]) / params[0]) / -params[1]).item()

print(f"Predicted number of epochs to reach near-zero loss: {predicted_epoch:.2f}")
