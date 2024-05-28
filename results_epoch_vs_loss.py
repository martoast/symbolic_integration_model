import numpy as np
import matplotlib.pyplot as plt

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

# Plotting the raw losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, end_losses, 'o-', label='End of Epoch Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs. Loss')
plt.legend()
plt.grid(True)
plt.show()
