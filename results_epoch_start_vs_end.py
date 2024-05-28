import matplotlib.pyplot as plt

# Loss values for each epoch
epoch_losses = [
    14.79, 7.58,  # Epoch 1
    7.40, 6.22,   # Epoch 2
    6.00, 5.18,   # Epoch 3
    5.35, 4.93,   # Epoch 4
    4.38, 3.85,   # Epoch 5
    4.16, 3.24,   # Epoch 6
    3.12, 3.63,   # Epoch 7
    2.76, 2.18,   # Epoch 8
    2.78, 3.48,   # Epoch 9
    2.05, 1.68    # Epoch 10
]

# Separate the starting and ending losses for each epoch
start_losses = epoch_losses[0::2]
end_losses = epoch_losses[1::2]

# Number of epochs
epochs = list(range(1, len(start_losses) + 1))

# Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, start_losses, label='Start of Epoch', marker='o')
plt.plot(epochs, end_losses, label='End of Epoch', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()
