import numpy as np
import matplotlib.pyplot as plt

batch_sizes = [18, 24, 32, 40, 48, 64]
num_epochs = [10, 15, 20]
accuracy_values_arr = [79.15, 82.21, 82.11, 81.59, 82.78, 82.84, 81.13, 83.68, 83.78, 83.68, 83.57, 82.77, 83.40, 81.64, 81.20, 83.97, 83.91, 83.12]

accuracy_values = []

for batch_size in batch_sizes:
    for num_epoch in num_epochs:
        index = batch_sizes.index(batch_size) * len(num_epochs) + num_epochs.index(num_epoch)
        print(f'{batch_size} {num_epoch} {index} {accuracy_values_arr[index]}')
        accuracy_values.append((batch_size, num_epoch, accuracy_values_arr[index]))

batch_sizes_plt = [entry[0] for entry in accuracy_values]
num_epochs_plt = [entry[1] for entry in accuracy_values]
accuracies = [entry[2] for entry in accuracy_values]

# Create a 3D scatter plot to visualize accuracy based on batch size and epochs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(batch_sizes_plt, num_epochs_plt, accuracies, c=accuracies, cmap='Reds')

# Add labels and title
ax.set_xlabel('Batch Size')
ax.set_ylabel('Number of Epochs')
ax.set_zlabel('Accuracy')
plt.title('Accuracy vs. Batch Size and Number of Epochs')

plt.show()
