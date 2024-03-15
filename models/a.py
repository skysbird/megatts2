import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
# Let's visualize the padding operation described by the user's code snippet

# Define a sample output tensor's size
output_size = (1, 10, 3)  # (batch_size, time_steps, features)
mel_max_length = 20  # A hypothetical desired length

# Create a sample output tensor
output = torch.rand(output_size)

# Perform the padding operation as per the user's code
# The padding is done only along the second dimension (time_steps)
padded_output = F.pad(output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))

# Now we'll convert the tensors to numpy for visualization
output_np = output.detach().numpy().squeeze()
padded_output_np = padded_output.detach().numpy().squeeze()

# Create a plot to visualize the tensors before and after padding
fig, axs = plt.subplots(2, 1, figsize=(12, 6))

# Plot the original output tensor
axs[0].imshow(output_np, aspect='auto')
axs[0].set_title("Original Output")
axs[0].set_xlabel("Feature Dimension")
axs[0].set_ylabel("Time Step")

# Plot the padded output tensor
axs[1].imshow(padded_output_np, aspect='auto')
axs[1].set_title("Padded Output")
axs[1].set_xlabel("Feature Dimension")
axs[1].set_ylabel("Time Step")

# Display the plot
plt.tight_layout()
plt.show()
