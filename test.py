import torch
import matplotlib.pyplot as plt

# Create a 2D tensor (like an image)
tensor_image = torch.randn(10, 10)  # 10x10 tensor

# Convert the tensor to a NumPy array
numpy_image = tensor_image.numpy()

# Plot the tensor as an image
plt.imshow(numpy_image)
plt.title('Random Image Tensor')
plt.colorbar()
plt.show()
