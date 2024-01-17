import pathlib
import numpy as np
import matplotlib.pyplot as plt

def save_np(arr, file_name="sobel_images.npy"):
	'''
		Save numpy array to file
	Arguments:
		- arr: The numpy array need to save
		- file_path: The file path that save numpy
	'''
	pwd = pathlib.Path()
	folder_path = pwd / "out"
	folder_path.mkdir(parents=True, exist_ok=True)
	file_path = folder_path / file_name
	np.save(file_path, arr)

def plot_images(images, file_name=None):
	'''
		Plot the gray images and save results
	Arguments:
		- images: The list of images to plot and save in format (n / 2, 2)
	'''
	# Create a figure with subplots
	n = len(images)
	fig, axs = plt.subplots(n // 2, 2, figsize=(100, 200))
	cols = ["Original", "Sobel"]
	for ax, col in zip(axs[0], cols):
		ax.set_title(col)

	for i in range(0, n, 2):
		# Plot each array on its own subplot
		axs[i // 2][0].imshow(images[i], cmap='gray')
		axs[i // 2][0].set_axis_off()
		axs[i // 2][1].imshow(images[i+1], cmap='gray')
		axs[i // 2][1].set_axis_off()

	# Adjust spacing and layout
	plt.tight_layout(pad=2.0)

	# Save the  images to folder imgs
	if file_name is not None:
		pwd = pathlib.Path()
		folder_path = pwd / "out"
		folder_path.mkdir(parents=True, exist_ok=True)
		file_path = folder_path / file_name
		fig.savefig(file_path)

	# Show the plot
	plt.show()