"""Test script to generate .pngs of the toy datasets."""

import toy_data
import matplotlib.pyplot as plt

__TOY_DATASETS = [
	'rings', 'moons', 'checkerboard', 'chelsea', 'mixture_of_2gaussians'
]

if __name__ == "__main__":
	for data in __TOY_DATASETS:
		toy_data.generate_density(data)
	plt.show()
