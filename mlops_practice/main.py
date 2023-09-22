from sklearn.datasets import load_digits
import torch


def main():
	X, y = load_digits(return_X_y=True)

	print(X.shape, y.shape)
	print(X[0], y[0])


if __name__ == '__main__':
	main()
