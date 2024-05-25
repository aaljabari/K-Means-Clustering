import pandas as pd

iris = pd.read_csv("datasets/iris.csv")
gaussians_6 = pd.read_csv("datasets/3gaussians-std0.6.csv")
gaussians_9 = pd.read_csv("datasets/3gaussians-std0.9.csv")
circles = pd.read_csv("datasets/circles.csv")
moons = pd.read_csv("datasets/moons.csv")


# Compute statistics
iris_stat = iris.describe()
gaussians_6_stat = gaussians_6.describe()
gaussians_9_stat = gaussians_9.describe()
circles_stat = circles.describe()
moons_stat = moons.describe()

# Print statistics
print("iris Dataset Statistics:\n")
print(iris_stat)
print("3gaussians-std0.6 Dataset Statistics:\n")
print(gaussians_6_stat)
print("\3gaussians-std0.9 Dataset Statistics:\n")
print(gaussians_9_stat)
print("\circles Dataset Statistics:\n")
print(circles_stat)
print("\moons Dataset Statistics:\n")
print(moons_stat)