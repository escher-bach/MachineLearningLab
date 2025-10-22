import ucimlrepo
import pandas as pd

iris = ucimlrepo.fetch_ucirepo(id=53) # Using the recommended function
data = iris.data.features
target = iris.data.targets

X = data.to_numpy()
y = target.to_numpy().reshape(-1) 