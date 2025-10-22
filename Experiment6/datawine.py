import ucimlrepo
import pandas as pd

wine = ucimlrepo.fetch_ucirepo(id=109) # Wine dataset
data = wine.data.features
target = wine.data.targets

X = data.to_numpy()
y = target.to_numpy().reshape(-1)