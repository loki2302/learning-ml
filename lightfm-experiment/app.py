from lightfm import LightFM
import numpy as np
from scipy.sparse import coo_matrix

# vertical users
# horizontal movies
scores = np.matrix([
  [1, 1, 0],
  [1, 1, 0],
  [1, 0, 0]
])

model = LightFM()
model.fit(coo_matrix(scores), epochs=20)

predictions = model.predict(
  np.array([2, 2, 2]), 
  np.array([0, 1, 2])
)
print(predictions[0] > predictions[1])
print(predictions[1] > predictions[2])
