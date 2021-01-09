import numpy as np
# пример работы с форматом npz
SEQUENCE = np.zeros((3,3))
print(SEQUENCE)
np.savez('1.npz', sequence_array=SEQUENCE)
SEQUENCE1 = np.load('1.npz')['sequence_array']
print(SEQUENCE1)

# converting list to array
arr = np.array([1,2,3,4])

