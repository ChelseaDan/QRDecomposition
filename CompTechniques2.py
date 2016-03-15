import numpy as np
import sys

height = 3 if (len(sys.argv) <= 1) else int(sys.argv[1])
width = height
EPSILON = 0.0001

def initialise_random_symmetric():
  matrix = np.asarray([[np.random.randint(1,10) if i >= j else 0 for i in range(0,width)] 
  for j in range(0, height)]) 
  matrixT = np.asarray(matrix).T
  return np.add(matrix, matrixT, matrix)

def qr_decomposition(matrix):
  q = np.zeros((width, height))
  r = np.zeros((width, height))
  for i in range(0,width):
    q[i] = matrix[i]
    for j in range(0,i):
      qj = q[j].T
      ui = matrix[i]
      r[j][i] = np.dot(qj,ui)
      q[i] = q[i] - np.dot(r[j][i], qj)
    r[i][i] = np.linalg.norm(q[i])
    q[i] = q[i] / r[i][i] 
  return np.asarray(q).T, np.asarray(r)

def qr_iterations(q, r):
  eigenvectors = q
  ak = np.dot(r,q)
  while(abs(abs(ak.sum()) - np.matrix.trace(ak)) > EPSILON):
    ak = np.dot(r,q)
    q,r = qr_decomposition(ak)
    eigenvectors = np.dot(eigenvectors, q)
  print('\n')
  print('EIGENVALUES on diagonal:')
  print(ak)
  print('\n')
  print('MATRIX of EIGENVECTORS:')
  print(eigenvectors) 

def main():
  print('\n')
  matrix = initialise_random_symmetric()
  print('MATRIX:')
  print(matrix)
  print('\n')
  q,r = qr_decomposition(matrix)
  print('Q:')
  print(np.asarray(q))
  print('\n')
  print('R:')
  print(np.asarray(r))
  print('\n')
  print('QR:')
  print(np.dot(np.asarray(q),np.asarray(r)))
  qr_iterations(q,r)

if __name__ == '__main__':main()


