import numpy as np
import sys

#set default to 3x3 matrix or first arg from command line.
height = 3 if (len(sys.argv) <= 1) else int(sys.argv[1])
width = height
EPSILON = 0.0001

def initialise_random_symmetric(f):
  #create upper triangular matrix.
  matrix = np.asarray([[np.random.randint(1,10) if i >= j else 0 for i in range(0,width)] for j in range(0, height)]) 
  #transpose upper triangular.
  matrixT = np.asarray(matrix).T
  #sum the matrices together to get a symmetric matrix.
  matrix = np.add(matrix, matrixT, matrix)
  f.write("Matrix: \n")
  np.savetxt(f, matrix, fmt='%10.3f')
  return matrix

def qr_decomposition(matrix):
  q = np.zeros((width, height))
  r = np.zeros((width, height))
  for i in range(0,width):
    q[:,i] = matrix[:,i]
    for j in range(0,i):
      r[j][i] = np.dot(q[:,j].T,matrix[:,i])
      q[:,i] = q[:,i] - np.dot(r[j][i], q[:,j])
    r[i][i] = np.linalg.norm(q[:,i])
    q[:,i] = q[:,i] / r[i][i]
  return q, r

def qr_iterations(q, r, f):
  eigenvectors = q
  #find the numpy eigenvalues and eigenvectors for later comparison.
  val, vec = np.linalg.eig(np.dot(q,r))
  ak = np.dot(r,q)
  write_q_r(f,q,r)
  #while the trace is not the dominant sum of the matrix.
  while(abs(abs(ak).sum() - np.matrix.trace(abs(ak))) > EPSILON):
    ak = np.dot(r,q)
    q,r = qr_decomposition(ak)
    eigenvectors = np.dot(eigenvectors, q)
  write_eigenval_vecs(f, ak, eigenvectors, val, vec)

def write_q_r(f,q,r):
  f.write("Q: \n")
  np.savetxt(f, q, fmt='%10.3f')
  f.write("R: \n")
  np.savetxt(f, r,fmt='%10.3f') 

def write_eigenval_vecs(f, ak, eigenvectors, val, vec):
  f.write("Numpy Eigenvalues: \n")
  np.savetxt(f, val, fmt='%10.3f')
  f.write("Eigenvalues: \n")  
  np.savetxt(f, ak.diagonal(), fmt='%10.3f')
  f.write("Matrix of eigenvectors (Numpy): \n")
  np.savetxt(f, vec, fmt='%10.3f')
  f.write("Matrix of eigenvectors: \n")
  np.savetxt(f, eigenvectors, fmt='%10.3f')

#run "python CompTechniques2.py" with a single number to indicate the dimension.
def main():
  f = open('output.txt','w')
  matrix = initialise_random_symmetric(f)
  q,r = qr_decomposition(matrix)
  qr_iterations(q,r,f)
  f.close()

if __name__ == '__main__':main()


