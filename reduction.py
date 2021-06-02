import numpy as np

def reduction(matrix, basic=True):
  fi = 0
  
  if basic is True:
    for id_, row in enumerate(matrix):
      fi += min(row)
      matrix[id_] -= min(row)
      
    for id_, column in enumerate(matrix.T):
      fi += min(column)
      matrix[:, id_] -= min(column)
  
  else:
    for id_, column in enumerate(matrix.T):
      fi += min(column)
      matrix[:, id_] -= min(column)

    for id_, row in enumerate(matrix):
      fi += min(row)
      matrix[id_] -= min(row)
    
  return matrix, fi


if __name__ == "__main__":
    mat = np.array([[20, 40, 10, 50], [100, 80, 30, 40], [10, 5, 60, 20], [70, 30, 10, 25]])

    matrix_true, fi_true = reduction(mat)
    print("Macierz zredukowana w kolejności wiersze-kolumny.\n")
    print(matrix_true)
    print("\nWartość fi: " + str(fi_true) + "\n")

    mat = np.array([[20, 40, 10, 50], [100, 80, 30, 40], [10, 5, 60, 20], [70, 30, 10, 25]])

    matrix_false, fi_false = reduction(mat, False)
    print("Macierz zredukowana w kolejności kolumny-wiersze.\n")
    print(matrix_false)
    print("\nWartość fi: " + str(fi_false) + "\n")
    
