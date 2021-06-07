import numpy as np
import copy
from math import inf

class NewVertex:
    def __init__(self, v = [], cost = 0):
        self.coordinates = v #krotka współrzednych
        self.cost = cost


class Solution:
    def __init__(self, cost_matrix, path=[], lb = 0):
        self.cost_matrix = cost_matrix#macierz kosztów
        self.path = path#lista par wierzcholków w scieżce nie posortowana
        self.lb = lb#lower bound
    def reduce_cost_matrix(self, basic=True):
        fi = 0
      
        if basic is True:
            for id_, row in enumerate(self.cost_matrix):
                if min(row) != inf:
                    fi += min(row)
                    self.cost_matrix[id_] -= min(row)
          
            for id_, column in enumerate(self.cost_matrix.T):
                if min(column) != inf:
                    fi += min(column)
                    self.cost_matrix[:, id_] -= min(column)
      
        else:
            for id_, column in enumerate(self.cost_matrix.T):
                fi += min(column)
                self.cost_matrix[:, id_] -= min(column)
    
            for id_, row in enumerate(self.cost_matrix):
                fi += min(row)
                self.cost_matrix[id_] -= min(row)
        return fi

    def choose_new_vertex(self):#zwraca obj. klasy NewVertex
        # znajdź indeksy elementów równych zero
        zero_indices = np.argwhere(self.cost_matrix == 0)

        new_vertex_id = zero_indices[0]
        cost = -1

        for zero in zero_indices:
            # Wyodrębnij rząd w którym znajduje się
            # dane zero oraz usuń to zero z otrzymanego wektora
            row = self.cost_matrix[zero[0], :]
            row = np.delete(row, zero[1])

            # Wyodrębnij kolumnę w której znajduje się
            # dane zero oraz usuń to zero z otrzymanego wektora
            col = self.cost_matrix[:, zero[1]]
            col = np.delete(col, zero[0])

            new_cost = np.amin(row) + np.amin(col)
            if new_cost > cost:
                cost = new_cost
                new_vertex_id = zero

        return NewVertex(new_vertex_id, cost)


    def update_cost_matrix(self, cordinates):#dostaje krotke wsp., nic nie zwraca; modyfikuje(zabrania powrotnego przejscia) self.cost_matrix 
        row_size = np.shape(self.cost_matrix)[0]
        col_size = np.shape(self.cost_matrix)[1]

        for i in range(row_size):
          self.cost_matrix[cordinates[0]][i] = inf
        for j in range(col_size):
          self.cost_matrix[j][cordinates[1]] = inf
        
        self.cost_matrix[cordinates[1]][cordinates[0]] = inf

        for row in range(np.shape(self.cost_matrix)[0]):
          for col in range(np.shape(self.cost_matrix)[1]):
            for el in self.path:
              if col == el[1]:
                self.cost_matrix[row][col] = inf
                
    def get_path():#zwraca listę kolejnych wierzchołków w sciezce korzystając z self.cost_matrix 2x2 i self.path
        pass




###
 # Given the optimal path, return the optimal cost.
 # @param optimal_path
 # @param m
 # @return Cost of the path.
 ##
def get_optimal_cost(optimal_path, m):
    cost = 0

    for idx in range(1, len(optimal_path)):
        cost += m[optimal_path[idx - 1]][optimal_path[idx]]

    ## Add the cost of returning from the last city to the initial one.
    cost += m[optimal_path[len(optimal_path) - 1]][optimal_path[0]]

    return cost


###
 # Create the right branch matrix with the chosen vertex forbidden and the new lower bound.
 # @param m
 # @param v
 # @param lb
 # @return New branch.
 ##
def create_right_branch_matrix(m, v, lb):
    m = copy.deepcopy(m)
    m[v[0]][v[1]] = inf
    return Solution(m, [], lb);


###
 # Retain only optimal ones (from all possible ones).
 # @param solutions
 # @return Vector of optimal solutions.
 ##
def filter_solutions(solutions):
    optimal_cost = min(solutions, key=lambda x: x[0])

    optimal_solutions = []
    for i in solutions:
        if i[0] == optimal_cost:
            optimal_solutions.append(i)

    return optimal_solutions


def solve_tsp(cm):
    lifo = []
    n = len(cm)
    n_levels = n-2
    
    lifo.append(Solution(cm))
    best_lb = inf
    end_solutions = []#lista rorwiązan koncowych
    
    while len(lifo):

        left_branch = lifo.pop()
        while (len(left_branch.path) != n_levels and left_branch.lb <= best_lb):
            # Repeat until a 2x2 matrix is obtained or the lower bound is too high...
            if len(left_branch.path) == 0:
                left_branch.lb = 0

            # 1. Reduce the matrix in rows and columns.
            # @TODO (KROK 1)
            new_cost = left_branch.reduce_cost_matrix();
            # 2. Update the lower bound and check the break condition.
            print("po rekukcji:\n", left_branch.cost_matrix, "\nlb:", new_cost)
            left_branch.lb = new_cost;
            if left_branch.lb > best_lb:
                break
            
            # 3. Get new vertex and the cost of not choosing it.
            #new_vertex = NewVertex(); # @TODO (KROK 2)
            new_vertex = left_branch.choose_new_vertex()
            # 4. @TODO Update the path - use append_to_path method.
            left_branch.path.append((new_vertex.coordinates))
            # 5. @TODO (KROK 3) Update the cost matrix of the left branch.
            left_branch.update_cost_matrix((new_vertex.coordinates))
            # 6. Update the right branch and push it to the LIFO.
            new_lower_bound = left_branch.lb + new_vertex.cost;
            lifo.append(create_right_branch_matrix(cm, new_vertex.coordinates,
                                                      new_lower_bound))
        

        if (left_branch.lb <= best_lb):
            # If the new solution is at least as good as the previous one,
            # save its lower bound and its path.
            best_lb = left_branch.lb
            new_path = left_branch.get_path()
            end_solutions.append((get_optimal_cost(new_path, cm), new_path));
        


    

    return filter_solutions(end_solutions); # Filter solutions to find only optimal ones.
    
    
      
      
def main():
    matrix = np.array([[inf, 10, 8, 19, 12], [10, inf, 20, 6, 3], [8, 20, inf, 4, 2], [19, 6, 4, inf, 7], [12, 3, 2, 7, inf]])
    print("Macierz początkowa:\n", matrix,"\n")
    print(solve_tsp(matrix))
    
    
    
        
        

main()
