import numpy as np
import copy
from math import inf



class NewVertex:
    def __init__(self, v=[], cost=0):
        self.coordinates = v  # krotka współrzednych
        self.cost = cost # optymistyczny koszt wyłączenia


class Solution:#klasa reprezentująca PP
    def __init__(self, cost_matrix, path=[], lb=0):
        self.cost_matrix = cost_matrix  # macierz kosztów
        self.path = path  # lista par wierzcholków w scieżce nie posortowana
        self.lb = lb  # lower bound

    def reduce_cost_matrix(self):#funkcja redukuje macierz w kolejnosci wiersze, kolumny i zwraca LB
        fi = 0
        for id_, row in enumerate(self.cost_matrix):
            if min(row) != inf:
                fi += min(row)
                self.cost_matrix[id_] -= min(row)
        for id_, column in enumerate(self.cost_matrix.T):
            if min(column) != inf:
                fi += min(column)
                self.cost_matrix[:, id_] -= min(column)

        return fi

    def choose_new_vertex(self):  # zwraca obj. klasy NewVertex reprezentujący odcinek o max optymistycznym koszcie wyłączenia
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
    
    
    
    def update_cost_matrix(self, cordinates):  # dostaje krotke wsp., nic nie zwraca; modyfikuje(zabrania powrotnego przejscia) self.cost_matrix
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
        self.block_cycles()
    
    def block_cycles(self):
        for m in range(len(self.path)):               
            path = copy.deepcopy(self.path)
            searching_1 = path[m][1]
            searching_2 = path[m][0]
            path.pop(0)

            while(len(path)):
                before = len(path)
                for i in range(len(path)):
                    if (path[i][0] == searching_1):
                        searching_1 = path[i][1]
                        path.pop(i)
                        break

                    if (path[i][1] == searching_2):
                        searching_2 = path[i][0]
                        path.pop(i)
                        break
    
                if before == len(path):
                    break
            
            self.cost_matrix[searching_2][searching_1] = inf
            self.cost_matrix[searching_1][searching_2] = inf

    def get_path(self):  # zwraca listę kolejnych wierzchołków w sciezce korzystając z macierzy 2x2 i nieposortowanej scieżki
        unsorted = []
        pth = []
        size = len(self.cost_matrix)
        self.block_cycles()

        self.reduce_cost_matrix()
        new_v = self.choose_new_vertex()
        self.path.append((new_v.coordinates))
        self.update_cost_matrix(new_v.coordinates)
        
        for r in range(size):
            for c in range(size):
                if self.cost_matrix[r][c] != float('inf'):
                    unsorted.append(NewVertex([r, c], self.cost_matrix[r][c]))
        
        for el in self.path:
            unsorted.append(NewVertex(el, 0))
        s_uns = len(unsorted)
        row = unsorted[0].coordinates[0]
        col = unsorted[0].coordinates[1]
        pth.append(row)
        pth.append(col)

        for i in range(s_uns):
            for j in range(s_uns):
                if unsorted[j].coordinates[0] == col:
                    col = unsorted[j].coordinates[1]
                    pth.append(unsorted[j].coordinates[1])
                    break
                if unsorted[j].coordinates[1] == row:
                    row = unsorted[j].coordinates[0]
                    pth.insert(0, unsorted[j].coordinates[0])
                    break
        self.path = pth
        return pth



#funkcja licząca (sprawdzająca) koszt przejscia po scieżce
def get_optimal_cost(optimal_path, m):
    cost = 0
    for idx in range(1, len(optimal_path)):
        cost = cost + m[optimal_path[idx - 1]][optimal_path[idx]]
    cost = cost + m[optimal_path[0]][optimal_path[-1]]

    return cost


#Podział PP i zabronienie przejscia
def create_right_branch_matrix(m, v, lb, path):
    m = copy.deepcopy(m)
    m[v[0]][v[1]] = inf
    return Solution(m, path, lb)


#funkcja znajdująca optymalne rozwiazania z znalezionych
def filter_solutions(solutions):
    optimal_cost = min(solutions, key=lambda x: x[0])[0]
    optimal_solutions = []
    for i in solutions:
        if i[0] == optimal_cost:
            optimal_solutions.append(i)

    return optimal_solutions


def solve_tsp(cm):
    lifo = []
    n = len(cm)
    n_levels = n - 2
    org_cm = copy.deepcopy(cm)
    lifo.append(Solution(cm))
    best_lb = inf #wartosc najlepszego dotychczas znalezionego rozw.(v*)
    end_solutions = []  # lista rorwiązan koncowych
    licznik_pp = 0

    while len(lifo): #dopóki wszystkie PP nie zamknięte
        left_branch = lifo.pop()#wybieramy nowy PP i usuwamy go z listy

        #jesli nie uzyskalismy macierzy 2x2 ani nie przekroczylimy najlepszego LB
        while (len(left_branch.path) != n_levels and left_branch.lb <= best_lb):
            licznik_pp += 1

            print("PP numer",licznik_pp,"LB:", left_branch.lb, "Path:", [list(x) for x in left_branch.path], "\nCost matrix:\n",
                  left_branch.cost_matrix, "\n")
            #redukcja macierzy i zwiększenie LB
            new_cost = left_branch.reduce_cost_matrix()
            left_branch.lb = left_branch.lb + new_cost

            #zamykamy PP jeżeli LB > wartosci odcinającej
            if left_branch.lb > best_lb:
                break
            
            #wyznaczenie odcinaka o max optymistycznym koszcie wyłączenia
            new_vertex = left_branch.choose_new_vertex()
            #dodanie nowego wierzchołka do scieżki
            left_branch.path.append((new_vertex.coordinates))
            PP_right_cm = copy.deepcopy(left_branch.cost_matrix)
            #wykreslenie wiersza i kolumny i zabronienie podcyklu
            left_branch.update_cost_matrix((new_vertex.coordinates))
            
            #podzielenie PP na 2 
            new_lower_bound = left_branch.lb + new_vertex.cost

            lifo.insert(0, create_right_branch_matrix(PP_right_cm, new_vertex.coordinates,
                                                   new_lower_bound, copy.deepcopy(left_branch.path[0:-1])))
            
        if (left_branch.lb <= best_lb):
            new_path = left_branch.get_path()
            best_lb = get_optimal_cost(new_path, org_cm)

            # dodanie rozw do listy rozwiązań całego problemu
            end_solutions.append((get_optimal_cost(new_path, org_cm), new_path))

    #zwrócenie tylko najlepsych rozwiązań
    return filter_solutions(end_solutions)


def main():
    # matrix = np.array([[inf, 10, 8,   19, 12],
    #                  [10, inf, 20,  6,  3],
    #                  [8,   20, inf, 4,  2],
    #                  [19,  6,  4, inf,  7],
    #                  [12,  3,  2,   7, inf]])
    matrix = np.array(
       [[inf, 3, 4, 20, 7, 9, 11, 7, 20, 1],
        [3, inf, 4, 6, 2, 3, 2, 3, 4,  5],
        [4, 4, inf, 5, 8, 4, 7, 1, 2,  6],
        [20, 6, 5, inf, 6, 4, 2, 2, 2,  3],
        [7, 2, 8, 6, inf, 5, 11, 3, 1, 5],
        [9, 3, 4, 4, 5, inf, 4, 4, 5,  5],
        [11, 2, 7, 2, 11, 4, inf, 4, 4, 3],
        [7, 3, 1, 2, 3,  4,  4, inf, 5, 7],
        [20, 4, 2, 2, 1,  5,  4, 5, inf, 2],
        [1, 5, 6, 3, 5,  5,  3, 7, 2, inf]])


    print("Macierz początkowa:\n", matrix, "\n")
    sol = solve_tsp(matrix)
    for i in sol:
        print("rozwiązanie optymalne koszt:", i[0], "ścieżka indeksów:", i[1],"\n")

main()
