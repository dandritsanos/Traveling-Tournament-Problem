from ttp import Solution, check_constraint_1, check_constraint_2, check_constraint_3, check_constraints, check_constraint_4
from tabu_search import TabuSearch
from greedy import GreedySolver
import numpy as np
import random
import pandas as pd

if __name__ == "__main__":
    n = 6

    distance_matrix = np.array([[0, 10, 15, 20, 25, 30],
                                [10, 0, 35, 25, 30, 20],
                                [15, 35, 0, 30, 20, 25],
                                [20, 25, 30, 0, 15, 10],
                                [25, 30, 20, 15, 0, 35],
                                [30, 20, 25, 10, 35, 0]])

    solver = GreedySolver(n, distance_matrix)
    sol = solver.create_solution()

    print("\nGenerated Schedule:")
    for r, round in enumerate(sol.schedule):
        print(f"Team {r + 1}: {round}")

    print(check_constraints(sol.schedule, n))
    print(sol.calculate_cost())

    TS = TabuSearch(sol)
    
    print(check_constraints(TS.recovery_n1(TS.current_solution.schedule, 0, 3, 4), n))