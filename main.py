from ttp import Solution
from greedy import GreedySolver
import numpy as np
import random

if __name__ == "__main__":
    n = 8
    distance_matrix = np.array([[0, 10, 15, 20, 25, 30, 35, 40],
                                [10, 0, 35, 25, 30, 40, 50, 60],
                                [15, 35, 0, 30, 25, 35, 45, 55],
                                [20, 25, 30, 0, 15, 25, 35, 45],
                                [25, 30, 25, 15, 0, 20, 30, 40],
                                [30, 40, 35, 25, 20, 0, 15, 25],
                                [35, 50, 45, 35, 30, 15, 0, 20],
                                [40, 60, 55, 45, 40, 25, 20, 0]])

    solver = GreedySolver(n, distance_matrix)
    sol = solver.create_solution()

    print("\nGenerated Schedule:")
    for r, round in enumerate(sol.schedule):
        print(f"Team {r + 1}: {round}")

    sol.check_constraints()
    print(sol.calculate_cost())