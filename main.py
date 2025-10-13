from ttp import Solution, check_constraint_1, check_constraint_2, check_constraint_3, check_constraints, check_constraint_4
from tabu_search import TabuSearch
from greedy import GreedySolver
import numpy as np
import random
import pandas as pd
from xml_loader import load_instance

if __name__ == "__main__":
    n, distance_matrix = load_instance('NL10.xml')

    solver = GreedySolver(n, distance_matrix)
    sol = solver.create_solution()

    print("\nGenerated Schedule:")
    for r, round in enumerate(sol.schedule):
        print(f"Team {r + 1}: {round}")

    print(check_constraints(sol.schedule, n))
    print(sol.calculate_cost())

    TS = TabuSearch(sol)
    TS.search()
    print("\nOptimized Schedule:")
    for r, round in enumerate(TS.best_solution.schedule):
        print(f"Team {r + 1}: {round}")