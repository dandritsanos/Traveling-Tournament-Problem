from ttp import Solution
import numpy as np


class TabuSearch:

    def __init__(self, initial_solution, tabu_size=10, max_iterations=100):
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.tabu_list = []
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations

    def recovery_n1(self, schedule, t1, t2, r):
        # Should return the new schedule after swapping teams t1 and t2 in round r and doing recovery
        pass
    
    def recovery_n2(self, schedule, t1, r1, r2):
        # Should return the new schedule after swapping rounds r1 and r2 for team t1 and doing recovery
        pass

    def get_neighbors(self, solution):
        neighbors = []

        for t1 in range(solution.n):
            for t2 in range(solution.n):
                if t1 != t2:
                    for r in range(solution.rounds):
                        new_schedule = self.recovery_n1(solution.schedule.copy(), t1, t2, r)
                        new_solution = Solution(solution.n, solution.distance_matrix)
                        new_solution.set_schedule(np.array(new_schedule))
                        neighbors.append(new_solution)
        
        for r1 in range(solution.rounds):
            for r2 in range(solution.rounds):
                if r1 != r2:
                    for t in range(solution.n):
                        new_schedule = self.recovery_n2(solution.schedule.copy(), t, r1, r2)
                        new_solution = Solution(solution.n, solution.distance_matrix)
                        new_solution.set_schedule(np.array(new_schedule))
                        neighbors.append(new_solution)

        return neighbors


    def search(self):
        iterations = 0
        while iterations < self.max_iterations:

            neighbors = self.get_neighbors(self.current_solution)
            best_neighbor = None
            best_cost = float('inf')

            for neighbor in neighbors:
                if neighbor not in self.tabu_list:
                    cost = neighbor.calculate_cost()
                    if cost < best_cost:
                        best_cost = cost
                        best_neighbor = neighbor

            if best_neighbor is None:
                break

            self.current_solution = best_neighbor
            if best_cost < self.best_solution.calculate_cost():
                self.best_solution = best_neighbor

            self.tabu_list.append(best_neighbor)
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)

            iterations += 1

        return self.best_solution
    

if __name__ == "__main__":
    n = 4
    distance_matrix = np.array([[0, 10, 15, 20],
                                [10, 0, 35, 25],
                                [15, 35, 0, 30],
                                [20, 25, 30, 0]])
    # Team 1: [ 2  3  4 -3 -4 -2]
    # Team 2: [-1  4  3 -4 -3  1]
    # Team 3: [-4 -1 -2  1  2  4]
    # Team 4: [ 3 -2 -1  2  1 -3]
    schedule = np.array([[ 2, 3, 4, -3, -4, -2],
                         [ -1, 4, 3, -4, -3, 1],
                         [ -4, -1, -2, 1, 2, 4],
                         [ 3, -2, -1, 2, 1, -3]])
    
    solution = Solution(n, distance_matrix)
    solution.set_schedule(schedule)

    print("Cost of the schedule:", solution.calculate_cost())
    print("Constraints satisfied:", solution.check_constraints())

    tabu_search = TabuSearch(solution, tabu_size=5, max_iterations=50)
    swap_example = tabu_search.recovery_n1(schedule.copy(), 1, 2, 2)
    print("Schedule after N1 swap (teams 1 and 2 in round 2):")
    print(np.array(swap_example))
    new_solution = Solution(n, distance_matrix)
    new_solution.set_schedule(np.array(swap_example))
    print("Cost after N1 swap:", new_solution.calculate_cost())
    print("Constraints satisfied after N1 swap:", new_solution.check_constraints())