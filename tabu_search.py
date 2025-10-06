from ttp import Solution, check_constraint_1, check_constraint_2, check_constraint_3, check_constraints, check_constraint_4
import numpy as np
from collections import defaultdict

def duplicate_indexes(lst):
    """
    Returns a sorted list of indexes of all elements that appear more than once.
    """
    seen = {}
    dup_indexes = set()

    for i, val in enumerate(lst):
        if val in seen:
            dup_indexes.add(seen[val])  # first occurrence
            dup_indexes.add(i)          # current duplicate
        else:
            seen[val] = i

    return sorted(dup_indexes)

class TabuSearch:

    def __init__(self, initial_solution, tabu_size=10, max_iterations=100):
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.tabu_list = []
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations

    def swap_teams(self, schedule, t1, t2, r):
        # Swap the games of team t1 and t2 in round r
        opponent_t1 = schedule[t1, r]
        opponent_t2 = schedule[t2, r]
        
        schedule[t1, r] = opponent_t2
        schedule[t2, r] = opponent_t1

        if opponent_t1 > 0:
            # Is t1 playing at home
            schedule[abs(opponent_t1) - 1, r] = -(t2 + 1)
        else:
            schedule[abs(opponent_t1) - 1, r] = (t2 + 1)

        if opponent_t2 > 0:
            # Is t2 playing at home
            schedule[abs(opponent_t2) - 1, r] = -(t1 + 1)
        else:
            schedule[abs(opponent_t2) - 1, r] = (t1 + 1)

        return schedule

    def recovery_n1(self, schedule, t1, t2, r, max_length = 9):
        
        altered_rounds = set()
        
        first_new_schedule = self.swap_teams(schedule, t1, t2, r).copy()
        if not check_constraint_4(first_new_schedule, first_new_schedule.shape[0]):
            print("Started by switching teams into itself, break chain")
            return False
        altered_rounds.add(r)

        while len(altered_rounds) < max_length and (check_constraint_1(first_new_schedule, first_new_schedule.shape[0]) == False or 
                                                    check_constraint_4(first_new_schedule, first_new_schedule.shape[0]) == False
            ):
            # Find duplicates in t1 schedule
            dup_indexes = duplicate_indexes(first_new_schedule[t1, :])
            next_switch = (set(dup_indexes) - altered_rounds).pop()
            
            self.swap_teams(first_new_schedule, t1, t2, next_switch)
            altered_rounds.add(next_switch)

        if len(altered_rounds) == max_length:
            print("Max length reached, returning None")
            return False
        
        else:
            print(f"Final schedule after recovery in {len(altered_rounds)} steps:")
            print(first_new_schedule)
            return first_new_schedule
    
    def recovery_n2(self, schedule, t1, r1, r2):
        #Swap the games of team t1 in rounds r1 and r2
        opponent_r1 = abs(schedule[t1, r1])
        opponent_r2 = abs(schedule[t1, r2])

        schedule_changed = schedule.copy()
        # Array which indexes represent teams, and values are sets of their opponents in rounds r1 and r2
        teams_opponents = [set() for _ in range(schedule.shape[0])]  

        for team in range(schedule.shape[0]):
            opponent_r1 = abs(schedule_changed[team, r1])
            opponent_r2 = abs(schedule_changed[team, r2])
            teams_opponents[team].add(opponent_r1)
            teams_opponents[team].add(opponent_r2)


        stack = [t1]
        visited = set()
        teams_to_swap = []

        # DFS to find all teams connected to t1 through opponents in r1 and r2
        while stack:
            current_team = stack.pop()
            if current_team not in visited:
                visited.add(current_team)
                teams_to_swap.append(current_team)
                for opponent in teams_opponents[current_team]:
                    if opponent -1 not in visited:
                        stack.append(opponent - 1)

        # Now swap the games in rounds r1 and r2 for all teams in teams_to_swap (team correlated with each other through opponents in r1 and r2)
        for team in teams_to_swap:
            schedule_changed[team, r1], schedule_changed[team, r2] = schedule_changed[team, r2], schedule_changed[team, r1]

        return schedule_changed

    def recovery_n3(self, schedule, r1, r2):
        # Swap the rounds r1 and r2 for all teams
        schedule_changed = schedule.copy()
        schedule_changed[:, r1], schedule_changed[:, r2] = schedule_changed[:, r2].copy(), schedule_changed[:, r1].copy()
        return schedule_changed

    def recovery_n4(self, schedule, t1, t2):
        # Swap the entire schedules of teams t1 and t2 (except for their direct matches)
        schedule_changed = schedule.copy()
        for r in range(schedule.shape[1]):
            if abs(schedule_changed[t1, r]) != t2 + 1 and abs(schedule_changed[t2, r]) != t1 + 1:
                schedule_changed[t1, r], schedule_changed[t2, r] = schedule_changed[t2, r], schedule_changed[t1, r]
        return schedule_changed
    
    def recovery_n5(self, schedule, t1, t2):
        # Swap home/away status of all matches between teams t1 and t2
        schedule_changed = schedule.copy()
        for r in range(schedule.shape[1]):
            if abs(schedule_changed[t1, r]) == t2 + 1:
                schedule_changed[t1, r] = -schedule_changed[t1, r]
                schedule_changed[t2, r] = -schedule_changed[t2, r]
        return schedule_changed

    def get_neighbors(self, solution):
        neighbors = []

        for t1 in range(solution.n):
            for t2 in range(solution.n):
                if t1 != t2:
                    # Note that t1 and t2 are 0-indexed
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

    TS = TabuSearch(solution)

    print(TS.current_solution.schedule)
    #TS.current_solution.schedule = TS.recovery_n2(TS.current_solution.schedule, 1, 0, 3)
    #TS.current_solution.schedule = TS.recovery_n3(TS.current_solution.schedule, 0, 3)    
    #TS.current_solution.schedule = TS.recovery_n4(TS.current_solution.schedule, 2, 3)
    TS.current_solution.schedule = TS.recovery_n5(TS.current_solution.schedule, 0, 1)
    print(TS.current_solution.schedule)
    print(check_constraints(TS.current_solution.schedule, TS.current_solution.n))

