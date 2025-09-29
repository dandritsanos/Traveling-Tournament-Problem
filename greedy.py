import numpy as np
from ttp import Solution
import random

class GreedySolver:

    def __init__(self, n, distance_matrix):
        self.n = n
        self.distance_matrix = distance_matrix

    def create_solution(self):
        # Generate initial rounds
        rounds = self.generate_initial_rounds()

        # Reshape rounds to fit solution class
        schedule = np.zeros((self.n, len(rounds)), dtype=int)
        for r, round in enumerate(rounds):
            for team, opponent in round.items():
                schedule[team - 1, r] = opponent

        # Create solution
        solution = Solution(self.n, self.distance_matrix)
        solution.set_schedule(schedule)
        
        return solution

    def generate_initial_rounds(self):
        print("Generating initial rounds...\n")
        n = self.n
        attempts = 0
        print("Starting to assign rounds... (max 10 attempts)\n")
        while attempts < 10:
            attempts += 1
            current_locations = [i for i in range(self.n)]
            rounds = []
            required_matches = {}
            for team in range(self.n):
                team += 1
                required_matches[team] = set(range(-n, n+1)) - {team, -team, 0}
            try:
                while any(required_matches.values()):
                    assigned_teams = self.generate_next_round(self.distance_matrix, current_locations, required_matches)
                    rounds.append(assigned_teams)

                    for team, opponent in assigned_teams.items():
                        # Update current location
                        if opponent > 0:
                            current_locations[team - 1] = team - 1
                        else:
                            current_locations[team - 1] = abs(opponent) - 1

                        # Remove played match from required matches
                        required_matches[team].remove(opponent)

                return rounds
            except:
                print(f"No solution (full schedule) on attempt {attempts}, retrying...")

    def generate_next_round(self, distance_matrix, current_locations, required_opponents):
        attempts = 0

        # Necessary as some greedy attempts may result in an unsolvable start
        while attempts < 10:
            attempts += 1

            try:
                assigned_teams = {}
                order = list(range(self.n))
                # Necessary to randomise order to avoid unsolvable starts
                random.shuffle(order)

                for team in order:
                    # Check if team not already has a match
                    if team + 1 not in assigned_teams:

                        # For each possible opponent find the distance
                        team += 1
                        current_location = current_locations[team - 1]
                        possible_opponents = required_opponents[team]
                        distances = {}

                        for n in possible_opponents:

                            # Play home match
                            if n > 0:
                                # Find distance from current location to home location
                                distances[n] = distance_matrix[current_location][team-1]

                            # Play away match
                            else:
                                # Find distance from current location to opponent's home location
                                distances[n] = distance_matrix[current_location][abs(n)-1]

                        # Select opponent with minimum distance that is not in assigned_teams
                        selected_opponent = min((n for n in distances if abs(n) not in assigned_teams), key=distances.get)
                        assigned_teams[team] = selected_opponent

                        # What to do if playing home
                        if selected_opponent > 0:
                            # Make sure the enemy is assigned to play away
                            assigned_teams[abs(selected_opponent)] = -team
                        else:
                            # Make sure the enemy is assigned to play home
                            assigned_teams[abs(selected_opponent)] = team

                    else:
                        # Team already has a match, nothing to do as the main result (assigned_teams) already contains the match.
                        pass

                return assigned_teams
            
            except:
                print(f"Failed to assign all teams (round_basis) on attempt {attempts}, retrying...")


if __name__ == "__main__":
    n = 4
    distance_matrix = np.array([[0, 10, 15, 20],
                                [10, 0, 35, 25],    
                                [15, 35, 0, 30],
                                [20, 25, 30, 0]])

    solver = GreedySolver(n, distance_matrix)
    sol = solver.create_solution()

    print("\nGenerated Schedule:")
    for r, round in enumerate(sol.schedule):
        print(f"Team {r + 1}: {round}")

    sol.check_constraints()
    print(sol.calculate_cost())