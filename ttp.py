import numpy as np

class Solution:

    def __init__(self, n, distance_matrix):
        self.n = n
        self.rounds = 2*(n-1)
        self.distance_matrix = distance_matrix
        self.schedule = np.zeros((self.n, self.rounds), dtype=int)
        self.abs_schedule = np.abs(self.schedule)

    def set_schedule(self, schedule):
        self.schedule = schedule
        self.abs_schedule = np.abs(schedule)

    def calculate_cost(self):
        cost = 0
        current_locations = np.array(range(self.n))
        for round in range(self.rounds):
            next_locations = current_locations.copy()
            for team in range(self.n):
                opponent = self.schedule[team, round]
                if opponent > 0:
                    # Game is at home
                    cost += self.distance_matrix[current_locations[team], team]
                    next_locations[team] = team
                else:
                    # Game is away
                    cost += self.distance_matrix[current_locations[team], abs(opponent) - 1]
                    next_locations[team] = abs(opponent) - 1

            current_locations = next_locations
        return cost

def check_constraints(schedule, n):
    c1 = check_constraint_1(schedule, n)
    c2 = check_constraint_2(schedule, n)
    c3 = check_constraint_3(schedule, n)
    print(f"Constraint 1: {c1}, Constraint 2: {c2}, Constraint 3: {c3}")
    return (c1 and c2 and c3)

def check_constraint_1(schedule, n):
    """Each team plays against each other team exactly once at home and once away."""
    n = n
    rounds = 2 * (n - 1)
    schedule = schedule

    # Check that each team plays against each other team exactly once at home and once away
    for team in range(n):
        home_games = set()
        away_games = set()
        for round in range(rounds):
            opponent = schedule[team, round]
            if opponent == 0:
                continue  # Bye week
            if opponent > 0:
                home_games.add(opponent)
            else:
                away_games.add(-opponent)
        if len(home_games) != n - 1 or len(away_games) != n - 1:
            return False
    return True

def check_constraint_2(schedule, n, UpperBound = 3):
    """No team plays more than UpperBound consecutive home or away games."""
    rounds = 2 * (n - 1)

    for team in range(n):
        consecutive_home = 0
        consecutive_away = 0
        for round in range(rounds):
            opponent = schedule[team, round]
            if opponent > 0:
                consecutive_home += 1
                consecutive_away = 0
            elif opponent < 0:
                consecutive_away += 1
                consecutive_home = 0
            else:
                consecutive_home = 0
                consecutive_away = 0
            
            if consecutive_home > UpperBound or consecutive_away > UpperBound:
                return False
    return True
    
def check_constraint_3(schedule, n):
    """Each match has one team playing home, and one team playing away"""
    rounds = 2 * (n - 1)

    for round in range(rounds):
        home_teams = set()
        away_teams = set()
        for team in range(n):
            opponent = schedule[team, round]
            if opponent > 0:
                home_teams.add(team + 1)
            elif opponent < 0:
                away_teams.add(team + 1)
        
        if len(home_teams) != len(away_teams):
            return False
        
        for team in home_teams:
            if -team not in schedule[:, round]:
                return False
        for team in away_teams:
            if team not in schedule[:, round]:
                return False
    return True
    
def check_constraint_4(schedule, n):
    # No team plays himself
    rounds = 2 * (n - 1)
    for team in range(n):
        for round in range(rounds):
            if abs(schedule[team, round]) == team + 1:
                return False
    return True

if __name__ == "__main__":
    n = 4
    distance_matrix = np.array([[0, 10, 15, 20],
                                [10, 0, 35, 25],
                                [15, 35, 0, 30],
                                [20, 25, 30, 0]])
    
    schedule = np.array([[ 2, 3, 4, -2, -3, -4],
                         [ -1, 4, 3, 1, -4, -3],
                         [ 4, -1, -2, -4, 1, 2],
                         [ -3, -2, -1, 3, 2, 1]])
    
    solution = Solution(n, distance_matrix)
    solution.set_schedule(schedule)
    
    # Stress test compute 1000 times:
    for _ in range(1000):
        cost = solution.calculate_cost()
    
    constraints = check_constraints(solution.schedule, solution.n)

    print("Cost of the schedule:", solution.calculate_cost())
