import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import itertools as it

# -------------------------
# Parse RobinX XML instance
# -------------------------
def parse_robinx_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    dist_map = defaultdict(dict)
    for d in root.iter('distance'):
        i = int(d.attrib['team1'])
        j = int(d.attrib['team2'])
        val = int(d.attrib['dist'])
        dist_map[i][j] = val

    teams = sorted(set(dist_map.keys()) | set(k for m in dist_map.values() for k in m.keys()))
    n = max(teams) + 1

    rr = 2
    fmt = root.find('./Structure/Format/numberRoundRobin')
    if fmt is not None and fmt.text is not None:
        try:
            rr = int(fmt.text.strip())
        except:
            rr = 2

    R = rr * (n - 1)

    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            else:
                if j in dist_map.get(i, {}):
                    dist[i][j] = dist_map[i][j]
                elif i in dist_map.get(j, {}):
                    dist[i][j] = dist_map[j][i]
                else:
                    raise ValueError(f"Missing distance between {i} and {j}")

    params = {
        'SE1_min': 1,
        'SE1_max': R,
        'max_streak': 3,
        'min_streak': 1,
        'window': 4,
        'return_home_end': True
    }

    ca3 = root.find('./Constraints/CA3')
    if ca3 is not None:
        try:
            params['max_streak'] = int(ca3.attrib.get('max', 3))
            params['window'] = int(ca3.attrib.get('window', 4))
            params['min_streak'] = int(ca3.attrib.get('min', 1))
        except:
            pass

    se1 = root.find('./Constraints/SeparationConstraints/SE1')
    if se1 is not None:
        try:
            params['SE1_min'] = int(se1.attrib.get('min', params['SE1_min']))
            params['SE1_max'] = int(se1.attrib.get('max', params['SE1_max']))
        except:
            pass

    return n, R, list(range(n)), dist, params

# -------------------------
# Column: feasible itinerary for a team
# -------------------------
class Column:
    def __init__(self, team, sequence, distance):
        self.team = team
        self.sequence = sequence
        self.distance = distance

# -------------------------
# Generate feasible sequences
# -------------------------
def generate_feasible_sequences(team, R, n, dist, params, existing_schedule):
    sequences = []

    def extend(seq, t, last_home, streak_home, streak_away):
        if t == R:
            # Add distance back home if needed
            total_dist = dist[team][seq[0]] + sum(dist[seq[i]][seq[i+1]] for i in range(R-1))
            if params.get('return_home_end', True):
                total_dist += dist[seq[-1]][team]
            sequences.append(Column(team, seq[:], total_dist))
            return

        for next_venue in range(n):
            next_home = (next_venue == team)

            # Check max streak
            if next_home and streak_home >= params['max_streak']:
                continue
            if not next_home and streak_away >= params['max_streak']:
                continue

            # Avoid immediate rematch
            conflict = False
            for other_team, other_seq in existing_schedule.items():
                if t < len(other_seq):
                    if other_seq[t] == next_venue and next_venue != team:
                        conflict = True
                        break
            if conflict:
                continue

            # Update streaks
            next_streak_home = streak_home + 1 if next_home else 0
            next_streak_away = streak_away + 1 if not next_home else 0

            seq.append(next_venue)
            extend(seq, t+1, next_home, next_streak_home, next_streak_away)
            seq.pop()

    # Start with first round options
    for first in range(n):
        if first == team:
            continue
        seq = [first]
        extend(seq, 1, first==team, 1 if first==team else 0, 0 if first==team else 1)

    sequences.sort(key=lambda c: c.distance)
    return sequences[:50]

# -------------------------
# Master problem
# -------------------------
def solve_master_problem(n, R, dist, params):
    schedule = {}
    columns = []

    for team in range(n):
        seqs = generate_feasible_sequences(team, R, n, dist, params, schedule)
        if not seqs:
            raise RuntimeError(f"No feasible sequence for team {team}")
        best_col = seqs[0]
        schedule[team] = best_col.sequence
        columns.append(best_col)

    m = gp.Model('TTP_CG_Master')
    m.setParam('OutputFlag', 1)
    m.setParam('TimeLimit', 300)  # 5-minute limit
    m.setParam('MIPGap', 0.01)

    col_vars = m.addVars(len(columns), vtype=GRB.BINARY, obj=[c.distance for c in columns], name='col')

    for team in range(n):
        idxs = [i for i, c in enumerate(columns) if c.team == team]
        m.addConstr(gp.quicksum(col_vars[i] for i in idxs) == 1)

    m.setObjective(gp.quicksum(col_vars[i]*columns[i].distance for i in range(len(columns))), GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        raise RuntimeError("No feasible solution found")

    final_schedule = {}
    ha_patterns = {i: [] for i in range(n)}
    total_dist = 0

    for i, c in enumerate(columns):
        if col_vars[i].X > 0.5:
            final_schedule[c.team] = c.sequence
            seq = c.sequence
            total_dist += c.distance
            for t, v in enumerate(seq):
                ha_patterns[c.team].append('H' if v == c.team else 'A')

    return final_schedule, ha_patterns, total_dist

# -------------------------
# Main
# -------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <instance.xml>")
        sys.exit(1)

    xml_file = sys.argv[1]
    print(f"Solving {xml_file}...")
    n, R, teams, dist, params = parse_robinx_xml(xml_file)
    schedule, ha_patterns, total_dist = solve_master_problem(n, R, dist, params)

    print(f"\nTotal travel distance: {total_dist}")
    print("Round-by-round schedule (A@H):")
    for t in range(R):
        round_games = []
        for team in range(n):
            opponent = schedule[team][t]
            if opponent != team:
                round_games.append(f"{team}@{opponent}")
        print(f"Round {t+1}: {', '.join(round_games)}")

    print("\nPer-team H/A patterns:")
    for team in teams:
        print(f"Team {team}: {''.join(ha_patterns[team])}")

if __name__ == '__main__':
    main()
