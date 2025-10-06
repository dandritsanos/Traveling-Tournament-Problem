
import sys
import os
import math
import itertools as it
import xml.etree.ElementTree as ET
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB



def parse_robinx_xml(xml_path):
    """Parse a RobinX-format TTP instance (like NL6.xml).
    Returns: n, rounds, teams(list of ids), dist[n][n], params(dict)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Teams: the RobinX schema doesn't always explicitly list teams for 'A' datatype
    # but distances list team indices. We infer n from the maximum team id.
    dists = []
    dist_map = defaultdict(dict)
    for d in root.iter('distance'):
        i = int(d.attrib['team1'])
        j = int(d.attrib['team2'])
        val = int(d.attrib['dist'])
        dist_map[i][j] = val

    # infer team ids
    teams = sorted(set(dist_map.keys()) | set(k for m in dist_map.values() for k in m.keys()))
    n = max(teams) + 1

    # number of Round Robins
    rr = 2
    fmt = root.find('./Structure/Format/numberRoundRobin')
    if fmt is not None and fmt.text is not None:
        try:
            rr = int(fmt.text.strip())
        except:
            rr = 2

    # Double round-robin => rounds = rr*(n-1)
    R = rr * (n - 1)

    # Build dense distance matrix and make sure diagonal is zero
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            else:
                # try to read either i->j or j->i; most instances are symmetric
                if j in dist_map.get(i, {}):
                    dist[i][j] = dist_map[i][j]
                elif i in dist_map.get(j, {}):
                    dist[i][j] = dist_map[j][i]
                else:
                    raise ValueError(f"Missing distance between {i} and {j} in XML")

    params = {
        'SE1_min': 1,  # no immediate rematch between a pair
        'SE1_max': R,  # set to R if not binding in instance
        'max_streak': 3,  # from CA3 intp=4 max=3 (prevents 4 consecutive H or A)
        'window': 4,
        'return_home_end': True,  # standard TR objective includes return to home after last round
    }

    # Try to read separation constraint if present
    se1 = root.find('./Constraints/SeparationConstraints/SE1')
    if se1 is not None:
        try:
            params['SE1_min'] = int(se1.attrib.get('min', params['SE1_min']))
            params['SE1_max'] = int(se1.attrib.get('max', params['SE1_max']))
        except:
            pass

    # Capacity CA3 => at most 3 H (and A) in any window of 4 rounds, which is equivalent to streak <= 3
    # We'll enforce as no 4 consecutive H or A.

    return n, R, list(range(n)), dist, params


def build_ttp_tr_model(n, R, teams, dist, params):
    m = gp.Model('TTP_TR_NL6')

    # Sets
    I = teams  # teams 0..n-1
    J = teams
    T = list(range(R))  # rounds 0..R-1

    # Decision vars
    # x[i,j,t] = 1 if team i plays AWAY at team j in round t (ordered pair, i!=j)
    x = m.addVars(((i, j, t) for i in I for j in J if i != j for t in T), vtype=GRB.BINARY, name='x')

    # Helper: v[i,k,t] = 1 if team i is AT venue k in round t (k==i means home)
    v = m.addVars(((i, k, t) for i in I for k in J for t in T), vtype=GRB.BINARY, name='v')

    # Transition vars to linearize travel: z[i,t,k,l] = 1 if team i is at k in t and at l in t+1
    z = m.addVars(((i, t, k, l) for i in I for t in range(R-1) for k in J for l in J), vtype=GRB.BINARY, name='z')

    # 1) Exactly one game per team per round
    for i in I:
        for t in T:
            home_games = gp.quicksum(x[j, i, t] for j in I if j != i)
            away_games = gp.quicksum(x[i, j, t] for j in I if j != i)
            m.addConstr(home_games + away_games == 1, name=f"one_game_{i}_{t}")

    # 2) Double round robin: for each ordered pair (i,j), play exactly once away (i at j)
    for i in I:
        for j in I:
            if i == j:
                continue
            m.addConstr(gp.quicksum(x[i, j, t] for t in T) == 1, name=f"once_{i}_at_{j}")

    # 3) Venue consistency: v linked to x
    #   - v[i,i,t] == home(i,t) = sum_j x[j,i,t]
    #   - for k != i: v[i,k,t] == x[i,k,t]
    for i in I:
        for t in T:
            m.addConstr(v[i, i, t] == gp.quicksum(x[j, i, t] for j in I if j != i), name=f"v_home_{i}_{t}")
            for k in I:
                if k == i:
                    continue
                m.addConstr(v[i, k, t] == x[i, k, t], name=f"v_away_{i}_{k}_{t}")
            # Exactly one venue per round for team i
            m.addConstr(gp.quicksum(v[i, k, t] for k in I) == 1, name=f"one_venue_{i}_{t}")

    # 4) No immediate rematches (SE1 min=1). Also enforce max separation if provided (usually nonbinding for R=10)
    se1_min = params.get('SE1_min', 1)
    se1_max = params.get('SE1_max', R)

    # Prevent playing the same opponent in consecutive rounds (either order/home-away)
    for i in I:
        for j in I:
            if i >= j:
                continue
            for t in range(R - 1):
                m.addConstr(x[i, j, t] + x[j, i, t] + x[i, j, t + 1] + x[j, i, t + 1] <= 1, name=f"no_back_to_back_{i}_{j}_{t}")

    # A simple (weaker) way to respect max separation is to avoid scheduling both matches too close to the edges,
    # but since NL6 has SE1 max=10 and R=10, we skip explicit max separation constraints here.

    # 5) No 4 consecutive home or 4 consecutive away games (CA3 with window=4, max=3)
    max_streak = params.get('max_streak', 3)
    window = params.get('window', 4)
    assert window == 4 and max_streak == 3
    for i in I:
        for t in range(R - (window - 1)):
            # Home count in window <= 3
            m.addConstr(gp.quicksum(v[i, i, t + w] for w in range(window)) <= max_streak,
                        name=f"home_window_{i}_{t}")
            # Away count in window <= 3
            m.addConstr(gp.quicksum(1 - v[i, i, t + w] for w in range(window)) <= max_streak,
                        name=f"away_window_{i}_{t}")

    # 6) Transition linking for travel (z)
    for i in I:
        for t in range(R - 1):
            # Flow conservation over k,l
            m.addConstr(gp.quicksum(z[i, t, k, l] for k in I for l in I) == 1, name=f"one_transition_{i}_{t}")
            for k in I:
                m.addConstr(gp.quicksum(z[i, t, k, l] for l in I) == v[i, k, t], name=f"z_out_{i}_{t}_{k}")
            for l in I:
                m.addConstr(gp.quicksum(z[i, t, k, l] for k in I) == v[i, l, t + 1], name=f"z_in_{i}_{t}_{l}")

    # Objective: Total travel distance (start -> first venue) + transitions + (last venue -> home)
    expr = gp.LinExpr()

    # start legs
    for i in I:
        for k in I:
            expr += dist[i][k] * v[i, k, 0]

    # transitions
    for i in I:
        for t in range(R - 1):
            for k in I:
                for l in I:
                    expr += dist[k][l] * z[i, t, k, l]

    # return home
    if params.get('return_home_end', True):
        for i in I:
            for k in I:
                expr += dist[k][i] * v[i, k, R - 1]

    m.setObjective(expr, GRB.MINIMIZE)



        # --- Symmetry breaking ---
    # Optional: a few symmetry breakers to speed up solving
    # (1) Fix team 0's first round to be at home (if feasible)
    m.addConstr(gp.quicksum(x[j, 0, 0] for j in I if j != 0) == 1, name="sym_team0_home_round0")



    m.Params.OutputFlag = 1
    m.Params.MIPGap = 0.0

    return m, x, v


def extract_schedule(n, R, x):
    """Return a schedule structure and per-team home/away strings.
    schedule[round] = list of (away, home)
    hadv[team] = string like 'HAAH...'
    """
    schedule = [[] for _ in range(R)]
    hadv = {i: [] for i in range(n)}

    for t in range(R):
        # games: for all i!=j with x[i,j,t] == 1
        pairs = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if x[i, j, t].X > 0.5:
                    pairs.append((i, j))  # i at j
        # build per-team H/A
        played = set()
        for (i, j) in pairs:
            schedule[t].append((i, j))
            played.add(i)
            played.add(j)
        for i in range(n):
            if i not in played:
                # shouldn't happen — every team must appear exactly once per round
                pass
        # H/A strings
        for i in range(n):
            ha = 'A' if any(i == away for (away, home) in schedule[t]) else 'H'
            hadv[i].append(ha)

    hadv = {i: ''.join(hadv[i]) for i in range(n)}
    return schedule, hadv


def main():
    xml_path = 'NL6.xml'
    if len(sys.argv) >= 2:
        xml_path = sys.argv[1]
    if not os.path.exists(xml_path):
        raise SystemExit(f"Could not find instance file at '{xml_path}'.")

    n, R, teams, dist, params = parse_robinx_xml(xml_path)
    print(f"Parsed instance: n={n} teams, rounds={R}, params={params}")

    m, x, v = build_ttp_tr_model(n, R, teams, dist, params)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        print(f"Solve ended with status {m.status}. Best obj = {m.ObjVal if m.SolCount else 'n/a'}")
        return

    print(f"Optimal total travel distance: {m.ObjVal:.0f}")

    schedule, hadv = extract_schedule(n, R, x)

    print("\nRound-by-round schedule (A@H):")
    for t, games in enumerate(schedule, start=1):
        games_str = '  '.join([f"{a}@{h}" for (a, h) in games])
        print(f"Round {t:2d}:  {games_str}")

    print("\nPer-team H/A patterns:")
    for i in range(n):
        print(f"Team {i}: {hadv[i]}")


if __name__ == '__main__':
    main()
