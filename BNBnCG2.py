import sys
import os
import math
import itertools as it
import xml.etree.ElementTree as ET
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

# -------------------------
# 1. Parse RobinX XML instance
# -------------------------
def parse_robinx_xml(xml_path):
    """Parse RobinX-format TTP instance (NLx.xml). Returns: n, R, teams, dist, params"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Distances
    dist_map = defaultdict(dict)
    for d in root.iter('distance'):
        i = int(d.attrib['team1'])
        j = int(d.attrib['team2'])
        val = int(d.attrib['dist'])
        dist_map[i][j] = val

    # Infer teams
    teams = sorted(set(dist_map.keys()) | set(k for m in dist_map.values() for k in m.keys()))
    n = max(teams) + 1

    # Round-robin count
    rr = 2
    fmt = root.find('./Structure/Format/numberRoundRobin')
    if fmt is not None and fmt.text is not None:
        try:
            rr = int(fmt.text.strip())
        except:
            rr = 2

    # Total rounds
    R = rr * (n - 1)

    # Dense distance matrix
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

    # Constraints and parameters
    params = {
        'SE1_min': 1,        # min separation
        'SE1_max': R,        # max separation
        'max_streak': 3,     # max H/A streak
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
# 2. Build Time-Indexed MILP (NL4/NL6)
# -------------------------
def build_ttp_tr_model(n, R, teams, dist, params):
    m = gp.Model('TTP_TR')

    I = teams
    J = teams
    T = list(range(R))

    # x[i,j,t] = 1 if i plays away at j in round t
    x = m.addVars(((i,j,t) for i in I for j in J if i!=j for t in T), vtype=GRB.BINARY, name='x')
    # v[i,k,t] = team i at venue k at round t (k==i -> home)
    v = m.addVars(((i,k,t) for i in I for k in J for t in T), vtype=GRB.BINARY, name='v')
    # z[i,t,k,l] = team i at k at t, at l at t+1
    z = m.addVars(((i,t,k,l) for i in I for t in range(R-1) for k in J for l in J), vtype=GRB.BINARY, name='z')

    # 1) Each team plays exactly once per round
    for i in I:
        for t in T:
            m.addConstr(gp.quicksum(x[j,i,t] for j in I if j!=i) + gp.quicksum(x[i,j,t] for j in I if j!=i) == 1)

    # 2) Double round-robin: each ordered pair once away
    for i in I:
        for j in I:
            if i==j: continue
            m.addConstr(gp.quicksum(x[i,j,t] for t in T) == 1)

    # 3) Venue consistency
    for i in I:
        for t in T:
            m.addConstr(v[i,i,t] == gp.quicksum(x[j,i,t] for j in I if j!=i))
            for k in I:
                if k==i: continue
                m.addConstr(v[i,k,t] == x[i,k,t])
            m.addConstr(gp.quicksum(v[i,k,t] for k in I) == 1)

    # 4) no immediate rematches
    for i in I:
        for j in I:
            if i >= j: continue
            for t in range(R-1):
                m.addConstr(x[i,j,t]+x[j,i,t]+x[i,j,t+1]+x[j,i,t+1] <= 1)

    # 5) max and min consecutive home/away
    window = params['window']
    max_streak = params['max_streak']
    min_streak = params.get('min_streak', 1)

    for i in I:
        for t in range(R - (window-1)):
            # maximum home
            m.addConstr(gp.quicksum(v[i,i,t+w] for w in range(window)) <= max_streak)
            # minimum home (optional)
            if min_streak > 0:
                m.addConstr(gp.quicksum(v[i,i,t+w] for w in range(window)) >= min_streak)
            # maximum away
            m.addConstr(gp.quicksum(1-v[i,i,t+w] for w in range(window)) <= max_streak)
            # minimum away (optional)
            if min_streak > 0:
                m.addConstr(gp.quicksum(1-v[i,i,t+w] for w in range(window)) >= min_streak)


    # 6) Transition linking
    for i in I:
        for t in range(R-1):
            m.addConstr(gp.quicksum(z[i,t,k,l] for k in I for l in J)==1)
            for k in I:
                m.addConstr(gp.quicksum(z[i,t,k,l] for l in J) == v[i,k,t])
            for l in I:
                m.addConstr(gp.quicksum(z[i,t,k,l] for k in I) == v[i,l,t+1])

    # Objective: travel distance
    expr = gp.LinExpr()
    for i in I:
        for k in I:
            expr += dist[i][k]*v[i,k,0]             # start to first venue
    for i in I:
        for t in range(R-1):
            for k in I:
                for l in I:
                    expr += dist[k][l]*z[i,t,k,l]
    if params.get('return_home_end',True):
        for i in I:
            for k in I:
                expr += dist[k][i]*v[i,k,R-1]
    m.setObjective(expr, GRB.MINIMIZE)

    # Symmetry breaker: team 0 home round 0
    m.addConstr(gp.quicksum(x[j,0,0] for j in I if j!=0) == 1)

    m.Params.OutputFlag = 1
    m.Params.MIPGap = 0.0
    return m, x, v

# -------------------------
# 3. Extract schedule and H/A
# -------------------------
def extract_schedule(n,R,x):
    schedule = [[] for _ in range(R)]
    hadv = {i: [] for i in range(n)}
    for t in range(R):
        for i in range(n):
            for j in range(n):
                if i==j: continue
                if x[i,j,t].X>0.5:
                    schedule[t].append((i,j))
        for i in range(n):
            ha = 'A' if any(i==away for (away,home) in schedule[t]) else 'H'
            hadv[i].append(ha)
    hadv = {i: ''.join(hadv[i]) for i in range(n)}
    return schedule,hadv

# -------------------------
# 4. Column generation solver (NL8/NL10)
# -------------------------
def solve_column_generation(inst):
    """
    Strong column-generation solver for NL8/NL10 instances.
    Generates feasible team columns using Berger schedules and solves the binary master problem.
    Returns: schedule (list of rounds of (away,home)), total_distance
    """
    import gurobipy as gp
    from gurobipy import GRB
    from collections import defaultdict

    n = inst['n']
    R = inst['R']
    teams = inst['teams']
    dist = inst['dist']
    params = inst['params']

    window = params.get('window', 4)
    max_streak = params.get('max_streak', 3)
    min_streak = params.get('min_streak', 1)

    # -------------------------
    # Build feasible columns for each team
    # Each column = full season schedule for the team (opponent, H/A)
    # -------------------------
    columns = defaultdict(list)

    def build_column(team_idx, opp_list):
        """Given team_idx and list of (opponent, H/A), build full column dict"""
        v = {}
        z = {}
        total_cost = 0
        last = team_idx
        for t, (opp, ha) in enumerate(opp_list):
            k = team_idx if ha == 'H' else opp
            v[(k, t)] = 1
            total_cost += dist[last][k]
            last = k
        total_cost += dist[last][team_idx]  # return home

        # build z
        for t in range(R - 1):
            k = next(key for key in range(n) if v.get((key, t), 0) == 1)
            l = next(key for key in range(n) if v.get((key, t + 1), 0) == 1)
            z[(t, k, l)] = 1

        # ensure full dicts
        v_full = {(k, t): int(v.get((k, t), 0)) for k in teams for t in range(R)}
        z_full = {(t, k, l): int(z.get((t, k, l), 0)) for t in range(R - 1) for k in teams for l in teams}

        return {'opp': [o for o, _ in opp_list],
                'ha': [ha for _, ha in opp_list],
                'v': v_full, 'z': z_full, 'cost': total_cost}

    # -------------------------
    # Initial Berger-style columns
    # -------------------------
    def initial_columns():
        cols = defaultdict(list)
        n_teams = n
        rounds = R
        half = n_teams // 2
        for i in range(n_teams):
            schedule_i = []
            arr = list(range(n_teams))
            for r in range(rounds):
                pairings = [(arr[k], arr[n_teams - 1 - k]) for k in range(half)]
                opp = None
                ha = 'H'
                for a, b in pairings:
                    if a == i:
                        opp = b
                        ha = 'H' if r % 2 == 0 else 'A'
                        break
                    elif b == i:
                        opp = a
                        ha = 'A' if r % 2 == 0 else 'H'
                        break
                schedule_i.append((opp, ha))
                # rotate all but first team
                arr = [arr[0]] + arr[-1:] + arr[1:-1]
            cols[i].append(build_column(i, schedule_i))
        return cols

    columns = initial_columns()

    # -------------------------
    # Build master problem
    # -------------------------
    master = gp.Model("TTP_CG")
    y = {}
    for i in teams:
        for c_idx in range(len(columns[i])):
            y[i, c_idx] = master.addVar(vtype=GRB.BINARY, name=f"y_{i}_{c_idx}")
    master.update()

    # 1) each team selects exactly one column
    for i in teams:
        master.addConstr(gp.quicksum(y[i, c_idx] for c_idx in range(len(columns[i]))) == 1)

    # 2) double round-robin: each ordered pair (i->j away) occurs once
    for i in teams:
        for j in teams:
            if i == j: continue
            expr = gp.LinExpr()
            for c_idx in range(len(columns[i])):
                for t in range(R):
                    if columns[i][c_idx]['opp'][t] == j and columns[i][c_idx]['ha'][t] == 'A':
                        expr += y[i, c_idx]
            master.addConstr(expr == 1)

    # 3) venue consistency: v[i,i,t] == sum_j x[j,i,t] etc
    for i in teams:
        for t in range(R):
            lhs = gp.LinExpr()
            for c_idx in range(len(columns[i])):
                lhs += columns[i][c_idx]['v'][(i, t)] * y[i, c_idx]
            rhs = gp.LinExpr()
            for j in teams:
                if j == i: continue
                for c_j in range(len(columns[j])):
                    if columns[j][c_j]['opp'][t] == i and columns[j][c_j]['ha'][t] == 'A':
                        rhs += y[j, c_j]
            master.addConstr(lhs == rhs)

    # 4) windowed streak constraints
    for i in teams:
        for t0 in range(R - (window - 1)):
            expr_h = gp.LinExpr()
            expr_a = gp.LinExpr()
            for c_idx in range(len(columns[i])):
                expr_h += sum(columns[i][c_idx]['v'][(i, t0 + w)] for w in range(window)) * y[i, c_idx]
                expr_a += sum(1 - columns[i][c_idx]['v'][(i, t0 + w)] for w in range(window)) * y[i, c_idx]
            master.addConstr(expr_h <= max_streak)
            master.addConstr(expr_h >= min_streak)
            master.addConstr(expr_a <= max_streak)
            master.addConstr(expr_a >= min_streak)

    # 5) objective
    obj = gp.LinExpr()
    for i in teams:
        for c_idx in range(len(columns[i])):
            obj += columns[i][c_idx]['cost'] * y[i, c_idx]
    master.setObjective(obj, GRB.MINIMIZE)

    # -------------------------
    # Solve
    # -------------------------
    master.Params.OutputFlag = 1
    master.Params.MIPGap = 0.0
    master.optimize()
    if master.status != GRB.OPTIMAL:
        raise RuntimeError("Integer master not solved")

    # -------------------------
    # Extract schedule
    # -------------------------
    chosen = {}
    for i in teams:
        for c_idx in range(len(columns[i])):
            if y[i, c_idx].X > 0.5:
                chosen[i] = columns[i][c_idx]
                break

    schedule = [[] for _ in range(R)]
    for t in range(R):
        for i in teams:
            col = chosen[i]
            j = col['opp'][t]
            ha = col['ha'][t]
            if ha == 'A':
                schedule[t].append((i, j))
    total_dist = master.ObjVal
    return schedule, total_dist


# -------------------------
# 5. Main
# -------------------------
def main():
    xml_path = 'NL4.xml'
    if len(sys.argv)>=2:
        xml_path = sys.argv[1]
    if not os.path.exists(xml_path):
        raise SystemExit(f"Could not find instance file at '{xml_path}'.")

    n,R,teams,dist,params = parse_robinx_xml(xml_path)
    print(f"Parsed instance: n={n}, rounds={R}, params={params}")

    if n <= 6:
        print("Solving with exact time-indexed MILP...")
        m,x,v = build_ttp_tr_model(n,R,teams,dist,params)
        m.optimize()
        if m.status != GRB.OPTIMAL:
            print(f"Solve ended with status {m.status}. Best obj={m.ObjVal if m.SolCount else 'n/a'}")
            return
        print(f"\nOptimal total travel distance: {m.ObjVal:.0f}\n")
        schedule,hadv = extract_schedule(n,R,x)
        print("Round-by-round schedule (A@H):")
        for t,games in enumerate(schedule, start=1):
            print(f"Round {t:2d}:  " + "  ".join([f"{a}@{h}" for (a,h) in games]))
        print("\nPer-team H/A patterns:")
        for i in range(n):
            print(f"Team {i}: {hadv[i]}")
    else:
        print("Solving with column generation BnB (NL8/NL10)...")
        schedule,total_dist = solve_column_generation({'n':n,'R':R,'teams':teams,'dist':dist,'params':params})
        print(f"\nOptimal total travel distance: {total_dist:.0f}\n")
        print("Round-by-round schedule (A@H):")
        for t,games in enumerate(schedule, start=1):
            print(f"Round {t:2d}:  " + "  ".join([f"{a}@{h}" for (a,h) in games]))
        print("\nPer-team H/A patterns:")
        # Compute per-team H/A patterns
        hadv = {i:'' for i in teams}
        for t,games in enumerate(schedule):
            for i in teams:
                ha = 'A' if any(i==away for (away,home) in games) else 'H'
                hadv[i] += ha
        for i in teams:
            print(f"Team {i}: {hadv[i]}")

if __name__=='__main__':
    main()
