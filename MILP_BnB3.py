import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import itertools as it

# -------------------------
# 1. Parse RobinX XML instance
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
        try: rr = int(fmt.text.strip())
        except: rr = 2

    R = rr * (n - 1)
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            else:
                dist[i][j] = dist_map.get(i, {}).get(j, dist_map.get(j, {}).get(i, None))
                if dist[i][j] is None:
                    raise ValueError(f"Missing distance {i}-{j}")

    params = {'SE1_min': 1, 'SE1_max': R, 'max_streak': 3, 'window': 4, 'return_home_end': True}
    ca3 = root.find('./Constraints/CA3')
    if ca3 is not None:
        params['max_streak'] = int(ca3.attrib.get('max', 3))
        params['window'] = int(ca3.attrib.get('window', 4))
        params['min_streak'] = int(ca3.attrib.get('min', 1))
    se1 = root.find('./Constraints/SeparationConstraints/SE1')
    if se1 is not None:
        params['SE1_min'] = int(se1.attrib.get('min', params['SE1_min']))
        params['SE1_max'] = int(se1.attrib.get('max', params['SE1_max']))

    return n, R, list(range(n)), dist, params

# -------------------------
# 2. Build full MILP for TTP
# -------------------------
def solve_ttp(n, R, teams, dist, params):
    m = gp.Model('TTP_Full')

    I = teams
    J = teams
    T = list(range(R))

    # Binary vars: x[i,j,t] = 1 if i plays away at j in round t
    x = m.addVars(((i,j,t) for i in I for j in J if i != j for t in T), vtype=GRB.BINARY, name='x')
    # Venue vars: v[i,k,t] = 1 if team i is at venue k in round t
    v = m.addVars(((i,k,t) for i in I for k in J for t in T), vtype=GRB.BINARY, name='v')
    # Transition vars: z[i,t,k,l] = i moves from venue k to l between t and t+1
    z = m.addVars(((i,t,k,l) for i in I for t in range(R-1) for k in J for l in J), vtype=GRB.BINARY, name='z')

    # 1) Each team plays exactly once per round
    for i in I:
        for t in T:
            m.addConstr(gp.quicksum(x[j,i,t] for j in I if j != i) + gp.quicksum(x[i,j,t] for j in I if j != i) == 1)

    # 2) Double round-robin: each ordered pair once away
    for i in I:
        for j in I:
            if i==j: continue
            m.addConstr(gp.quicksum(x[i,j,t] for t in T) == 1)

    # 3) Venue consistency
    for i in I:
        for t in T:
            m.addConstr(v[i,i,t] == gp.quicksum(x[j,i,t] for j in I if j != i))
            for k in I:
                if k==i: continue
                m.addConstr(v[i,k,t] == x[i,k,t])
            m.addConstr(gp.quicksum(v[i,k,t] for k in I) == 1)

    # 4) No immediate rematches
    for i in I:
        for j in I:
            if i>=j: continue
            for t in range(R-1):
                m.addConstr(x[i,j,t] + x[j,i,t] + x[i,j,t+1] + x[j,i,t+1] <= 1)

    # 5) Max/min consecutive home/away
    window = params['window']
    max_streak = params['max_streak']
    min_streak = params.get('min_streak',1)
    for i in I:
        for t in range(R-(window-1)):
            m.addConstr(gp.quicksum(v[i,i,t+w] for w in range(window)) <= max_streak)
            if min_streak > 0:
                m.addConstr(gp.quicksum(v[i,i,t+w] for w in range(window)) >= min_streak)
            m.addConstr(gp.quicksum(1-v[i,i,t+w] for w in range(window)) <= max_streak)
            if min_streak > 0:
                m.addConstr(gp.quicksum(1-v[i,i,t+w] for w in range(window)) >= min_streak)

    # 6) Transition linking
    for i in I:
        for t in range(R-1):
            m.addConstr(gp.quicksum(z[i,t,k,l] for k in I for l in J) == 1)
            for k in I:
                m.addConstr(gp.quicksum(z[i,t,k,l] for l in J) == v[i,k,t])
            for l in I:
                m.addConstr(gp.quicksum(z[i,t,k,l] for k in I) == v[i,l,t+1])

    # Objective: total travel distance
    expr = gp.LinExpr()
    for i in I:
        for k in I:
            expr += dist[i][k] * v[i,k,0]  # start to first venue
    for i in I:
        for t in range(R-1):
            for k in I:
                for l in I:
                    expr += dist[k][l] * z[i,t,k,l]
    if params.get('return_home_end', True):
        for i in I:
            for k in I:
                expr += dist[k][i] * v[i,k,R-1]
    m.setObjective(expr, GRB.MINIMIZE)

    # -------------------------
    # Callback to track best solution at each node
    # -------------------------
    best_sol = {'obj': float('inf'), 'x_vals': None}

    def node_callback(model, where):
        if where == GRB.Callback.MIP:
            obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
            if obj_best < best_sol['obj']:
                best_sol['obj'] = obj_best
                # Optionally print per-node best
                print(f"Node best objective so far: {obj_best}")

    # Solve MILP with 5-minute limit
    m.Params.OutputFlag = 1
    m.Params.MIPGap = 0.01
    m.Params.TimeLimit = 86400  # 24 hours
    m.optimize(node_callback)

    # Retrieve the best solution found
    if m.SolCount > 0:
        x_sol = {(i,j,t): x[i,j,t].X for i in I for j in J if i!=j for t in T}
        schedule = {}
        ha_patterns = {i: [] for i in I}
        for t in T:
            schedule[t] = []
            for i in I:
                for j in J:
                    if i!=j and x_sol[i,j,t] > 0.5:
                        schedule[t].append(f"{i}@{j}")
                        ha_patterns[i].append('A')
                        ha_patterns[j].append('H')
        return schedule, ha_patterns, best_sol['obj']
    else:
        raise RuntimeError("No feasible solution found")

# -------------------------
# 3. Main
# -------------------------
if __name__=="__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py <instance.xml>")
        sys.exit(1)

    xml_path = sys.argv[1]
    print(f"\nSolving instance: {xml_path}")
    n, R, teams, dist, params = parse_robinx_xml(xml_path)
    schedule, ha_patterns, obj = solve_ttp(n, R, teams, dist, params)
    
    print(f"\nBest travel distance found: {obj}")
    print("Round-by-round schedule (A@H):")
    for t in range(R):
        print(f"Round {t+1}: {schedule[t]}")
    print("\nPer-team H/A patterns:")
    for i in teams:
        print(f"Team {i}: {''.join(ha_patterns[i])}")
