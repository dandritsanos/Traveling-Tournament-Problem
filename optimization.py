# rmp_seed.py
import xml.etree.ElementTree as ET
from collections import defaultdict
import gurobipy as gp
import random
from gurobipy import GRB

# ---------- Parse NL (RobinX) instance ----------
def parse_robinx_xml(path):
    root = ET.parse(path).getroot()
    teams = [int(t.attrib["id"]) for t in root.findall(".//Teams/team")]
    n = len(teams)
    slots = [int(s.attrib["id"]) for s in root.findall(".//Slots/slot")]
    S = len(slots)
    dist = [[0]*n for _ in range(n)]
    for d in root.findall(".//Distances/distance"):
        i = int(d.attrib["team1"]); j = int(d.attrib["team2"])
        val = int(d.attrib["dist"])
        dist[i][j] = val; dist[j][i] = val
    # streak cap U from CA3 intp=4, max=3 (no 4-in-a-row)
    U = 3
    for c in root.findall(".//CapacityConstraints/CA3"):
        if c.attrib.get("intp") == "4":
            U = int(c.attrib.get("max", "3"))
            break
    return teams, S, dist, U

# ---------- H/A patterns ----------
def sliding_ok(hap, U):
    W = U + 1
    for i in range(0, len(hap) - W + 1):
        win = hap[i:i+W]
        if win.count('H') > U or win.count('A') > U:
            return False
    return True

def build_hap(len_slots, away_count, U):
    hap = ['H' if s % 2 == 0 else 'A' for s in range(len_slots)]
    needA = away_count - hap.count('A')
    if needA != 0:
        target = 'A' if needA > 0 else 'H'
        for s in range(len_slots):
            if needA == 0: break
            if hap[s] != target:
                old = hap[s]; hap[s] = target
                if not sliding_ok(hap, U): hap[s] = old
                else: needA += -1 if target == 'A' else +1
    assert hap.count('A') == away_count and sliding_ok(hap, U)
    return hap

def diversify_hap(hap, U):
    n = len(hap)
    for r in (1,2):
        h2 = hap[r:] + hap[:r]
        if sliding_ok(h2, U): return h2
    h2 = list(reversed(hap))
    return h2 if sliding_ok(h2, U) else hap[:]

# ---------- Away orders ----------
def nearest_neighbor_order(t, opponents, dist):
    order, unvis = [], set(opponents); curr = t
    while unvis:
        o = min(unvis, key=lambda x: dist[curr][x])
        order.append(o); unvis.remove(o); curr = o
    return order

def greedy_furthest_order(t, opponents, dist):
    order, unvis = [], set(opponents); curr = t
    while unvis:
        o = max(unvis, key=lambda x: dist[curr][x])
        order.append(o); unvis.remove(o); curr = o
    return order

# ---------- Assemble per-team column ----------
def assemble_column(t, hap, away_order, dist):
    nslots = len(hap)
    opp = [None]*nslots
    away_slots = [s for s,h in enumerate(hap) if h=='A']
    assert len(away_slots) == len(away_order)
    for s,o in zip(away_slots, away_order): opp[s] = o

    venue = [t]
    for s in range(nslots):
        venue.append(t if hap[s]=='H' else opp[s])

    cost = sum(dist[venue[s]][venue[s+1]] for s in range(nslots))

    # selector flags A^{t}_{t0,s}
    A = {(t0,s): int(hap[s]=='A' and opp[s]==t0)
         for s in range(nslots) for t0 in range(len(dist)) if t0 != t}

    return {"team": t, "HA": hap[:], "opp": opp, "venue": venue, "cost": cost, "A": A}

def build_two_seeds_per_team(xml_path, num_random=3):
    teams, S, dist, U = parse_robinx_xml(xml_path)
    n = len(teams)
    away_count = n - 1

    # base H/A patterns
    hap1 = build_hap(S, away_count, U)          # alternating start H
    hap2 = diversify_hap(hap1, U)               # shifted/reversed
    hap3 = ['A' if s % 2 == 0 else 'H' for s in range(S)]  # alternating start A
    hap_list = [hap1, hap2, hap3]

    cols = defaultdict(list)
    for t in teams:
        opps = [o for o in teams if o != t]
        # opponent orders
        orders = [
            nearest_neighbor_order(t, opps, dist),
            greedy_furthest_order(t, opps, dist)
        ]
        for _ in range(num_random):
            rand_opps = opps[:]
            random.shuffle(rand_opps)
            orders.append(rand_opps)

        # combine patterns × orders
        for hap in hap_list:
            for order in orders:
                cols[t].append(assemble_column(t, hap, order, dist))

    return teams, S, cols

# ---------- Build and solve RMP ----------
def solve_rmp_with_gurobi(xml_path):
    teams, S, cols = build_two_seeds_per_team(xml_path)

    # variable index list
    var_list = []
    for t in teams:
        for p_idx in range(len(cols[t])):
            var_list.append((t, p_idx))

    m = gp.Model("RMP_seed_only")
    m.Params.OutputFlag = 1

    # variables k >= 0
    k = { (t,p): m.addVar(lb=0.0, name=f"k_{t}_{p}")
          for (t,p) in var_list }

    # objective
    m.setObjective(gp.quicksum(cols[t][p]["cost"] * k[(t,p)]
                               for (t,p) in var_list), GRB.MINIMIZE)

    # convexity per team
    for t in teams:
        m.addConstr(gp.quicksum(k[(t,p)] for p in range(len(cols[t]))) == 1,
                    name=f"conv_{t}")

    # coupling per (team,slot): away_by_t + away_to_t = 1
    for t in teams:
        for s in range(S):
            expr = gp.LinExpr()
            # t away at someone in slot s
            for p in range(len(cols[t])):
                expr += (1 if cols[t][p]["HA"][s]=='A' else 0) * k[(t,p)]
            # someone away to t in slot s
            for u in teams:
                if u == t: continue
                for p in range(len(cols[u])):
                    expr += cols[u][p]["A"].get((t,s), 0) * k[(u,p)]
            m.addConstr(expr == 1, name=f"cpl_{t}_{s}")

    m.optimize()

    print("\n=== RMP RESULT ===")
    if m.Status == GRB.OPTIMAL:
        print("Status: OPTIMAL")
        print(f"Objective: {m.objVal:.2f}")
        for (t,p), var in k.items():
            if var.X > 1e-7:
                print(f"k[{t},{p}] = {var.X:.4f}")
    elif m.Status == GRB.INFEASIBLE:
        print("Status: INFEASIBLE (seed columns not enough to satisfy coupling).")
        m.computeIIS()
        m.write("rmp_seed_only.ilp")  # conflict file (optional)
        print("IIS written to rmp_seed_only.ilp")
    else:
        print("Status:", m.Status)

if __name__ == "__main__":
    # Change this path to your file (e.g., NL4.xml or NL10.xml)
    solve_rmp_with_gurobi("NL4.xml")
