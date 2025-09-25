#!/usr/bin/env python3
"""
BNP_NL8_PropB_BP_rcfix_kbest.py

Upgrades:
  • Correct reduced costs: subtract μ_{t,s} and μ_{opp,s} on away arcs (pricing & Prop A)
  • K-best pricing (default K=3): add up to K negative columns per team per CG iteration
  • Tight solver tolerances: OptimalityTol=1e-9, FeasibilityTol=1e-9 (LP & IP)
  • Everything else preserved: Prop B pruning, Prop A, dynamic NRC, strong branching, HA fallback,
    feasibility seeding, symmetry at middle, Berger+2opt seeds, root IP=180s

Run:
    python -u BNP_NL8_PropB_BP_rcfix_kbest.py NL8.xml
"""

import sys
import time
import heapq
import random
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple

import gurobipy as gp
from gurobipy import GRB


# =========================
# Instance parsing & utils
# =========================

def parse_instance(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    teams = []
    for team in root.findall(".//Teams/team"):
        tid = int(team.attrib["id"])
        name = team.attrib.get("name", str(tid))
        teams.append((tid, name))
    teams.sort()
    team_ids = [t for t, _ in teams]
    team_names = {t: n for t, n in teams}
    n = len(team_ids)
    nbar = n - 1
    S = len(root.findall(".//Slots/slot"))
    D = [[0] * n for _ in range(n)]
    for d in root.findall(".//Distances/distance"):
        i = int(d.attrib["team1"])
        j = int(d.attrib["team2"])
        dist = int(d.attrib["dist"])
        D[i][j] = dist
    # CA3 (L..U runs)
    U, L = 3, 1
    for ca3 in root.findall(".//CapacityConstraints/CA3"):
        if ca3.attrib.get("max"): U = int(ca3.attrib["max"])
        if ca3.attrib.get("min"): L = int(ca3.attrib["min"])
    return {"n": n, "nbar": nbar, "team_ids": team_ids, "team_names": team_names,
            "D": D, "num_slots": S, "U": U, "L": L}


def tour_travel_cost(team, opponents, HA, D):
    prev = team
    total = 0
    for opp, ha in zip(opponents, HA):
        venue = team if ha == 'H' else opp
        total += D[prev][venue]
        prev = venue
    total += D[prev][team]
    return total


# =========================
# Berger seeding & 2-opt on away slots within halves
# =========================

def generate_single_rr(team_ids):
    teams = list(team_ids)
    n = len(teams)
    assert n % 2 == 0
    rounds = []
    for r in range(n - 1):
        matches = []
        for i in range(n // 2):
            a = teams[i]
            b = teams[-(i + 1)]
            matches.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(matches)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds


def generate_double_rr(team_ids):
    first = generate_single_rr(team_ids)
    second = [[(b, a) for (a, b) in rnd] for rnd in first]
    return first + second


def build_seed_tours(inst):
    teams = inst["team_ids"]
    S = inst["num_slots"]
    rounds = generate_double_rr(teams)
    assert len(rounds) == S
    tours = {t: ([], []) for t in teams}
    for s, rnd in enumerate(rounds):
        for (h, a) in rnd:
            tours[h][0].append(a); tours[h][1].append('H')
            tours[a][0].append(h); tours[a][1].append('A')
    return tours


def improve_team_tour_by_2opt(inst, team, opps, HA, tries=200):
    """Swap away opponents within each half while preserving H/A and L..U."""
    D = inst["D"]; S = inst["num_slots"]; nbar = inst["nbar"]
    seen = set(); variants = []
    def add_variant(o, h):
        key = (tuple(o), tuple(h))
        if key not in seen:
            seen.add(key); variants.append((list(o), list(h)))
    add_variant(opps, HA)
    def cost_of(o, h): return tour_travel_cost(team, o, h, D)
    for (start, end) in [(0, nbar), (nbar, S)]:
        away_pos = [s for s in range(start, end) if HA[s] == 'A']
        if len(away_pos) <= 1: continue
        o = list(opps); h = list(HA); best = cost_of(o, h)
        for _ in range(tries):
            i, j = random.sample(away_pos, 2)
            if i > j: i, j = j, i
            o[i], o[j] = o[j], o[i]
            c = cost_of(o, h)
            if c + 1e-9 < best:
                best = c; add_variant(o, h)
            else:
                o[i], o[j] = o[j], o[i]
    scored = { (tuple(o), tuple(h)): cost_of(o, h) for (o, h) in variants }
    variants = sorted(scored.items(), key=lambda kv: kv[1])
    K = min(20, len(variants))
    return [(list(o), list(h)) for ((o, h), _) in variants[:K]]


# =========================
# Master LP with branching + NRC cuts
# =========================

class MasterLP:
    def __init__(self, inst, forced_include=None, forced_exclude=None,
                 forced_home=None, forced_away=None, gurobi_output=True):
        self.inst = inst
        self.team_ids = inst["team_ids"]
        self.S = inst["num_slots"]
        self.columns = {t: [] for t in self.team_ids}
        self.model = None
        self.lam = {}
        self.team_constr = {}
        self.couple_constr = {}
        self.branch_include = set(forced_include) if forced_include else set()   # (home,s,away)
        self.branch_exclude = set(forced_exclude) if forced_exclude else set()
        self.forced_home = set(forced_home) if forced_home else set()           # (team,s)
        self.forced_away = set(forced_away) if forced_away else set()
        self.force_home_constr = {}
        self.force_away_constr = {}
        self.nrc_cuts = []
        self._built = False
        self._gurobi_output = gurobi_output

    def add_column(self, team, col):
        idx = len(self.columns[team])
        self.columns[team].append(col)
        if not self._built:
            return
        v = self.model.addVar(lb=0.0, ub=1.0, name=f"lam_{team}_{idx}")
        self.lam[(team, idx)] = v
        self.model.chgCoeff(self.team_constr[team], v, 1.0)
        # coupling contributions
        for s in range(self.S):
            if col["HA"][s] == 'A':
                opp = col["opponents"][s]
                self.model.chgCoeff(self.couple_constr[(team, s)], v, 1.0)
                self.model.chgCoeff(self.couple_constr[(opp, s)], v, 1.0)
        # forced HA
        for s in range(self.S):
            if (team, s) in self.forced_home and col["HA"][s] == 'A':
                self.model.chgCoeff(self.force_home_constr[(team, s)], v, 1.0)
            if (team, s) in self.forced_away and col["HA"][s] == 'H':
                self.model.chgCoeff(self.force_away_constr[(team, s)], v, 1.0)
        # objective incremental
        self.model.setObjective(self.model.getObjective() + col["cost"] * v, GRB.MINIMIZE)
        self.model.update()

    def build(self):
        m = gp.Model("TTP_master")
        m.Params.OutputFlag = 1 if self._gurobi_output else 0
        # Tight tolerances for exactness
        m.Params.OptimalityTol = 1e-9
        m.Params.FeasibilityTol = 1e-9
        self.model = m
        self.lam = {}
        # λ variables
        for t in self.team_ids:
            for idx, _ in enumerate(self.columns[t]):
                self.lam[(t, idx)] = self.model.addVar(lb=0.0, ub=1.0, name=f"lam_{t}_{idx}")
        self.model.update()
        # each team selects exactly one tour
        for t in self.team_ids:
            expr = gp.quicksum(self.lam[(t, idx)] for idx in range(len(self.columns[t])))
            self.team_constr[t] = self.model.addConstr(expr == 1, name=f"oneTour_{t}")
        # coupling constraints per (team,slot)
        for t in self.team_ids:
            for s in range(self.S):
                expr = gp.LinExpr()
                for idx, col in enumerate(self.columns[t]):
                    if col["HA"][s] == 'A': expr += self.lam[(t, idx)]
                for t2 in self.team_ids:
                    if t2 == t: continue
                    for idx, col in enumerate(self.columns[t2]):
                        if col["HA"][s] == 'A' and col["opponents"][s] == t:
                            expr += self.lam[(t2, idx)]
                self.couple_constr[(t, s)] = self.model.addConstr(expr == 1, name=f"couple_{t}_{s}")
        # include / exclude
        for (home, s, away) in self.branch_include:
            expr = gp.LinExpr()
            for idx, col in enumerate(self.columns[away]):
                if col["HA"][s] == 'A' and col["opponents"][s] == home:
                    expr += self.lam[(away, idx)]
            for idx, col in enumerate(self.columns[home]):
                if col["HA"][s] == 'H' and col["opponents"][s] == away:
                    expr += self.lam[(home, idx)]
            self.model.addConstr(expr == 1, name=f"force_inc_{home}_{s}_{away}")
        for (home, s, away) in self.branch_exclude:
            expr = gp.LinExpr()
            for idx, col in enumerate(self.columns[away]):
                if col["HA"][s] == 'A' and col["opponents"][s] == home:
                    expr += self.lam[(away, idx)]
            for idx, col in enumerate(self.columns[home]):
                if col["HA"][s] == 'H' and col["opponents"][s] == away:
                    expr += self.lam[(home, idx)]
            self.model.addConstr(expr == 0, name=f"force_exc_{home}_{s}_{away}")
        # forced HA (HA branching)
        for (t, s) in self.forced_home:
            expr = gp.LinExpr()
            for idx, col in enumerate(self.columns[t]):
                if col["HA"][s] == 'A': expr += self.lam[(t, idx)]
            self.force_home_constr[(t, s)] = self.model.addConstr(expr == 0, name=f"force_home_{t}_{s}")
        for (t, s) in self.forced_away:
            expr = gp.LinExpr()
            for idx, col in enumerate(self.columns[t]):
                if col["HA"][s] == 'H': expr += self.lam[(t, idx)]
            self.force_away_constr[(t, s)] = self.model.addConstr(expr == 0, name=f"force_away_{t}_{s}")
        # objective
        obj = gp.quicksum(
            self.columns[t][idx]["cost"] * self.lam[(t, idx)]
            for t in self.team_ids
            for idx in range(len(self.columns[t]))
        )
        self.model.setObjective(obj, GRB.MINIMIZE)
        self._built = True

    def solve_lp(self, time_limit=None):
        for v in self.model.getVars(): v.vtype = GRB.CONTINUOUS
        self.model.update()
        self.model.Params.IntegralityFocus = 0
        self.model.Params.Method = 1  # dual simplex
        self.model.Params.OptimalityTol = 1e-9
        self.model.Params.FeasibilityTol = 1e-9
        if time_limit is not None:
            self.model.Params.TimeLimit = time_limit
        else:
            self.model.Params.TimeLimit = 1e100
        self.model.optimize()
        if self.model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and self.model.SolCount > 0:
            try:
                obj_val = self.model.ObjVal
            except gp.GurobiError:
                obj_val = float('inf')
            try:
                dual_team = {t: self.team_constr[t].Pi for t in self.team_ids}
                dual_couple = {(t, s): self.couple_constr[(t, s)].Pi
                               for t in self.team_ids for s in range(self.S)}
            except gp.GurobiError:
                dual_team, dual_couple = {}, {}
        else:
            obj_val = float('inf'); dual_team, dual_couple = {}, {}
        return self.model.Status, obj_val, dual_team, dual_couple

    def solve_ip(self, time_limit=None):
        """Solve the RMP as an IP to improve the incumbent (optional)."""
        for v in self.model.getVars():
            v.vtype = GRB.BINARY
        self.model.update()
        self.model.Params.OptimalityTol = 1e-9
        self.model.Params.FeasibilityTol = 1e-9
        if time_limit is not None:
            self.model.Params.TimeLimit = time_limit
        else:
            self.model.Params.TimeLimit = 1e100
        self.model.optimize()
        sol = {}
        if self.model.SolCount > 0 and self.model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            for (t, idx), var in self.lam.items():
                if var.X > 0.5:
                    sol[t] = (idx, self.columns[t][idx])
            try:
                obj = self.model.ObjVal
            except gp.GurobiError:
                obj = None
            return self.model.Status, obj, sol
        return self.model.Status, None, None


# =========================
# Reduced-cost graph + Prop B pruning helpers
# =========================

def build_rc_edges(inst, team, dual_couple):
    """
    Reduced-cost arc (s,u)->(s+1,v):
      If v==team (home): rc = D[u][v]
      If v!=team (away): rc = D[u][v] - μ_{team,s} - μ_{v,s}
    This matches how coupling columns contribute to the master.
    """
    n = inst["n"]; S = inst["num_slots"]; D = inst["D"]
    rc_edges = {}
    for s in range(S):
        for u in range(n):
            lst = []
            for v in range(n):
                if v == team:
                    rc = D[u][v]
                else:
                    rc = D[u][v] - dual_couple.get((team, s), 0.0) - dual_couple.get((v, s), 0.0)
                lst.append((v, rc))
            rc_edges[(s, u)] = lst
    return rc_edges

def rc_forward_dp(inst, team, rc_edges):
    n = inst["n"]; S = inst["num_slots"]
    INF = 1e18
    dist = {(s, u): INF for s in range(S+1) for u in range(n)}
    dist[(0, team)] = 0.0
    for s in range(S):
        for u in range(n):
            d = dist[(s, u)]
            if d >= INF/2: continue
            for (v, rc) in rc_edges.get((s, u), []):
                nd = d + rc
                if nd < dist[(s+1, v)]:
                    dist[(s+1, v)] = nd
    return dist

def rc_backward_dp(inst, team, rc_edges):
    n = inst["n"]; S = inst["num_slots"]; D = inst["D"]
    INF = 1e18
    dist = {(s, u): INF for s in range(S+1) for u in range(n)}
    for u in range(n):
        dist[(S, u)] = D[u][team]
    for s in range(S-1, -1, -1):
        for u in range(n):
            best = INF
            for (v, rc) in rc_edges.get((s, u), []):
                cand = rc + dist[(s+1, v)]
                if cand < best: best = cand
            dist[(s, u)] = best
    return dist


# =========================
# Exact pricing (SPPRC) + Prop B + symmetry + forced HA/away target (K-best)
# =========================

Label = namedtuple("Label", ["cost", "u", "rlen", "last_ha", "mask", "prev"])

def pricing_exact_propB_Kbest(inst, team, dual_team, dual_couple, existing_cols,
                              forced_inc, forced_exc, forced_home_slots, forced_away_slots,
                              incumbent_UB, K_best=3, apply_symmetry=True, eps=1e-8,
                              force_away_to=None, accept_nonimproving=False):
    """
    Return up to K_best negative reduced-cost tours for 'team'.
    """
    n = inst["n"]; S = inst["num_slots"]; D = inst["D"]; U = inst["U"]; L = inst["L"]
    nbar = inst["nbar"]
    teams = inst["team_ids"]
    opps = [x for x in teams if x != team]
    opp_index = {opps[i]: i for i in range(len(opps))}
    FULL = (1 << len(opps)) - 1

    rc_edges = build_rc_edges(inst, team, dual_couple)

    # forced HA per slot
    if forced_home_slots:
        for s in forced_home_slots:
            for u in range(n):
                rc_edges[(s, u)] = [(v, rc) for (v, rc) in rc_edges[(s, u)] if v == team]
    if forced_away_slots:
        for s in forced_away_slots:
            for u in range(n):
                rc_edges[(s, u)] = [(v, rc) for (v, rc) in rc_edges[(s, u)] if v != team]
    # forced specific away target
    if force_away_to:
        for s, h in force_away_to.items():
            for u in range(n):
                rc_edges[(s, u)] = [(v, rc) for (v, rc) in rc_edges[(s, u)] if v == h]

    # DP bounds for Prop B
    rc_fw = rc_forward_dp(inst, team, rc_edges)
    rc_bw = rc_backward_dp(inst, team, rc_edges)

    # Prop B: prune arcs if FW + arc + BW - π_t >= 0
    pruned = {}
    for (s, u), lst in rc_edges.items():
        kept = []
        for (v, rc) in lst:
            lb = rc_fw.get((s, u), 1e15) + rc + rc_bw.get((s+1, v), 1e15) - dual_team.get(team, 0.0)
            if lb < -1e-9:
                kept.append((v, rc))
        pruned[(s, u)] = kept

    # symmetry at middle for min-id team
    if apply_symmetry and team == min(teams):
        s_mid = nbar
        for u in range(n):
            lst = pruned.get((s_mid, u), [])
            pruned[(s_mid, u)] = [(v, rc) for (v, rc) in lst if u <= v]

    # labeling
    layer = [dict() for _ in range(S+1)]
    start = Label(cost=0.0, u=team, rlen=0, last_ha=None, mask=0, prev=None)
    layer[0][(team, 0, None, 0)] = start

    def relax(dic, key, lab):
        old = dic.get(key)
        if (old is None) or (lab.cost < old.cost - 1e-12):
            dic[key] = lab

    for s in range(S):
        cur = layer[s]; nxt = layer[s+1]
        for (u, rlen, last_ha, mask), lab in cur.items():
            for (v, rc) in pruned.get((s, u), []):
                ha = 'H' if v == team else 'A'
                # L..U runs
                if last_ha is None:
                    new_rlen = 1
                elif ha == last_ha:
                    new_rlen = rlen + 1
                else:
                    if rlen < L: continue
                    new_rlen = 1
                if new_rlen > U: continue
                new_mask = mask
                if ha == 'A':
                    if v not in opp_index: continue
                    bit = 1 << opp_index[v]
                    if (mask & bit) != 0:
                        continue
                    new_mask = mask | bit
                # forced excludes (only meaningful for away)
                banned = False
                for (home, ss, away) in forced_exc:
                    if ss == s and ha == 'A' and away == team and home == v:
                        banned = True; break
                if banned: continue

                relax(nxt, (v, new_rlen, ha, new_mask),
                      Label(cost=lab.cost + rc, u=v, rlen=new_rlen, last_ha=ha, mask=new_mask,
                            prev=(s, (u, rlen, last_ha, mask))))

    # collect terminal candidates (K-best)
    terms = []
    for (u, rlen, last_ha, mask), lab in layer[S].items():
        if mask != FULL: continue
        if last_ha is not None and rlen < L: continue
        rc_total = lab.cost + D[u][team]
        rdc = rc_total - dual_team.get(team, 0.0)
        terms.append((rdc, rc_total, (u, rlen, last_ha, mask), lab))

    # sort by reduced cost
    terms.sort(key=lambda x: x[0])

    cols = []
    for rdc, rc_total, key, lab in terms:
        if (not accept_nonimproving) and rdc >= -1e-9:
            break
        # reconstruct venues
        venues = []
        cur = lab
        s = S
        while cur.prev is not None:
            prev_s, prev_key = cur.prev
            venues.append(cur.u)
            s = prev_s
            cur = layer[s][prev_key]
        venues.reverse()
        HA = ['H' if v == team else 'A' for v in venues]
        opponents = [ (team if h == 'H' else v) for v, h in zip(venues, HA) ]
        true_cost = tour_travel_cost(team, opponents, HA, D)
        keycol = (tuple(opponents), tuple(HA))
        if keycol in existing_cols.get(team, set()):
            continue
        cols.append({"opponents": opponents, "HA": HA, "cost": true_cost, "rdc": rdc})
        if len(cols) >= K_best:
            break

    return cols if cols else None


# =========================
# NRC separation (no consecutive repeaters)
# =========================

def expr_y(master, home, s, away):
    expr = gp.LinExpr()
    for idx, col in enumerate(master.columns[away]):
        if col["HA"][s] == 'A' and col["opponents"][s] == home:
            expr += master.lam[(away, idx)]
    return expr

def separate_nrc_cuts(master, eps=1e-6, max_new_cuts=200):
    added = 0
    teams = master.inst["team_ids"]
    S = master.inst["num_slots"]

    def y_val(home, s, away):
        val = 0.0
        for idx, col in enumerate(master.columns[away]):
            if col["HA"][s] == 'A' and col["opponents"][s] == home:
                v = master.lam[(away, idx)].X
                if v: val += v
        return val

    for i in teams:
        for j in teams:
            if i == j: continue
            for s in range(S - 1):
                lhs1 = y_val(i, s, j) + y_val(j, s + 1, i)
                if lhs1 > 1.0 + eps:
                    cut = expr_y(master, i, s, j) + expr_y(master, j, s + 1, i) <= 1
                    master.model.addConstr(cut, name=f"NRC1_{i}_{s}_{j}")
                    added += 1
                    if added >= max_new_cuts:
                        master.model.update()
                        return added
                lhs2 = y_val(j, s, i) + y_val(i, s + 1, j)
                if lhs2 > 1.0 + eps:
                    cut = expr_y(master, j, s, i) + expr_y(master, i, s + 1, j) <= 1
                    master.model.addConstr(cut, name=f"NRC2_{j}_{s}_{i}")
                    added += 1
                    if added >= max_new_cuts:
                        master.model.update()
                        return added
    if added:
        master.model.update()
    return added


# =========================
# Proposition A (tour elimination) — RC fixed (subtract both μ_{t,s}, μ_{opp,s})
# =========================

def apply_proposition_A(master, dual_team, dual_couple, incumbent_UB, node_LB, safety_keep=1):
    threshold = max(0.0, incumbent_UB - node_LB)
    fixed = 0
    for t in master.inst["team_ids"]:
        rcs = []
        for idx, col in enumerate(master.columns[t]):
            rc = col["cost"] - dual_team.get(t, 0.0)
            for s in range(master.S):
                if col["HA"][s] == 'A':
                    opp = col["opponents"][s]
                    rc -= dual_couple.get((t, s), 0.0)
                    rc -= dual_couple.get((opp, s), 0.0)
            rcs.append((rc, idx))
        rcs_sorted = sorted(rcs, key=lambda x: x[0])
        keep = set(idx for _, idx in rcs_sorted[:safety_keep])
        for rc, idx in rcs_sorted[safety_keep:]:
            if rc >= threshold - 1e-9:
                var = master.lam[(t, idx)]
                if var.UB > 0.0:
                    var.UB = 0.0
                    fixed += 1
    if fixed:
        master.model.update()
    return fixed


# =========================
# Initialization & feasibility seeding
# =========================

def seed_initial_columns(inst, master, extra_2opt=True, max_extra_per_team=20, tries_per_team=200):
    teams = inst["team_ids"]
    tours = build_seed_tours(inst)

    # Base Berger column
    for t in teams:
        opps, HA = tours[t]
        c = tour_travel_cost(t, opps, HA, inst["D"])
        master.columns[t].append({"opponents": list(opps), "HA": list(HA), "cost": c})

    # Extra 2-opt variants
    if extra_2opt:
        for t in teams:
            opps, HA = tours[t]
            variants = improve_team_tour_by_2opt(inst, t, list(opps), list(HA), tries=tries_per_team)
            base = (tuple(opps), tuple(HA))
            added = 0
            for (o, h) in variants:
                if (tuple(o), tuple(h)) == base:
                    continue
                c = tour_travel_cost(t, o, h, inst["D"])
                master.columns[t].append({"opponents": o, "HA": h, "cost": c})
                added += 1
                if added >= max_extra_per_team:
                    break

def seed_initital_columns(*args, **kwargs):
    return seed_initial_columns(*args, **kwargs)


def ensure_branch_feasibility(inst, master,
                              forced_inc, forced_home, forced_away,
                              existing_cols):
    """Add at least one compatible column per affected team at this node."""
    zero_duals_team = {t: 0.0 for t in inst["team_ids"]}
    zero_duals_cpl  = {(t, s): 0.0 for t in inst["team_ids"] for s in range(inst["num_slots"])}

    def add_if_new(team, col):
        key = (tuple(col["opponents"]), tuple(col["HA"]))
        if key not in existing_cols[team]:
            master.columns[team].append(col)
            existing_cols[team].add(key)

    # Include events: ensure away team has a column realizing (home, s, away)
    for (home, s, away) in forced_inc:
        cols = pricing_exact_propB_Kbest(inst, away, zero_duals_team, zero_duals_cpl, existing_cols,
                                         forced_inc=set(), forced_exc=set(),
                                         forced_home_slots=set(), forced_away_slots={s},
                                         incumbent_UB=float('inf'),
                                         apply_symmetry=False,
                                         force_away_to={s: home},
                                         accept_nonimproving=True, K_best=1)
        if cols:
            col = cols[0]
            add_if_new(away, {"opponents": col["opponents"], "HA": col["HA"], "cost": col["cost"]})

    # Forced Away
    for (t, s) in forced_away:
        cols = pricing_exact_propB_Kbest(inst, t, zero_duals_team, zero_duals_cpl, existing_cols,
                                         forced_inc=set(), forced_exc=set(),
                                         forced_home_slots=set(), forced_away_slots={s},
                                         incumbent_UB=float('inf'),
                                         apply_symmetry=False,
                                         force_away_to=None,
                                         accept_nonimproving=True, K_best=1)
        if cols:
            col = cols[0]
            add_if_new(t, {"opponents": col["opponents"], "HA": col["HA"], "cost": col["cost"]})

    # Forced Home
    for (t, s) in forced_home:
        cols = pricing_exact_propB_Kbest(inst, t, zero_duals_team, zero_duals_cpl, existing_cols,
                                         forced_inc=set(), forced_exc=set(),
                                         forced_home_slots={s}, forced_away_slots=set(),
                                         incumbent_UB=float('inf'),
                                         apply_symmetry=False,
                                         force_away_to=None,
                                         accept_nonimproving=True, K_best=1)
        if cols:
            col = cols[0]
            add_if_new(t, {"opponents": col["opponents"], "HA": col["HA"], "cost": col["cost"]})


# =========================
# Column generation (NRC + Prop A + K-best pricing)
# =========================

def column_generation_with_branch(inst,
                                  forced_inc, forced_exc,
                                  forced_home=None, forced_away=None,
                                  incumbent_UB=float('inf'),
                                  time_limit_per_lp=None, max_iters=200, gurobi_output=True,
                                  K_best=3):
    teams = inst["team_ids"]
    forced_home = set(forced_home) if forced_home else set()
    forced_away = set(forced_away) if forced_away else set()

    master = MasterLP(inst,
                      forced_include=forced_inc, forced_exclude=forced_exc,
                      forced_home=forced_home, forced_away=forced_away,
                      gurobi_output=gurobi_output)

    # Seed columns (Berger + 2-opt pool)
    seed_initial_columns(inst, master, extra_2opt=True, max_extra_per_team=20, tries_per_team=200)

    # Track existing sets
    existing_cols = {t: set((tuple(col["opponents"]), tuple(col["HA"])) for col in master.columns[t])
                     for t in teams}

    # Ensure feasibility for this node
    ensure_branch_feasibility(inst, master,
                              forced_inc=set(forced_inc),
                              forced_home=set(forced_home),
                              forced_away=set(forced_away),
                              existing_cols=existing_cols)

    # Refresh existing
    existing_cols = {t: set((tuple(col["opponents"]), tuple(col["HA"])) for col in master.columns[t])
                     for t in teams}

    # Build master
    master.build()

    it = 0
    node_LB = float('inf')
    while it < max_iters:
        it += 1
        total_cols = sum(len(master.columns[t]) for t in teams)
        print(f"[CG] Iter {it}, total columns = {total_cols}", flush=True)

        # Solve LP
        status, obj, dual_team, dual_couple = master.solve_lp(time_limit=time_limit_per_lp)
        print(f"[CG]   LP status = {status}, obj = {obj}", flush=True)
        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or not dual_team or not dual_couple:
            print("[CG]   No duals or LP failed; stopping CG at this node.", flush=True)
            break
        node_LB = obj

        # Separate NRC
        n_added = separate_nrc_cuts(master, eps=1e-6, max_new_cuts=200)
        if n_added > 0:
            print(f"[CG]   Added {n_added} NRC cuts; re-optimizing LP.", flush=True)
            continue

        # Prop A
        n_fixed = apply_proposition_A(master, dual_team, dual_couple,
                                      incumbent_UB=incumbent_UB, node_LB=node_LB, safety_keep=1)
        if n_fixed > 0:
            print(f"[CG]   Prop A fixed {n_fixed} columns (UB-LB={max(0.0,incumbent_UB-node_LB):.2f}); re-optimizing LP.", flush=True)
            continue

        # K-best Pricing
        found_any = False
        for t in teams:
            forced_home_slots = {s for (tt, s) in forced_home if tt == t}
            forced_away_slots = {s for (tt, s) in forced_away if tt == t}
            cols = pricing_exact_propB_Kbest(inst, t, dual_team, dual_couple, existing_cols,
                                             forced_inc, forced_exc,
                                             forced_home_slots, forced_away_slots,
                                             incumbent_UB, K_best=K_best, eps=1e-8)
            if cols:
                for col in cols:
                    if col["rdc"] < -1e-8:
                        master.add_column(t, {"opponents": col["opponents"], "HA": col["HA"], "cost": col["cost"]})
                        existing_cols[t].add((tuple(col["opponents"]), tuple(col["HA"])))
                        found_any = True
                        print(f"[CG]   + Column for team {t}: rdc={col['rdc']:.6f}, cost={col['cost']}", flush=True)

        if not found_any:
            print("[CG]   No improving columns -> LP optimal for this node.", flush=True)
            break

    # final LP bound
    status, obj, _, _ = master.solve_lp(time_limit=time_limit_per_lp)
    print(f"[CG] Finished with LP bound = {obj}", flush=True)
    return master, status, obj


# =========================
# Branching helpers
# =========================

def rank_fractional_events(master, eps=1e-6):
    event_val = defaultdict(float)
    for (t, idx), var in master.lam.items():
        val = var.X if var.X is not None else 0.0
        col = master.columns[t][idx]
        for s in range(master.S):
            if col["HA"][s] == 'A':
                home = col["opponents"][s]
                away = t
                event_val[(home, s, away)] += val
    items = []
    for ev, v in event_val.items():
        if eps < v < 1 - eps:
            items.append((abs(0.5 - v), ev, v))
    items.sort(key=lambda x: x[0])
    return [(ev, val) for _, ev, val in items]

def pick_fractional_HA(master, eps=1e-6):
    best = None; best_gap = 1.0
    for t in master.inst["team_ids"]:
        for s in range(master.S):
            away_frac = 0.0
            for idx, col in enumerate(master.columns[t]):
                if col["HA"][s] == 'A':
                    away_frac += master.lam[(t, idx)].X or 0.0
            if eps < away_frac < 1 - eps:
                gap = abs(0.5 - away_frac)
                if gap < best_gap:
                    best_gap = gap
                    best = (t, s, away_frac)
    return best

def strong_branch_eval(inst, node_inc, node_exc, node_home, node_away, branch_obj,
                       incumbent_UB, time_limit_probe=10, K_best=2):
    def run_cg(inc, exc, fh, fa):
        _, st, obj = column_generation_with_branch(inst, inc, exc, fh, fa,
                                                   incumbent_UB=incumbent_UB,
                                                   time_limit_per_lp=None,
                                                   max_iters=80, gurobi_output=False,
                                                   K_best=K_best)
        return obj if st in (GRB.OPTIMAL, GRB.TIME_LIMIT) else float('inf')

    if isinstance(branch_obj, tuple) and len(branch_obj) == 3 and isinstance(branch_obj[1], int):
        ev = branch_obj
        inc = set(node_inc); exc = set(node_exc)
        fh = set(node_home); fa = set(node_away)

        inc.add(ev); b_inc = run_cg(inc, exc, fh, fa)
        inc = set(node_inc); exc = set(node_exc); exc.add(ev)
        b_exc = run_cg(inc, exc, fh, fa)
        return b_inc, b_exc

    _, team, s = branch_obj
    fh = set(node_home); fa = set(node_away)
    inc = set(node_inc); exc = set(node_exc)

    fh1 = set(fh); fh1.add((team, s))
    b_home = run_cg(inc, exc, fh1, fa)

    fa1 = set(fa); fa1.add((team, s))
    b_away = run_cg(inc, exc, fh, fa1)

    return b_home, b_away


# =========================
# Best-bound Branch-and-Price
# =========================

def branch_and_price_bestbound(inst, time_limit_minutes=60,
                               b1=5, MaxCand=30, gamma_c=1.7):
    start = time.time()
    limit = 60 * time_limit_minutes

    incumbent = {"obj": float('inf'), "sol": None}
    root = {"forced_inc": frozenset(), "forced_exc": frozenset(),
            "force_home": frozenset(), "force_away": frozenset(), "bound": 0.0}

    pq = []
    heapq.heappush(pq, (0.0, root))
    nodes = 0
    root_gap = None

    while pq:
        bound, node = heapq.heappop(pq)
        nodes += 1
        print(f"\n[B&B] Node {nodes}: bound={bound:.2f}, "
              f"|inc|={len(node['forced_inc'])}, |exc|={len(node['forced_exc'])}, "
              f"|home|={len(node['force_home'])}, |away|={len(node['force_away'])}", flush=True)

        master, status, lpobj = column_generation_with_branch(
            inst,
            set(node["forced_inc"]), set(node["forced_exc"]),
            set(node["force_home"]), set(node["force_away"]),
            incumbent_UB=incumbent["obj"],
            time_limit_per_lp=None, max_iters=200,
            gurobi_output=True, K_best=3
        )
        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            print("[B&B]  LP failed; prune node.", flush=True)
            continue
        if lpobj >= incumbent["obj"] - 1e-9:
            print(f"[B&B]  Pruned by bound: lpobj={lpobj:.2f} >= incumbent={incumbent['obj']:.2f}", flush=True)
            continue

        # LP integrality
        lp_all_int = True
        for v in master.model.getVars():
            if abs(v.X - round(v.X)) > 1e-9:
                lp_all_int = False; break
        if lp_all_int:
            sol = {}
            for (t, idx), var in master.lam.items():
                if var.X > 0.5:
                    sol[t] = (idx, master.columns[t][idx])
            if lpobj < incumbent["obj"] - 1e-9:
                incumbent["obj"] = lpobj; incumbent["sol"] = sol
                print(f"[B&B]  LP integral at {lpobj:.2f} -> new incumbent; node closed.", flush=True)
            else:
                print("[B&B]  LP integral; node closed.", flush=True)
            continue

        # Branching candidates from this LP
        cand_events = rank_fractional_events(master)
        fallback_HA = None
        if not cand_events:
            fallback_HA = pick_fractional_HA(master)
            if fallback_HA is None:
                print("[B&B]  No fractional events or H/A; closing node.", flush=True)
                continue
            print(f"[B&B]  Fallback HA candidate: (team={fallback_HA[0]}, slot={fallback_HA[1]}, away_frac={fallback_HA[2]:.3f})", flush=True)

        # Optional IP to improve UB (root 180s; others 60s)
        ip_time = 180 if nodes == 1 else 60
        st_ip, obj_ip, sol_ip = master.solve_ip(time_limit=ip_time)
        if obj_ip is not None and obj_ip < incumbent["obj"] - 1e-9:
            incumbent["obj"] = obj_ip; incumbent["sol"] = sol_ip
            print(f"[B&B]  New incumbent from IP: {obj_ip:.2f}", flush=True)

        if root_gap is None:
            if incumbent["obj"] < float('inf'):
                root_gap = max(0.0, incumbent["obj"] - lpobj)
            else:
                root_gap = 1.0
            print(f"[B&B]  Root gap set to {root_gap:.2f}", flush=True)

        # Branch
        if cand_events:
            node_gap = max(0.0, incumbent["obj"] - lpobj)
            ratio = min(1.0, (node_gap / (root_gap + 1e-12))) if root_gap > 0 else 1.0
            num_cands = int(b1 + MaxCand * (ratio ** gamma_c))
            num_cands = max(1, min(len(cand_events), num_cands))
            cand_events = cand_events[:num_cands]
            print(f"[B&B]  Strong branching over {len(cand_events)} event-candidates (ratio={ratio:.3f}).", flush=True)

            best_choice = None; best_worse = -1e18
            for (ev, val) in cand_events:
                b_inc, b_exc = strong_branch_eval(inst,
                                                  node["forced_inc"], node["forced_exc"],
                                                  node["force_home"], node["force_away"],
                                                  ev,
                                                  incumbent_UB=incumbent["obj"],
                                                  time_limit_probe=10, K_best=2)
                worse = min(b_inc, b_exc)
                print(f"        cand {ev} -> include={b_inc:.2f}, exclude={b_exc:.2f}", flush=True)
                if worse > best_worse:
                    best_worse = worse; best_choice = ("EV", ev, b_inc, b_exc)

            if best_choice is None:
                print("[B&B]  No candidate produced bounds; close node.", flush=True)
                continue

            _, ev, b_inc, b_exc = best_choice
            print(f"[B&B]  Branch on event {ev} (min-child bound={min(b_inc,b_exc):.2f})", flush=True)

            child_inc = {"forced_inc": frozenset(set(node["forced_inc"]) | {ev}),
                         "forced_exc": node["forced_exc"],
                         "force_home": node["force_home"], "force_away": node["force_away"],
                         "bound": b_inc}
            child_exc = {"forced_inc": node["forced_inc"],
                         "forced_exc": frozenset(set(node["forced_exc"]) | {ev}),
                         "force_home": node["force_home"], "force_away": node["force_away"],
                         "bound": b_exc}
            if b_inc < incumbent["obj"] - 1e-9:
                heapq.heappush(pq, (b_inc, child_inc))
            if b_exc < incumbent["obj"] - 1e-9:
                heapq.heappush(pq, (b_exc, child_exc))

        else:
            t, s, away_frac = fallback_HA
            b_home, b_away = strong_branch_eval(inst,
                                                node["forced_inc"], node["forced_exc"],
                                                node["force_home"], node["force_away"],
                                                ('HA', t, s),
                                                incumbent_UB=incumbent["obj"],
                                                time_limit_probe=10, K_best=2)
            print(f"[B&B]  HA strong-branch (team={t}, slot={s}) -> forceHome={b_home:.2f}, forceAway={b_away:.2f}", flush=True)

            child_home = {"forced_inc": node["forced_inc"], "forced_exc": node["forced_exc"],
                          "force_home": frozenset(set(node["force_home"]) | {(t, s)}),
                          "force_away": node["force_away"], "bound": b_home}
            child_away = {"forced_inc": node["forced_inc"], "forced_exc": node["forced_exc"],
                          "force_home": node["force_home"],
                          "force_away": frozenset(set(node["force_away"]) | {(t, s)}),
                          "bound": b_away}
            if b_home < incumbent["obj"] - 1e-9:
                heapq.heappush(pq, (b_home, child_home))
            if b_away < incumbent["obj"] - 1e-9:
                heapq.heappush(pq, (b_away, child_away))

        if time.time() - start > limit:
            print("[B&B]  Time limit reached; stopping.", flush=True)
            break

    return incumbent, nodes


# =========================
# Pretty print & main
# =========================

def print_solution(inst, incumbent):
    if incumbent["sol"] is None:
        print("No incumbent found.", flush=True)
        return
    print("\nFinal incumbent objective:", incumbent["obj"], flush=True)
    print("Schedule:", flush=True)

    chosen = {t: incumbent["sol"][t][1] for t in inst["team_ids"]}
    S = inst["num_slots"]
    home_opp = {(i, s): None for i in inst["team_ids"] for s in range(S)}
    for j in inst["team_ids"]:
        colj = chosen[j]
        for s in range(S):
            if colj["HA"][s] == 'A':
                i = colj["opponents"][s]
                if home_opp[(i, s)] is None:
                    home_opp[(i, s)] = j

    header = "Team".ljust(10) + " | " + " | ".join([f"S{s+1}" for s in range(S)])
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for i in inst["team_ids"]:
        col = chosen[i]
        line = []
        for s in range(S):
            if col["HA"][s] == 'H':
                opp = home_opp[(i, s)]
            else:
                opp = col["opponents"][s]
            tag = f"{col['HA'][s]}{inst['team_names'][opp]}"
            line.append(tag)
        print(inst["team_names"][i].ljust(10) + " | " + " | ".join(line), flush=True)
        print(f"  cost: {col['cost']}", flush=True)


def main(xml_path):
    inst = parse_instance(xml_path)
    print(f"Parsed instance: {inst['n']} teams, {inst['num_slots']} slots; U={inst['U']}, L={inst['L']}", flush=True)
    random.seed(42)
    start = time.time()
    incumbent, nodes = branch_and_price_bestbound(inst, time_limit_minutes=60,
                                                  b1=5, MaxCand=30, gamma_c=1.7)
    elapsed = time.time() - start
    print(f"\nSearch finished in {elapsed:.2f}s, nodes processed: {nodes}", flush=True)
    print_solution(inst, incumbent)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python BNP_NL8_PropB_BP_rcfix_kbest.py NL8.xml", flush=True)
        sys.exit(1)
    main(sys.argv[1])
