#!/usr/bin/env python3
"""
BNP_NL_IntRC_BP_beta_PB_sym.py

Branch-and-Price for double round-robin TTP using:
  • Integer-scaled reduced-cost arithmetic (no epsilons)
  • Exact paper arc RC with β duals from NRC cuts:
      c̃^t_ijs = d_ij - [π_ts + π_is if i in T_-t else 0]
                 - [β_ijs if i<j and s!=2n̄ ; β_jis if i>j and s!=2n̄ ; 0 otherwise]
  • Pricing returns single best negative column
  • Proposition B arc elimination via integer fw/bw DPs on these RCs
  • Label lower bounds with bw DP and simple key dominance
  • Middle-slot symmetry breaking at the boundary between halves (slot n̄-1):
        if team t plays away there, enforce opponent < t
  • Proposition A (tour elimination) with integer RC and β
  • Dynamic NRC separation (store handles to read β duals)
  • Dual stabilization (π, μ, β)
  • Event strong branching (plus fallback H/A), IP pushes for bounds
  • Seeding: Berger + per-team 2-opt variants (within halves)

Run:
    python -u BNP_NL_IntRC_BP_beta_PB_sym.py NL8.xml
"""

import sys, time, heapq, random
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple

import gurobipy as gp
from gurobipy import GRB

# =========================
# Integer scaling for duals
# =========================
SCALE = 10**6
def to_int_dual(x: float) -> int:
    return int(round(x * SCALE))


# =========================
# Instance parsing & utils
# =========================
def parse_instance(xml_path):
    tree = ET.parse(xml_path); root = tree.getroot()
    teams=[]
    for t in root.findall(".//Teams/team"):
        tid=int(t.attrib["id"]); name=t.attrib.get("name",str(tid))
        teams.append((tid,name))
    teams.sort()
    team_ids=[t for t,_ in teams]; names={t:n for t,n in teams}
    n=len(team_ids); nbar=n-1
    S = len(root.findall(".//Slots/slot"))  # should be 2*nbar
    D=[[0]*n for _ in range(n)]
    for d in root.findall(".//Distances/distance"):
        i=int(d.attrib["team1"]); j=int(d.attrib["team2"]); dist=int(d.attrib["dist"])
        D[i][j]=dist
    U,L=3,1
    for ca3 in root.findall(".//CapacityConstraints/CA3"):
        if ca3.attrib.get("max"): U=int(ca3.attrib["max"])
        if ca3.attrib.get("min"): L=int(ca3.attrib["min"])
    return {"n":n,"nbar":nbar,"num_slots":S,"team_ids":team_ids,"team_names":names,"D":D,"U":U,"L":L}

def tour_travel_cost(team, opponents, HA, D):
    prev=team; tot=0
    for opp,ha in zip(opponents,HA):
        venue = team if ha=='H' else opp
        tot += D[prev][venue]; prev=venue
    tot += D[prev][team]
    return tot


# =========================
# Berger + 2-opt (within halves)
# =========================
def generate_single_rr(team_ids):
    teams=list(team_ids); n=len(teams)
    rounds=[]
    for r in range(n-1):
        matches=[]
        for i in range(n//2):
            a=teams[i]; b=teams[-(i+1)]
            matches.append((a,b) if (r+i)%2==0 else (b,a))
        rounds.append(matches)
        teams=[teams[0]]+[teams[-1]]+teams[1:-1]
    return rounds

def generate_double_rr(team_ids):
    first=generate_single_rr(team_ids)
    second=[[(b,a) for (a,b) in rnd] for rnd in first]
    return first+second

def build_seed_tours(inst):
    teams=inst["team_ids"]; S=inst["num_slots"]
    rounds=generate_double_rr(teams)
    tours={t:([],[]) for t in teams}
    for s,rnd in enumerate(rounds):
        for (h,a) in rnd:
            tours[h][0].append(a); tours[h][1].append('H')
            tours[a][0].append(h); tours[a][1].append('A')
    return tours

def improve_team_tour_by_2opt(inst, team, opps, HA, tries=200):
    """Swap away opponents within each half; keep H/A and L..U."""
    D=inst["D"]; S=inst["num_slots"]; nbar=inst["nbar"]
    seen=set(); variants=[]
    def add(o,h):
        key=(tuple(o),tuple(h))
        if key not in seen: seen.add(key); variants.append((list(o),list(h)))
    add(opps,HA)
    def cost(o,h): return tour_travel_cost(team,o,h,D)
    for (st,en) in [(0,nbar),(nbar,S)]:
        away=[s for s in range(st,en) if HA[s]=='A']
        if len(away)<=1: continue
        o=list(opps); h=list(HA)
        base=cost(o,h)
        for _ in range(tries):
            i,j=random.sample(away,2)
            if i>j: i,j=j,i
            o[i],o[j]=o[j],o[i]
            c=cost(o,h)
            if c<base:
                add(o,h); base=c
            else:
                o[i],o[j]=o[j],o[i]
    variants=sorted({(tuple(o),tuple(h)):tour_travel_cost(team,o,h,D) for (o,h) in variants}.items(),
                    key=lambda kv: kv[1])
    return [(list(o),list(h)) for (o,h),_ in variants[:20]]


# =========================
# Master LP with NRC storage
# =========================
class MasterLP:
    def __init__(self, inst, forced_include=None, forced_exclude=None,
                 forced_home=None, forced_away=None, gurobi_output=True):
        self.inst=inst; self.S=inst["num_slots"]; self.team_ids=inst["team_ids"]
        self.columns={t:[] for t in self.team_ids}
        self.model=None; self.lam={}
        self.team_constr={}; self.couple_constr={}
        # branching
        self.branch_include=set(forced_include) if forced_include else set()
        self.branch_exclude=set(forced_exclude) if forced_exclude else set()
        self.forced_home=set(forced_home) if forced_home else set()
        self.forced_away=set(forced_away) if forced_away else set()
        self.force_home_constr={}; self.force_away_constr={}
        # NRC dual storage
        self.nrc1={}  # (i,s,j): y(i,s,j)+y(j,s+1,i) <= 1 -> β_ijs
        self.nrc2={}  # (j,s,i): y(j,s,i)+y(i,s+1,j) <= 1 -> β_jis
        self._built=False; self._gurobi_output=gurobi_output

    def add_column(self, team, col):
        idx=len(self.columns[team]); self.columns[team].append(col)
        if not self._built: return
        v=self.model.addVar(lb=0.0,ub=1.0,name=f"lam_{team}_{idx}")
        self.lam[(team,idx)]=v
        self.model.chgCoeff(self.team_constr[team], v, 1.0)
        for s in range(self.S):
            if col["HA"][s]=='A':
                opp=col["opponents"][s]
                self.model.chgCoeff(self.couple_constr[(team,s)], v, 1.0)
                self.model.chgCoeff(self.couple_constr[(opp,s)], v, 1.0)
        for s in range(self.S):
            if (team,s) in self.forced_home and col["HA"][s]=='A':
                self.model.chgCoeff(self.force_home_constr[(team,s)], v, 1.0)
            if (team,s) in self.forced_away and col["HA"][s]=='H':
                self.model.chgCoeff(self.force_away_constr[(team,s)], v, 1.0)
        self.model.setObjective(self.model.getObjective()+col["cost"]*v, GRB.MINIMIZE)
        self.model.update()

    def build(self):
        m=gp.Model("TTP_master"); m.Params.OutputFlag=1 if self._gurobi_output else 0
        m.Params.OptimalityTol=1e-9; m.Params.FeasibilityTol=1e-9
        self.model=m; self.lam={}
        # λ-vars
        for t in self.team_ids:
            for idx,_ in enumerate(self.columns[t]):
                self.lam[(t,idx)]=self.model.addVar(lb=0.0,ub=1.0,name=f"lam_{t}_{idx}")
        self.model.update()
        # one tour
        for t in self.team_ids:
            self.team_constr[t]=self.model.addConstr(
                gp.quicksum(self.lam[(t,idx)] for idx in range(len(self.columns[t])))==1,
                name=f"oneTour_{t}")
        # couple constraints per (team,slot)
        for t in self.team_ids:
            for s in range(self.S):
                expr=gp.LinExpr()
                for idx,col in enumerate(self.columns[t]):
                    if col["HA"][s]=='A': expr += self.lam[(t,idx)]
                for t2 in self.team_ids:
                    if t2==t: continue
                    for idx,col in enumerate(self.columns[t2]):
                        if col["HA"][s]=='A' and col["opponents"][s]==t:
                            expr += self.lam[(t2,idx)]
                self.couple_constr[(t,s)] = self.model.addConstr(expr==1, name=f"couple_{t}_{s}")
        # branch include/exclude
        for (home,s,away) in self.branch_include:
            expr=gp.LinExpr()
            for idx,col in enumerate(self.columns[away]):
                if col["HA"][s]=='A' and col["opponents"][s]==home: expr += self.lam[(away,idx)]
            for idx,col in enumerate(self.columns[home]):
                if col["HA"][s]=='H' and col["opponents"][s]==away: expr += self.lam[(home,idx)]
            self.model.addConstr(expr==1, name=f"force_inc_{home}_{s}_{away}")
        for (home,s,away) in self.branch_exclude:
            expr=gp.LinExpr()
            for idx,col in enumerate(self.columns[away]):
                if col["HA"][s]=='A' and col["opponents"][s]==home: expr += self.lam[(away,idx)]
            for idx,col in enumerate(self.columns[home]):
                if col["HA"][s]=='H' and col["opponents"][s]==away: expr += self.lam[(home,idx)]
            self.model.addConstr(expr==0, name=f"force_exc_{home}_{s}_{away}")
        # HA branching
        for (t,s) in self.forced_home:
            expr=gp.LinExpr()
            for idx,col in enumerate(self.columns[t]):
                if col["HA"][s]=='A': expr += self.lam[(t,idx)]
            self.force_home_constr[(t,s)] = self.model.addConstr(expr==0, name=f"force_home_{t}_{s}")
        for (t,s) in self.forced_away:
            expr=gp.LinExpr()
            for idx,col in enumerate(self.columns[t]):
                if col["HA"][s]=='H': expr += self.lam[(t,idx)]
            self.force_away_constr[(t,s)] = self.model.addConstr(expr==0, name=f"force_away_{t}_{s}")
        # objective
        obj=gp.quicksum(self.columns[t][idx]["cost"]*self.lam[(t,idx)]
                        for t in self.team_ids for idx in range(len(self.columns[t])))
        self.model.setObjective(obj, GRB.MINIMIZE)
        self._built=True

    def solve_lp(self, time_limit=None):
        for v in self.model.getVars(): v.vtype=GRB.CONTINUOUS
        self.model.update()
        self.model.Params.IntegralityFocus=0
        self.model.Params.Method=1
        self.model.Params.OptimalityTol=1e-9
        self.model.Params.FeasibilityTol=1e-9
        self.model.Params.TimeLimit=(time_limit if time_limit is not None else 1e100)
        self.model.optimize()
        if self.model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and self.model.SolCount>0:
            try: obj=self.model.ObjVal
            except gp.GurobiError: obj=float('inf')
            try:
                dual_team={t:self.team_constr[t].Pi for t in self.team_ids}
                dual_couple={(t,s):self.couple_constr[(t,s)].Pi for t in self.team_ids for s in range(self.S)}
                beta1={k:self.nrc1[k].Pi for k in self.nrc1}  # β_ijs
                beta2={k:self.nrc2[k].Pi for k in self.nrc2}  # β_jis
            except gp.GurobiError:
                dual_team,dual_couple,beta1,beta2 = {},{},{},{}
        else:
            obj=float('inf'); dual_team,dual_couple,beta1,beta2 = {},{},{},{}
        return self.model.Status, obj, dual_team, dual_couple, beta1, beta2

    def solve_ip(self, time_limit=None):
        for v in self.model.getVars(): v.vtype=GRB.BINARY
        self.model.update()
        self.model.Params.OptimalityTol=1e-9
        self.model.Params.FeasibilityTol=1e-9
        self.model.Params.TimeLimit=(time_limit if time_limit is not None else 1e100)
        self.model.optimize()
        sol={}
        if self.model.SolCount>0 and self.model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            for (t,idx),var in self.lam.items():
                if var.X>0.5: sol[t]=(idx,self.columns[t][idx])
            try: obj=self.model.ObjVal
            except gp.GurobiError: obj=None
            return self.model.Status, obj, sol
        return self.model.Status, None, None


# =========================
# NRC separation (store handles)
# =========================
def expr_y(master, home, s, away):
    e=gp.LinExpr()
    for idx,col in enumerate(master.columns[away]):
        if col["HA"][s]=='A' and col["opponents"][s]==home:
            e += master.lam[(away,idx)]
    return e

def separate_nrc_cuts(master, eps=1e-6, max_new_cuts=200):
    added=0; teams=master.inst["team_ids"]; S=master.inst["num_slots"]
    def yv(i,s,j):
        tot=0.0
        for idx,col in enumerate(master.columns[j]):
            if col["HA"][s]=='A' and col["opponents"][s]==i:
                val=master.lam[(j,idx)].X
                if val: tot+=val
        return tot
    for i in teams:
        for j in teams:
            if i==j: continue
            for s in range(S-1):
                lhs1 = yv(i,s,j)+yv(j,s+1,i)
                if lhs1>1.0+eps:
                    c = master.model.addConstr(expr_y(master,i,s,j)+expr_y(master,j,s+1,i)<=1,
                                               name=f"NRC1_{i}_{s}_{j}")
                    master.nrc1[(i,s,j)] = c  # β_ijs
                    added+=1
                    if added>=max_new_cuts: master.model.update(); return added
                lhs2 = yv(j,s,i)+yv(i,s+1,j)
                if lhs2>1.0+eps:
                    c = master.model.addConstr(expr_y(master,j,s,i)+expr_y(master,i,s+1,j)<=1,
                                               name=f"NRC2_{j}_{s}_{i}")
                    master.nrc2[(j,s,i)] = c  # β_jis
                    added+=1
                    if added>=max_new_cuts: master.model.update(); return added
    if added: master.model.update()
    return added


# =========================
# Proposition B helpers (build arc RC with β; fw/bw DPs)
# =========================
def build_rc_edges_with_beta_int(inst, team, mu_int, beta1_int, beta2_int,
                                 forced_home_slots, forced_away_slots,
                                 forced_to_map, symmetry_middle=True):
    """
    Build integer reduced-cost arcs for pricing with β, applying HA slot forces and middle symmetry.
    Arc: (s,u)->v with rc:
      rc = D[u][v]*SCALE - (mu_t,s + mu_v,s) [if v!=team]
           - β_ijs (if v<team and s!=S-1) - β_jis (if v>team and s!=S-1)
    """
    n=inst["n"]; S=inst["num_slots"]; D=inst["D"]; nbar=inst["nbar"]
    rc_edges={}
    for s in range(S):
        middle = (s == nbar-1)  # middle arc is between s and s+1
        for u in range(n):
            lst=[]
            for v in range(n):
                # Forced HA slots
                if s in forced_home_slots and v!=team: continue
                if s in forced_away_slots and v==team: continue
                # Forced include for this team at slot s -> must go to that home if away
                if s in forced_to_map and v != forced_to_map[s]: continue

                # Middle symmetry: require opponent < team if away at the middle arc
                if symmetry_middle and middle and v!=team:
                    if not (v < team): continue

                rc = D[u][v]*SCALE
                if v != team:  # away arc
                    rc -= mu_int[(team,s)] + mu_int[(v,s)]
                    if s != S-1:
                        if v < team:
                            rc -= beta1_int.get((v,s,team), 0)  # β_ijs
                        elif v > team:
                            rc -= beta2_int.get((team,s,v), 0)  # β_jis
                lst.append((v, rc))
            rc_edges[(s,u)] = lst
    return rc_edges

def rc_forward_dp_int(inst, rc_edges, team):
    n=inst["n"]; S=inst["num_slots"]
    INF=10**30
    fw={(s,u):INF for s in range(S+1) for u in range(n)}
    fw[(0,team)]=0
    for s in range(S):
        for u in range(n):
            d=fw[(s,u)]
            if d>=INF//2: continue
            for (v,rc) in rc_edges[(s,u)]:
                nd=d+rc
                if nd<fw[(s+1,v)]: fw[(s+1,v)]=nd
    return fw

def rc_backward_dp_int(inst, rc_edges, team):
    n=inst["n"]; S=inst["num_slots"]; D=inst["D"]
    INF=10**30
    bw={(s,u):INF for s in range(S+1) for u in range(n)}
    for u in range(n):
        bw[(S,u)] = D[u][team]*SCALE
    for s in range(S-1,-1,-1):
        for u in range(n):
            best=INF
            for (v,rc) in rc_edges[(s,u)]:
                cand=rc + bw[(s+1,v)]
                if cand<best: best=cand
            bw[(s,u)] = best
    return bw


# =========================
# Pricing: single-best with β + Prop B + label bounds
# =========================
PLabel = namedtuple("PLabel","cost u rlen last_ha mask prev")

def pricing_single_best_int(inst, team,
                            dual_team, dual_couple, beta1, beta2,
                            forced_inc, forced_exc,
                            forced_home_slots, forced_away_slots,
                            existing_cols):
    n=inst["n"]; S=inst["num_slots"]; D=inst["D"]; U=inst["U"]; L=inst["L"]; nbar=inst["nbar"]
    teams=inst["team_ids"]; opps=[x for x in teams if x!=team]
    opp_index={opps[i]:i for i in range(len(opps))}; FULL=(1<<len(opps))-1

    # Scale duals
    pi_t = to_int_dual(dual_team.get(team,0.0))
    mu = {(t,s): to_int_dual(dual_couple.get((t,s),0.0)) for t in teams for s in range(S)}
    b1 = {k: to_int_dual(beta1[k]) for k in beta1}   # β_ijs
    b2 = {k: to_int_dual(beta2[k]) for k in beta2}   # β_jis

    # Forced-to map: (home,s,away=team) -> v=home at slot s
    forced_to={}
    for (home,s,away) in forced_inc:
        if away==team: forced_to[s]=home

    # Build arc graph with β + forces + middle symmetry
    rc_edges = build_rc_edges_with_beta_int(inst, team, mu, b1, b2,
                                            forced_home_slots, forced_away_slots,
                                            forced_to_map=forced_to, symmetry_middle=True)
    # Proposition B: fw/bw DPs (integer) and arc pruning
    fw = rc_forward_dp_int(inst, rc_edges, team)
    bw = rc_backward_dp_int(inst, rc_edges, team)

    pruned={}
    for (s,u), lst in rc_edges.items():
        keep=[]
        fw_su=fw[(s,u)]
        for (v,rc) in lst:
            if fw_su + rc + bw[(s+1,v)] - pi_t < 0:
                keep.append((v,rc))
        pruned[(s,u)] = keep

    # Labeling with integer costs + bw lower bound
    layer=[dict() for _ in range(S+1)]
    start=PLabel(cost=0, u=team, rlen=0, last_ha=None, mask=0, prev=None)
    layer[0][(team,0,None,0)] = start

    def relax(dic,key,lab):
        old=dic.get(key)
        if (old is None) or (lab.cost < old.cost):
            dic[key]=lab

    for s in range(S):
        cur, nxt = layer[s], layer[s+1]
        for (u,rlen,last_ha,mask), lab in cur.items():
            # state lower bound (bw from current u)
            if lab.cost + bw[(s,u)] - pi_t >= 0:
                continue
            for (v,rc) in pruned[(s,u)]:
                ha = 'H' if v==team else 'A'
                # forced excludes
                if ha=='A':
                    bad=False
                    for (home,ss,away) in forced_exc:
                        if ss==s and away==team and home==v: bad=True; break
                    if bad: continue
                # L..U
                if last_ha is None:
                    nr=1
                elif ha==last_ha:
                    nr=rlen+1
                else:
                    if rlen<L: continue
                    nr=1
                if nr>U: continue
                # away-opp mask
                new_mask=mask
                if ha=='A':
                    if v not in opp_index: continue
                    bit=1<<opp_index[v]
                    if (mask & bit)!=0: continue
                    new_mask = mask | bit
                # next-state lower bound
                if lab.cost + rc + bw[(s+1,v)] - pi_t >= 0:
                    continue
                relax(nxt, (v,nr,ha,new_mask),
                      PLabel(cost=lab.cost+rc, u=v, rlen=nr, last_ha=ha, mask=new_mask,
                             prev=(s,(u,rlen,last_ha,mask))))

    # Pick best terminal (add return, subtract π_t)
    best_rc=None; best_key=None; best_lab=None
    for (u,rlen,last_ha,mask), lab in layer[S].items():
        if mask!=FULL: continue
        if last_ha is not None and rlen< L: continue
        total = lab.cost + D[u][team]*SCALE
        rdc = total - pi_t
        if (best_rc is None) or (rdc < best_rc):
            best_rc=rdc; best_key=(u,rlen,last_ha,mask); best_lab=lab

    if best_rc is None or best_rc >= 0:
        return None

    # Reconstruct venues
    venues=[]; cur=best_lab; s=S
    while cur.prev is not None:
        ps, pk = cur.prev
        venues.append(cur.u)
        s=ps; cur=layer[s][pk]
    venues.reverse()
    HA=['H' if v==team else 'A' for v in venues]
    opponents=[(team if h=='H' else v) for v,h in zip(venues,HA)]

    # True cost & exact column RC with β (for consistency/report)
    true_cost = tour_travel_cost(team, opponents, HA, D)
    rc_col = true_cost*SCALE - pi_t
    for s in range(S):
        if HA[s]=='A':
            opp = opponents[s]
            rc_col -= mu[(team,s)] + mu[(opp,s)]
            if s != S-1:
                if opp < team: rc_col -= b1.get((opp,s,team),0)
                elif opp > team: rc_col -= b2.get((team,s,opp),0)

    keycol=(tuple(opponents),tuple(HA))
    if keycol in existing_cols.get(team,set()):
        return None
    return {"opponents":opponents,"HA":HA,"cost":true_cost,"rdc_int":rc_col}


# =========================
# Proposition A (integer RC with β)
# =========================
def apply_proposition_A(master, dual_team, dual_couple, beta1, beta2,
                        incumbent_UB, node_LB, safety_keep=2):
    # Gate by a strict positive gap and availability of a finite UB
    if incumbent_UB == float('inf'):
        return 0
    gap = incumbent_UB - node_LB
    if gap is None or gap < 1.0:
        return 0  # don't prune unless UB–LB >= 1 distance unit

    S = master.S
    teams = master.inst["team_ids"]
    thr = max(0, int(round(gap * SCALE)))
    pi  = {t: to_int_dual(dual_team.get(t, 0.0)) for t in teams}
    mu  = {(t, s): to_int_dual(dual_couple.get((t, s), 0.0))
           for t in teams for s in range(S)}
    b1  = {k: to_int_dual(beta1[k]) for k in beta1}
    b2  = {k: to_int_dual(beta2[k]) for k in beta2}
    fixed = 0
    for t in teams:
        rcs=[]
        for idx,col in enumerate(master.columns[t]):
            rc=col["cost"]*SCALE - pi[t]
            for s in range(S):
                if col["HA"][s]=='A':
                    opp=col["opponents"][s]
                    rc -= mu[(t,s)] + mu[(opp,s)]
                    if s != S-1:
                        if opp < t: rc -= b1.get((opp,s,t),0)
                        elif opp > t: rc -= b2.get((t,s,opp),0)
            rcs.append((rc,idx))
        rcs.sort(key=lambda x:x[0])
        keep=set(idx for _,idx in rcs[:safety_keep])
        for rc,idx in rcs[safety_keep:]:
            if rc >= thr:
                var=master.lam[(t,idx)]
                if var.UB>0.0:
                    var.UB=0.0; fixed+=1
    if fixed: master.model.update()
    return fixed


# =========================
# Initialization (seed & feasibility)
# =========================
def seed_initial_columns(inst, master, extra_2opt=True, max_extra_per_team=20, tries_per_team=200):
    tours=build_seed_tours(inst)
    for t in inst["team_ids"]:
        o,h=tours[t]
        c=tour_travel_cost(t,o,h,inst["D"])
        master.columns[t].append({"opponents":list(o),"HA":list(h),"cost":c})
    if extra_2opt:
        for t in inst["team_ids"]:
            o,h=tours[t]
            base=(tuple(o),tuple(h))
            added=0
            for (oo,hh) in improve_team_tour_by_2opt(inst,t,list(o),list(h),tries=tries_per_team):
                if (tuple(oo),tuple(hh))==base: continue
                c=tour_travel_cost(t,oo,hh,inst["D"])
                master.columns[t].append({"opponents":oo,"HA":hh,"cost":c})
                added+=1
                if added>=max_extra_per_team: break

def ensure_branch_feasibility(inst, master, forced_inc, forced_home, forced_away, existing_cols):
    teams=inst["team_ids"]; S=inst["num_slots"]; D=inst["D"]
    def add(t,col):
        key=(tuple(col["opponents"]),tuple(col["HA"]))
        if key not in existing_cols[t]:
            master.columns[t].append(col); existing_cols[t].add(key)
    for (home,s,away) in forced_inc:
        for idx,col in enumerate(master.columns[away]):
            if col["HA"][s]=='A' and col["opponents"][s]==home:
                add(away,col); break
        else:
            HA=['H']*S; opp=[away]*S
            HA[s]='A'; opp[s]=home
            others=[t for t in teams if t not in (away,home)]
            k=0
            for ss in range(S):
                if ss==s: continue
                if k<len(others):
                    HA[ss]='A'; opp[ss]=others[k]; k+=1
            c=tour_travel_cost(away,opp,HA,D)
            add(away,{"opponents":opp,"HA":HA,"cost":c})
    for (t,s) in forced_away:
        for idx,col in enumerate(master.columns[t]):
            if col["HA"][s]=='A': add(t,col); break
    for (t,s) in forced_home:
        for idx,col in enumerate(master.columns[t]):
            if col["HA"][s]=='H': add(t,col); break


# =========================
# Column Generation driver
# =========================
def column_generation_with_branch(inst,
                                  forced_inc, forced_exc,
                                  forced_home=None, forced_away=None,
                                  incumbent_UB=float('inf'),
                                  time_limit_per_lp=None, max_iters=220, gurobi_output=True):
    """
    Column Generation at a B&B node with:
      - NO stall detection (always continues CG until no improving columns)
      - Pruning via incumbent_UB (if LB >= UB, node is fathomed)
      - NRC separation, dual stabilization, Prop A fixing, and single-best pricing per team.
    """
    teams = inst["team_ids"]
    forced_home = set(forced_home) if forced_home else set()
    forced_away = set(forced_away) if forced_away else set()

    master = MasterLP(inst, forced_inc, forced_exc, forced_home, forced_away, gurobi_output)
    seed_initial_columns(inst, master, extra_2opt=True, max_extra_per_team=20, tries_per_team=200)

    existing_cols = {t: set((tuple(c["opponents"]), tuple(c["HA"])) for c in master.columns[t]) for t in teams}
    ensure_branch_feasibility(inst, master, set(forced_inc), set(forced_home), set(forced_away), existing_cols)
    existing_cols = {t: set((tuple(c["opponents"]), tuple(c["HA"])) for c in master.columns[t]) for t in teams}

    master.build()

    it = 0
    node_LB = float('inf')
    prev_pi = None
    prev_mu = None
    prev_b1 = None
    prev_b2 = None
    alpha = 0.7            # dual stabilization weight (adaptive)
    improve_thresh = 1e-3  # relative improvement threshold for adapting alpha
    last_obj = None

    while it < max_iters:
        it += 1
        priced_since_last = locals().get('priced_since_last', False)
        print(f"[CG] Iter {it}, total columns = {sum(len(master.columns[t]) for t in teams)}", flush=True)

        status, obj, dual_team, dual_couple, beta1, beta2 = master.solve_lp(time_limit=time_limit_per_lp)
        print(f"[CG]   LP status = {status}, obj = {obj}", flush=True)

        # If LP failed or no duals, stop CG at this node
        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or not dual_team or not dual_couple:
            print("[CG]   No duals or LP failed; stopping CG.", flush=True)
            break

        node_LB = obj

        # --- PRUNING with incumbent UB (fathom the node early) ---
        if incumbent_UB < float('inf') and node_LB >= incumbent_UB - 1e-9:
            print(f"[CG]   Node pruned by bound: LB={node_LB:.2f} >= UB={incumbent_UB:.2f}", flush=True)
            break

        # --- Dual stabilization (simple exponential smoothing) ---
        if prev_pi is not None:
            sm_pi = {t: (1 - alpha) * dual_team[t] + alpha * prev_pi.get(t, 0.0) for t in teams}
            sm_mu = {(t, s): (1 - alpha) * dual_couple[(t, s)] + alpha * prev_mu.get((t, s), 0.0)
                     for t in teams for s in range(master.S)}
            sm_b1 = {k: (1 - alpha) * beta1[k] + alpha * (prev_b1.get(k, 0.0) if prev_b1 else 0.0) for k in beta1}
            sm_b2 = {k: (1 - alpha) * beta2[k] + alpha * (prev_b2.get(k, 0.0) if prev_b2 else 0.0) for k in beta2}
        else:
            sm_pi, sm_mu, sm_b1, sm_b2 = dual_team, dual_couple, beta1, beta2

        # --- Separate NRC cuts (2-term or 4-term depending on your separator) ---
        n_added = separate_nrc_cuts(master, eps=1e-6, max_new_cuts=200)
        if n_added > 0:
            print(f"[CG]   Added {n_added} NRC cuts; re-optimizing.", flush=True)
            prev_pi, prev_mu, prev_b1, prev_b2 = dual_team, dual_couple, beta1, beta2
            last_obj = obj
            continue

        # --- Proposition A fixing (only after we actually priced something last round) ---
        if priced_since_last:
            n_fixed = apply_proposition_A(master, sm_pi, sm_mu, sm_b1, sm_b2,
                                          incumbent_UB, node_LB, safety_keep=2)
            if n_fixed > 0:
                print(f"[CG]   Prop A fixed {n_fixed} columns (UB-LB={max(0.0, incumbent_UB - node_LB):.2f}); re-optimizing.", flush=True)
                prev_pi, prev_mu, prev_b1, prev_b2 = dual_team, dual_couple, beta1, beta2
                last_obj = obj
                priced_since_last = False
                continue
            else:
                priced_since_last = False

        # --- Pricing: single best per team ---
        found = False
        for t in teams:
            fhome = {s for (tt, s) in forced_home if tt == t}
            faway = {s for (tt, s) in forced_away if tt == t}
            col = pricing_single_best_int(inst, t, sm_pi, sm_mu, sm_b1, sm_b2,
                                          forced_inc, forced_exc, fhome, faway, existing_cols)
            if col and col["rdc_int"] < 0:
                master.add_column(t, col)
                existing_cols[t].add((tuple(col["opponents"]), tuple(col["HA"])))
                found = True
                print(f"[CG]   + Column for team {t}: rdc_int={col['rdc_int']}, cost={col['cost']}", flush=True)
                priced_since_last = True

        # --- Adapt stabilization weight (optional heuristic) ---
        if last_obj is not None and last_obj > 0:
            rel_imp = max(0.0, (last_obj - obj) / last_obj)
            alpha = max(0.3, alpha * 0.8) if rel_imp > improve_thresh else min(0.95, alpha * 1.1)
        last_obj = obj
        prev_pi, prev_mu, prev_b1, prev_b2 = dual_team, dual_couple, beta1, beta2

        # If no improving columns were found, LP is optimal for the full master -> stop CG
        if not found:
            print("[CG]   No improving columns -> LP optimal.", flush=True)
            break

    # Final LP solve (report bound)
    status, obj, _, _, _, _ = master.solve_lp(time_limit=time_limit_per_lp)
    print(f"[CG] Finished with LP bound = {obj}", flush=True)
    return master, status, obj



# =========================
# Branching helpers & B&P
# =========================
def rank_fractional_events(master, eps=1e-6):
    event_val=defaultdict(float)
    for (t,idx),var in master.lam.items():
        v=var.X or 0.0; col=master.columns[t][idx]
        for s in range(master.S):
            if col["HA"][s]=='A':
                home=col["opponents"][s]; away=t
                event_val[(home,s,away)] += v
    items=[]
    for ev,val in event_val.items():
        if eps < val < 1-eps: items.append((abs(0.5-val), ev, val))
    items.sort(key=lambda x:x[0])
    return [(ev,val) for _,ev,val in items]

def pick_fractional_HA(master, eps=1e-6):
    best=None; best_gap=1.0
    for t in master.inst["team_ids"]:
        for s in range(master.S):
            away_frac=0.0
            for idx,col in enumerate(master.columns[t]):
                if col["HA"][s]=='A':
                    away_frac += master.lam[(t,idx)].X or 0.0
            if eps < away_frac < 1-eps:
                gap=abs(0.5-away_frac)
                if gap<best_gap: best_gap=gap; best=(t,s,away_frac)
    return best

def branch_and_price_bestbound(inst, time_limit_minutes=180, b1=5, MaxCand=30, gamma_c=1.7):
    start=time.time(); limit=60*time_limit_minutes
    incumbent={"obj":float('inf'),"sol":None}
    root={"forced_inc":frozenset(),"forced_exc":frozenset(),
          "force_home":frozenset(),"force_away":frozenset(),"bound":0.0}
    pq=[]; heapq.heappush(pq,(0.0,root)); nodes=0; root_gap=None

    while pq:
        bound,node = heapq.heappop(pq); nodes+=1
        print(f"\n[B&B] Node {nodes}: bound={bound:.2f}, |inc|={len(node['forced_inc'])}, "
              f"|exc|={len(node['forced_exc'])}, |home|={len(node['force_home'])}, |away|={len(node['force_away'])}", flush=True)

        master,status,lpobj = column_generation_with_branch(
            inst, set(node["forced_inc"]), set(node["forced_exc"]),
            set(node["force_home"]), set(node["force_away"]),
            incumbent_UB=incumbent["obj"],
            time_limit_per_lp=None, max_iters=220, gurobi_output=True
        )
        if status not in (GRB.OPTIMAL,GRB.TIME_LIMIT):
            print("[B&B]  LP failed; prune.", flush=True); continue
        if lpobj >= incumbent["obj"] - 1e-9:
            print(f"[B&B]  Pruned by bound: {lpobj:.2f} >= {incumbent['obj']:.2f}", flush=True); continue

        # LP integral?
        all_int=True
        for v in master.model.getVars():
            if abs(v.X-round(v.X))>1e-9: all_int=False; break
        if all_int:
            sol={}
            for (t,idx),var in master.lam.items():
                if var.X>0.5: sol[t]=(idx,master.columns[t][idx])
            if lpobj < incumbent["obj"]-1e-9:
                incumbent["obj"]=lpobj; incumbent["sol"]=sol
                print(f"[B&B]  LP integral -> new incumbent {lpobj:.2f}; node closed.", flush=True)
            else:
                print("[B&B]  LP integral; node closed.", flush=True)
            continue

        # candidates
        cand_events=rank_fractional_events(master)
        fallback_HA=None if cand_events else pick_fractional_HA(master)

        # IP push (root longer)
        ip_time=600 if nodes==1 else 90
        st_ip,obj_ip,sol_ip = master.solve_ip(time_limit=ip_time)
        if obj_ip is not None and obj_ip < incumbent["obj"]-1e-9:
            incumbent["obj"]=obj_ip; incumbent["sol"]=sol_ip
            print(f"[B&B]  New incumbent from IP: {obj_ip:.2f}", flush=True)

        if root_gap is None:
            root_gap = max(0.0, incumbent["obj"]-lpobj) if incumbent["obj"]<float('inf') else 1.0
            print(f"[B&B]  Root gap set to {root_gap:.2f}", flush=True)

        local_b1,local_MaxCand = (10,60) if nodes==1 else (b1,MaxCand)

        if cand_events:
            node_gap=max(0.0, incumbent["obj"]-lpobj)
            ratio = min(1.0, (node_gap/(root_gap+1e-12))) if root_gap>0 else 1.0
            num=min(len(cand_events), max(1,int(local_b1 + local_MaxCand*(ratio**gamma_c))))
            cand_events=cand_events[:num]
            print(f"[B&B]  Strong branching over {len(cand_events)} event-candidates (ratio={ratio:.3f}).", flush=True)

            best=None; best_worse=-1e18
            for (ev,val) in cand_events:
                # include child
                inc=set(node["forced_inc"]); inc.add(ev)
                _,stA,bA = column_generation_with_branch(inst, inc, set(node["forced_exc"]),
                                                         set(node["force_home"]), set(node["force_away"]),
                                                         incumbent_UB=incumbent["obj"],
                                                         time_limit_per_lp=None, max_iters=80, gurobi_output=False)
                # exclude child
                exc=set(node["forced_exc"]); exc.add(ev)
                _,stB,bB = column_generation_with_branch(inst, set(node["forced_inc"]), exc,
                                                         set(node["force_home"]), set(node["force_away"]),
                                                         incumbent_UB=incumbent["obj"],
                                                         time_limit_per_lp=None, max_iters=80, gurobi_output=False)
                worse=min(bA,bB)
                print(f"        cand {ev} -> include={bA:.2f}, exclude={bB:.2f}", flush=True)
                if worse>best_worse: best_worse=worse; best=(ev,bA,bB)
            if best is None:
                print("[B&B]  No candidate produced bounds; close node.", flush=True); continue
            ev,bA,bB=best
            print(f"[B&B]  Branch on event {ev} (min-child bound={min(bA,bB):.2f})", flush=True)
            childA={"forced_inc":frozenset(set(node["forced_inc"])|{ev}),
                    "forced_exc":node["forced_exc"],
                    "force_home":node["force_home"],"force_away":node["force_away"],"bound":bA}
            childB={"forced_inc":node["forced_inc"],
                    "forced_exc":frozenset(set(node["forced_exc"])|{ev}),
                    "force_home":node["force_home"],"force_away":node["force_away"],"bound":bB}
            if bA < incumbent["obj"]-1e-9: heapq.heappush(pq,(bA,childA))
            if bB < incumbent["obj"]-1e-9: heapq.heappush(pq,(bB,childB))
        elif fallback_HA is not None:
            t,s,_=fallback_HA
            print(f"[B&B]  Fallback H/A branching at (team={t}, slot={s})", flush=True)
            fh=set(node["force_home"]); fh.add((t,s))
            _,_,bA = column_generation_with_branch(inst,set(node["forced_inc"]),set(node["forced_exc"]),
                                                   fh,set(node["force_away"]),
                                                   incumbent_UB=incumbent["obj"],
                                                   time_limit_per_lp=None, max_iters=80, gurobi_output=False)
            fa=set(node["force_away"]); fa.add((t,s))
            _,_,bB = column_generation_with_branch(inst,set(node["forced_inc"]),set(node["forced_exc"]),
                                                   set(node["force_home"]),fa,
                                                   incumbent_UB=incumbent["obj"],
                                                   time_limit_per_lp=None, max_iters=80, gurobi_output=False)
            childH={"forced_inc":node["forced_inc"],"forced_exc":node["forced_exc"],
                    "force_home":frozenset(fh),"force_away":node["force_away"],"bound":bA}
            childA={"forced_inc":node["forced_inc"],"forced_exc":node["forced_exc"],
                    "force_home":node["force_home"],"force_away":frozenset(fa),"bound":bB}
            if bA < incumbent["obj"]-1e-9: heapq.heappush(pq,(bA,childH))
            if bB < incumbent["obj"]-1e-9: heapq.heappush(pq,(bB,childA))
        else:
            print("[B&B]  No fractional events or H/A; node closed.", flush=True)
            continue

        if time.time()-start>limit:
            print("[B&B]  Time limit reached; stopping.", flush=True); break

    return incumbent, nodes


# =========================
# Reporting & main
# =========================
def print_solution(inst, incumbent):
    if incumbent["sol"] is None:
        print("No incumbent found.", flush=True); return
    print("\nFinal incumbent objective:", incumbent["obj"], flush=True)
    print("Schedule:", flush=True)
    chosen={t:incumbent["sol"][t][1] for t in inst["team_ids"]}
    S=inst["num_slots"]; names=inst["team_names"]
    home_opp={(i,s):None for i in inst["team_ids"] for s in range(S)}
    for j in inst["team_ids"]:
        col=chosen[j]
        for s in range(S):
            if col["HA"][s]=='A':
                i=col["opponents"][s]
                if home_opp[(i,s)] is None: home_opp[(i,s)]=j
    header="Team".ljust(10)+" | "+" | ".join([f"S{s+1}" for s in range(S)])
    print(header); print("-"*len(header))
    for i in inst["team_ids"]:
        col=chosen[i]; row=[]
        for s in range(S):
            opp = home_opp[(i,s)] if col["HA"][s]=='H' else col["opponents"][s]
            row.append(f"{col['HA'][s]}{names[opp]}")
        print(names[i].ljust(10)+" | "+" | ".join(row))
        print(f"  cost: {col['cost']}")

def main(xml_path):
    inst=parse_instance(xml_path)
    print(f"Parsed instance: {inst['n']} teams, {inst['num_slots']} slots; U={inst['U']}, L={inst['L']}", flush=True)
    random.seed(42)
    start=time.time()
    incumbent,nodes = branch_and_price_bestbound(inst, time_limit_minutes=180,
                                                 b1=5, MaxCand=30, gamma_c=1.7)
    print(f"\nSearch finished in {time.time()-start:.2f}s, nodes processed: {nodes}", flush=True)
    print_solution(inst, incumbent)

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python BNP_NL_IntRC_BP_beta_PB_sym.py NL8.xml", flush=True)
        sys.exit(1)
    main(sys.argv[1])
