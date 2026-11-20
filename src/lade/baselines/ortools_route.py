"""OR-Tools baseline: shortest OPEN path from the courier's position through all open packages.
Pickup option: time windows as soft constraints (penalty per minute late), travel at a fixed speed.
NOT YET RUN (OR-Tools was not installed where this was written). Check against the official LaDe code:
the paper does not say which cost or time limit it used.
"""
import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .rules import by_n


def solve_one(xy, tw_end_min=None, speed_km_min=0.25, late_penalty=10, time_limit_ms=200):
    """xy [n,2] km relative to courier. Returns local visiting order."""
    n = len(xy)
    if n == 1:
        return np.array([0])
    pts = np.vstack([[0, 0], xy])                      # node 0 = courier start
    D = np.rint(np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1)) * 1000).astype(int)   # metres
    D = np.vstack([np.hstack([D, np.zeros((n + 1, 1), int)]), np.zeros((1, n + 2), int)])       # dummy end, free
    m = pywrapcp.RoutingIndexManager(n + 2, 1, [0], [n + 1])
    r = pywrapcp.RoutingModel(m)
    cb = r.RegisterTransitCallback(lambda i, j: int(D[m.IndexToNode(i), m.IndexToNode(j)]))
    r.SetArcCostEvaluatorOfAllVehicles(cb)
    if tw_end_min is not None:
        tcb = r.RegisterTransitCallback(lambda i, j: int(D[m.IndexToNode(i), m.IndexToNode(j)] / 1000 / speed_km_min))
        r.AddDimension(tcb, 10_000, 10_000, True, "time")
        dim = r.GetDimensionOrDie("time")
        for k in range(n):
            dim.SetCumulVarSoftUpperBound(m.NodeToIndex(k + 1), max(0, int(tw_end_min[k])), late_penalty)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(time_limit_ms)
    sol = r.SolveWithParameters(p)
    order, i = [], r.Start(0)
    while not r.IsEnd(i):
        node = m.IndexToNode(i)
        if 1 <= node <= n:
            order.append(node - 1)
        i = sol.Value(r.NextVar(i))
    return np.array(order)


def ortools_route(nodes, use_windows=False, time_limit_ms=200):
    out = pd.Series(0, index=nodes.index, dtype=np.int64)
    for n, idx, a in by_n(nodes, ["dx_km", "dy_km", "to_window_end"]):
        ranks = np.empty((a["dx_km"].shape[0], n), int)
        for b in range(a["dx_km"].shape[0]):
            o = solve_one(np.c_[a["dx_km"][b], a["dy_km"][b]], a["to_window_end"][b] if use_windows else None,
                          time_limit_ms=time_limit_ms)
            ranks[b, o] = np.arange(1, n + 1)
        out.loc[idx] = ranks.ravel()
    return out
