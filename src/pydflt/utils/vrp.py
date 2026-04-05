#!/usr/bin/env python
# Source: https://github.com/khalil-research/CaVE
"""
Vehicle routing probelm
"""

from collections import defaultdict

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel

# WARNING: for the sake of simplicity and the guarantee of pure Binary variables,
# the case where only one node is accessed is not taken into account. However,
# this problem can be handeled by adding a dummy of depot.


class VrpABModel(optGrbModel):
    """
    This abstract class is optimization model for capacitated vehicle routing probelm

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
        demands (list(int)): List of customer demands
        capacity (int): Vehicle capacity
        num_vehicles (int): Number of vehicle
    """

    def __init__(self, num_nodes, demands, capacity, num_vehicles):
        """
        Args:
            num_nodes (int): number of nodes
            demands (list(int)): customer demands
            capacity (int): vehicle capacity
            num_vehicles (int): number of vehicle
        """
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]
        self.demands = demands
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        super().__init__()

    @property
    def num_cost(self):
        return len(self.edges)

    def setObj(self, c):  # noqa: N802
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        obj = gp.quicksum(c[i] * self.x[e] for i, e in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        # solve
        self._model.optimize()
        sol = np.zeros(len(self.edges), dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = int(np.round(self.x[e].x))
        return sol, self._model.objVal

    def getTour(self, sol):  # noqa: N802
        """
        A method to get a tour from solution

        Args:
            sol (list): solution

        Returns:
            list: a VRP tour
        """
        # active edges
        edges = defaultdict(list)
        for i, (u, v) in enumerate(self.edges):
            if sol[i] > 1e-2:
                edges[u].append(v)
                edges[v].append(u)
        # get tour
        route = []
        while edges[0]:  # candidates
            v_curr = 0
            tour = [0]
            v_next = edges[v_curr][0]
            # remove used edges
            edges[v_curr].remove(v_next)
            edges[v_next].remove(v_curr)
            while v_next != 0:
                tour.append(v_next)
                # go to next node
                if not edges[v_next]:  # visit single customer
                    v_curr, v_next = v_next, 0
                else:
                    v_curr, v_next = v_next, edges[v_next][0]
                    # remove used edges
                    edges[v_curr].remove(v_next)
                    edges[v_next].remove(v_curr)
            # back to depot
            tour.append(0)
            route.append(tour)
        return route

    def copy(self):
        """
        A method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = type(self)(self.num_nodes, self.demands, self.capacity, self.num_vehicles)
        # copy params
        for attr in dir(self._model.Params):
            if not attr.startswith("_"):
                try:
                    # get value
                    val = self._model.getParamInfo(attr)[2]
                    # set value
                    new_model._model.setParam(attr, val)
                except gp.GurobiError:
                    # ignore non-param
                    pass
        return new_model


class VrpModel(VrpABModel):
    """
    This class is optimization model for capacitated vehicle routing probelm

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
        demands (list(int)): List of customer demands
        capacity (int): Vehicle capacity
        num_vehicles (int): Number of vehicle
    """

    def _getModel(self):  # noqa: N802
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("vrp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        for i, j in self.edges:
            x[j, i] = x[i, j]
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstr(x.sum(0, "*") <= 2 * self.num_vehicles)  # depot degree
        m.addConstrs(x.sum(i, "*") == 2 for i in self.nodes if i != 0)  # 2 degree
        # activate lazy constraints
        m._x = x
        m._q = {i: self.demands[i - 1] for i in self.nodes[1:]}
        m._Q = self.capacity
        m._edges = self.edges
        m.Params.lazyConstraints = 1
        return m, x

    def _vrp_callback(self, model, where):
        """
        A method to add k-path lazy constraints for CVRP
        """
        if where == GRB.Callback.MIPSOL:
            # check subcycle with unionfind
            uf = UnionFind(self.num_nodes)
            for u, v in model._edges:
                if u == 0 or v == 0:
                    continue
                if model.cbGetSolution(model._x[u, v]) > 1e-2:
                    uf.union(u, v)
            # go through subcycles
            for component in uf.get_components():
                if len(component) < 3:
                    continue
                # rounded capacity inequalities
                k = int(np.ceil(np.sum([model._q[v] for v in component]) / model._Q))
                # edges with both end-vertex in S
                edges_s = [(u, v) for u in component for v in component if u < v]
                # add k-path cut
                if len(component) >= 3 and ((len(edges_s) >= len(component)) or (k > 1)):
                    # constraint expression
                    constr = gp.quicksum(model._x[e] for e in edges_s) <= len(component) - k
                    # add lazy constraints
                    model.cbLazy(constr)
                    # store lazy constraints to find all binding constraints
                    self.lazy_constraints.append(constr)

    def solve(self):
        """
        A method to solve model
        """
        # init lazy constraints
        self.lazy_constraints = []

        # create a callback function with access to method variables
        def vrp_callback(model, where):
            self._vrp_callback(model, where)

        # solve
        self._model.optimize(vrp_callback)
        sol = np.zeros(len(self.edges), dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = int(np.round(self.x[e].x))
        return sol, self._model.objVal


class VrpModel2(VrpABModel):
    """
    This class is optimization model for capacitated vehicle routing probelm

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
        demands (list(int)): List of customer demands
        capacity (int): Vehicle capacity
        num_vehicles (int): Number of vehicle
    """

    def _getModel(self):  # noqa: N802
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("vrp")
        # turn off output
        m.Params.outputFlag = 0
        # variables
        directed_edges = [*self.edges, *[(j, i) for (i, j) in self.edges]]
        x = m.addVars(directed_edges, name="x", vtype=GRB.BINARY)
        u = m.addVars(self.nodes, name="u", lb=[0, *list(self.demands)], ub=self.capacity, vtype=GRB.CONTINUOUS)
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(gp.quicksum(x[i, j] for j in self.nodes if j != i) == 1 for i in self.nodes if i != 0)  # 2 degree
        m.addConstrs(gp.quicksum(x[i, j] for i in self.nodes if i != j) == 1 for j in self.nodes if j != 0)  # 2 degree
        m.addConstr(x.sum(0, "*") <= self.num_vehicles)  # depot degree
        m.addConstr(x.sum("*", 0) <= self.num_vehicles)  # depot degree
        m.addConstrs(u[i] - u[j] + self.capacity * x[i, j] <= self.capacity - self.demands[j - 1] for i, j in directed_edges if i != 0 and j != 0)  # capacity
        return m, x

    def setObj(self, c):  # noqa: N802
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        obj = gp.quicksum(c[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if self.x[i, j].x > 1e-2 or self.x[j, i].x > 1e-2:
                sol[k] = 1
        return sol, self._model.objVal

    def relax(self):
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = VrpModel2Rel(self.num_nodes, self.demands, self.capacity, self.num_vehicles)
        return model_rel


class VrpModel2Rel(VrpModel2):
    """
    This class is relaxation of vrpModel2.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):  # noqa: N802
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("vrp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        directed_edges = [*self.edges, *[(j, i) for (i, j) in self.edges]]
        x = m.addVars(directed_edges, name="x", vtype=GRB.CONTINUOUS)
        u = m.addVars(self.nodes, name="u", lb=[0, *list(self.demands)], ub=self.capacity, vtype=GRB.CONTINUOUS)
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(gp.quicksum(x[i, j] for j in self.nodes if j != i) == 1 for i in self.nodes if i != 0)  # 2 degree
        m.addConstrs(gp.quicksum(x[i, j] for i in self.nodes if i != j) == 1 for j in self.nodes if j != 0)  # 2 degree
        m.addConstr(x.sum(0, "*") <= self.num_vehicles)  # depot degree
        m.addConstr(x.sum("*", 0) <= self.num_vehicles)  # depot degree
        m.addConstrs(u[i] - u[j] + self.capacity * x[i, j] <= self.capacity - self.demands[j - 1] for i, j in directed_edges if i != 0 and j != 0)  # capacity
        return m, x

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(len(self.edges))
        for k, (i, j) in enumerate(self.edges):
            sol[k] = self.x[i, j].x + self.x[j, i].x
        return sol, self._model.objVal

    def relax(self):
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")

    def getTour(self, sol):  # noqa: N802
        """
        A forbidden method to get a tour from solution
        """
        raise RuntimeError("Relaxation Model has no integer solution.")


class UnionFind:
    """
    Union-find disjoint sets that provides methods to perform find and union
    operations on elements.
    """

    def __init__(self, n):
        """
        A method to create the union-find structure.
        Args:
            n (int): number of elements
        """
        self.parent = list(range(n))

    def find(self, i):
        """
        A method to find the root of the set that element 'i' belongs to.
        """
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """
        A method to perform the union of the sets that contain elements 'i' and 'j'
        """
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False

    def get_components(self):
        """
        A method to list all disjoint sets in the current structure.
        """
        comps = defaultdict(list)
        for i in range(len(self.parent)):
            comps[self.find(i)].append(i)
        return list(comps.values())
