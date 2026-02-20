#source: https://github.com/ML-KULeuven/Solver-Free-DFL/

import time
from collections import deque

import numpy as np
import scipy.linalg
import gurobipy as gp
from gurobipy import GRB


def convert_to_slack_form(model: gp.Model) -> gp.Model:
    slack_model = gp.Model("slack_form")
    slack_model.setParam("OutputFlag", 0)

    var_map = {v.varName: slack_model.addVar(lb=v.lb, vtype="C", name=v.varName) for v in model.getVars()}
    slack_model.update()

    for i, constr in enumerate(model.getConstrs()):
        sense = constr.sense
        lhs = gp.LinExpr()
        row = model.getRow(constr)
        for j in range(row.size()):
            lhs += row.getCoeff(j) * var_map[row.getVar(j).varName]
        rhs = constr.rhs

        if sense == GRB.LESS_EQUAL:
            slack_var = slack_model.addVar(lb=0, name=f"slack_{i}")
            slack_model.addConstr(lhs + slack_var == rhs)
        elif sense == GRB.GREATER_EQUAL:
            slack_var = slack_model.addVar(lb=0, name=f"slack_{i}")
            slack_model.addConstr(-lhs + slack_var == -rhs)
        else:
            slack_model.addConstr(lhs == rhs)

    for var in model.getVars():
        if var.ub != float("inf"):
            slack_var = slack_model.addVar(lb=0, name=f"slack_{var.varName}_ub")
            slack_model.addConstr(var_map[var.varName] + slack_var == var.ub)

    obj_expr = gp.LinExpr()
    for var in model.getVars():
        obj_expr += var.obj * var_map[var.varName]
    slack_model.setObjective(obj_expr, model.ModelSense)

    slack_model.update()
    return slack_model


def get_constraints_matrix_form_slack_model(model: gp.Model) -> tuple[np.ndarray, np.ndarray]:
    xs = model.getVars()
    A = []
    b = []
    for constr in model.getConstrs():
        if constr.sense != GRB.EQUAL:
            raise ValueError("Constraints have to be of the form Ax == b")
        a_i = []
        for x in xs:
            a_i.append(model.getCoeff(constr, x))
        A.append(a_i)
        b.append(constr.rhs)
    if not A:
        return np.zeros((0, len(xs)), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.array(A), np.array(b)


def get_adjacent_vertices(slack_model: gp.Model, A: np.ndarray, max_iterations_degenerate: int = 250,
    max_adjacent_vertices: int = 1000) -> list[np.ndarray]:
    if A.ndim < 2 or A.size == 0:
        return []
    all_vars = slack_model.getVars()
    sol = np.array([var.x for var in all_vars])

    basis_status = np.array([var.VBasis for var in all_vars])
    basic_indices = np.where(basis_status == 0)[0]
    non_basic_indices = np.where(basis_status != 0)[0]

    sigma = np.sum(sol[basic_indices] < 1e-10)
    if sigma == 0:
        return get_adjacent_vertices_non_degenerate_case(A, basic_indices, non_basic_indices, sol,
                                                         max_adjacent_vertices=max_adjacent_vertices)
    use_tnp_rule = True
    return list(
        get_adjacent_vertices_degenerate_case(A, basic_indices, non_basic_indices, sol, use_tnp_rule,
                                              max_iterations=max_iterations_degenerate, max_adjacent_vertices=max_adjacent_vertices)
    )


def get_adjacent_vertices_non_degenerate_case(
    A: np.ndarray,
    basic_indices: np.ndarray,
    non_basic_indices: np.ndarray,
    sol: np.ndarray,
    max_adjacent_vertices: int = 1000
) -> list[np.ndarray]:
    A_basic = A[:, basic_indices]
    dirs = scipy.linalg.solve(-A_basic, A[:, non_basic_indices])
    basic_var_values = sol[basic_indices]

    adjacent_vertices = []
    for i, entering_var_index in enumerate(non_basic_indices):
        direction = dirs[:, i]
        indices_dir_neg = np.where(direction < 0)[0]
        basic_var_values_dir_neg = basic_var_values[indices_dir_neg]
        negative_dir_values = direction[indices_dir_neg]
        ratios = -basic_var_values_dir_neg / negative_dir_values
        if ratios.size != 0:
            min_ratio = np.min(ratios)
            complete_dir = np.zeros(A.shape[1])
            complete_dir[basic_indices] = direction
            complete_dir[entering_var_index] = 1
            new_sol = sol + min_ratio * complete_dir
            adjacent_vertices.append(new_sol)
        if len(adjacent_vertices) >= max_adjacent_vertices:
            return adjacent_vertices

    return adjacent_vertices


def get_adjacent_vertices_degenerate_case(
    A: np.ndarray,
    basic_indices: np.ndarray,
    non_basic_indices: np.ndarray,
    sol: np.ndarray,
    use_tnp_rule: bool,
    max_iterations: int = 250,
    max_adjacent_vertices: int = 1000
) -> set[tuple[float, ...]]:
    basic_indices = np.array(sorted(basic_indices))
    non_basic_indices = np.array(sorted(non_basic_indices))

    t_found = False
    while not t_found:
        A_basic = A[:, basic_indices]
        if A_basic.shape[0] != A_basic.shape[1]:
            print("Skipping degenerate adjacent vertices: non-square basis matrix.")
            return set()
        basic_var_values = sol[basic_indices]
        A_non_basic = A[:, non_basic_indices]
        dirs = scipy.linalg.solve(A_basic, A_non_basic)
        indices_basic_var_zero = np.where(basic_var_values == 0)[0]
        dirs_indices_basic_var_zero = dirs[indices_basic_var_zero, :]
        indices_transition_columns = np.where(np.all(dirs_indices_basic_var_zero <= 0, axis=0))[0]
        if len(indices_transition_columns) > 0:
            t_found = True
        else:
            basic_indices, non_basic_indices = search_for_transition_column(A, basic_indices, non_basic_indices, sol)

    t = non_basic_indices[indices_transition_columns][0]

    B = A[:, basic_indices]
    B_inv = np.linalg.inv(B)
    x_B = sol[basic_indices]
    elements_still_to_fix = np.where(x_B == 0)[0]

    if use_tnp_rule:
        tmp = B_inv @ -A
        cols = find_partial_lexipos(tmp, required_rows=elements_still_to_fix)
        if cols is not None:
            basic_indices_2 = cols
            B_hat = -A[:, basic_indices_2]
            B_hat_prime = scipy.linalg.solve(B, B_hat)
            B_inv_L = np.concatenate([np.expand_dims(x_B, axis=1), B_hat_prime], axis=1)
            if not is_lexicofeasible(B_inv_L):
                return set()
        else:
            B_hat = A[:, basic_indices]
    else:
        B_hat = A[:, basic_indices]

    all_adjacent_vertices = set()
    visited_bases = {tuple(sorted(basic_indices))}

    queue = [(basic_indices, non_basic_indices)]
    iteration = 0
    while queue and (len(all_adjacent_vertices) < max_adjacent_vertices):
        iteration += 1
        if iteration > max_iterations:
            break
        curr_basic_indices, curr_non_basic_indices = queue.pop(0)

        adjacent_vertices, new_basic_indices_list, new_non_basic_indices_list = get_adjacent_vertices_degenerate_case_helper(
            A, curr_basic_indices, curr_non_basic_indices, sol, t, B_hat
        )

        for adjacent_vertex in adjacent_vertices:
            all_adjacent_vertices.add(tuple(adjacent_vertex))

        if len(queue) < max_iterations:
            for new_basic_indices, new_non_basic_indices in zip(new_basic_indices_list, new_non_basic_indices_list):
                new_basic_indices_tuple = tuple(sorted(new_basic_indices))
                if new_basic_indices_tuple not in visited_bases:
                    visited_bases.add(new_basic_indices_tuple)
                    queue.append((sorted(new_basic_indices), sorted(new_non_basic_indices)))

    return all_adjacent_vertices


def get_adjacent_vertices_degenerate_case_helper(
    A: np.ndarray,
    basic_indices: np.ndarray,
    non_basic_indices: np.ndarray,
    sol: np.ndarray,
    t: int,
    B_hat: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    A_basic = A[:, basic_indices]
    if A_basic.shape[0] != A_basic.shape[1]:
        print("Skipping degenerate adjacent vertices: non-square basis matrix.")
        return [], [], []
    B_hat_prime = scipy.linalg.solve(A_basic, B_hat)
    basic_var_values = sol[basic_indices]
    A_non_basic = A[:, non_basic_indices]
    dirs = scipy.linalg.solve(A_basic, A_non_basic)

    t_index = np.where(non_basic_indices == t)[0]
    x_B = sol[basic_indices]

    EPSILON = 1e-10
    indices_basic_var_zero = np.where(basic_var_values == 0)[0]
    transition_column = dirs[indices_basic_var_zero, :]
    transition_column = transition_column[:, t_index]
    assert np.all(transition_column <= EPSILON), "Transition column must have all elements <= 0"

    adjacent_vertices = []
    new_basic_indices_list = []
    new_non_basic_indices_list = []

    for j, entering_var_index in enumerate(non_basic_indices):
        direction = dirs[:, j]
        indices_dir_pos = np.where(dirs[:, j] > EPSILON)[0]
        basic_var_values_dir_pos = basic_var_values[indices_dir_pos]
        positive_dir_values = direction[indices_dir_pos]
        ratios = basic_var_values_dir_pos / positive_dir_values

        if ratios.size != 0:
            min_ratio = np.min(ratios)
            min_ratio_indices = indices_dir_pos[np.where(ratios == min_ratio)[0]]

            if min_ratio > 0:
                complete_dir = np.zeros(A.shape[1])
                complete_dir[basic_indices] = dirs[:, j]
                complete_dir[entering_var_index] = -1
                new_sol = sol - min_ratio * complete_dir
                adjacent_vertices.append(new_sol)

            elif min_ratio == 0:
                ratios_2 = dirs[min_ratio_indices, t_index] / dirs[min_ratio_indices, j]
                max_value = np.max(ratios_2)
                num_maximizers = np.sum(ratios_2 == max_value)
                if num_maximizers == 1:
                    i = min_ratio_indices[np.argmax(ratios_2)]
                else:
                    indices = [q for q in range(len(ratios_2)) if ratios_2[q] == max_value]
                    candidates = min_ratio_indices[indices]

                    lex_vectors = []
                    for i in candidates:
                        ratio = x_B[i] / direction[i]
                        row_vector = B_hat_prime[i, :] / direction[i]
                        lex_vector = np.concatenate([[ratio], row_vector])
                        lex_vectors.append((i, lex_vector))
                    i = min(lex_vectors, key=lambda x: tuple(x[1]))[0]

                leaving_var_index = basic_indices[i]
                new_basic_indices = np.copy(basic_indices)
                new_basic_indices[i] = entering_var_index
                new_non_basic_indices = np.copy(non_basic_indices)
                new_non_basic_indices[j] = leaving_var_index

                new_basic_indices_list.append(new_basic_indices)
                new_non_basic_indices_list.append(new_non_basic_indices)

    return adjacent_vertices, new_basic_indices_list, new_non_basic_indices_list


def search_for_transition_column(
    A: np.ndarray,
    basic_indices: np.ndarray,
    non_basic_indices: np.ndarray,
    sol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    while True:
        A_basic = A[:, basic_indices]
        dirs = np.linalg.solve(-A_basic, A[:, non_basic_indices])
        basic_var_values = sol[basic_indices]

        EPSILON = 1e-10
        i = np.random.choice(len(non_basic_indices))
        entering_var_index = non_basic_indices[i]

        direction = dirs[:, i]
        indices_dir_neg = np.where(direction < -EPSILON)[0]
        basic_var_values_dir_neg = basic_var_values[indices_dir_neg]
        negative_dir_values = direction[indices_dir_neg]
        ratios = -basic_var_values_dir_neg / negative_dir_values

        if ratios.size != 0:
            min_ratio = np.min(ratios)
            min_ratio_indices = indices_dir_neg[np.where(ratios == min_ratio)[0]]
            if min_ratio > 0:
                break
            min_ratio_index = np.random.choice(min_ratio_indices)
            leaving_var_index = basic_indices[min_ratio_index]

            new_basic_indices = np.copy(basic_indices)
            new_basic_indices[min_ratio_index] = entering_var_index
            new_non_basic_indices = np.copy(non_basic_indices)
            new_non_basic_indices[i] = leaving_var_index

            basic_indices = new_basic_indices
            non_basic_indices = new_non_basic_indices

    return basic_indices, non_basic_indices


def find_partial_lexipos(A: np.ndarray, required_rows: np.ndarray) -> list[int] | None:
    start_time = time.time()
    m, n = A.shape
    R = list(required_rows)

    pos_cols = {i: list(np.where(A[i] > 0)[0]) for i in R}
    neg_cols = {i: set(np.where(A[i] < 0)[0]) for i in R}
    R_order = sorted(R, key=lambda i: len(pos_cols[i]))

    pivots = {}
    graph = {}

    def has_cycle() -> bool:
        visited = {}

        def dfs(u):
            visited[u] = 1
            for v in graph.get(u, ()):
                if visited.get(v, 0) == 1:
                    return True
                if visited.get(v, 0) == 0 and dfs(v):
                    return True
            visited[u] = 2
            return False

        for u in graph:
            if visited.get(u, 0) == 0 and dfs(u):
                return True
        return False

    def backtrack_req(k: int):
        if time.time() - start_time > 3:
            return None
        if k == len(R_order):
            return True
        row = R_order[k]
        for c in pos_cols[row]:
            old_graph = {u: set(v) for u, v in graph.items()}
            pivots[row] = c
            graph.setdefault(c, set())

            for prev_row in R_order[:k]:
                prev_c = pivots[prev_row]
                if c in neg_cols[prev_row]:
                    graph[prev_c].add(c)
                if prev_c in neg_cols[row]:
                    graph[c].add(prev_c)

            if not has_cycle():
                result = backtrack_req(k + 1)
                if result is None:
                    return None
                if result:
                    return True

            graph.clear()
            graph.update(old_graph)
            pivots.pop(row, None)
        return False

    if backtrack_req(0) is None:
        return None
    if not backtrack_req(0):
        return None

    P = set(pivots.values())
    extras_needed = m - len(P)
    extras = [c for c in range(n) if c not in P][:extras_needed]
    if len(extras) < extras_needed:
        return None

    S = list(P) + extras

    for c in extras:
        graph.setdefault(c, set())
        for i in R:
            if c in neg_cols[i]:
                graph[pivots[i]].add(c)

    in_deg = {c: 0 for c in S}
    for u in graph:
        for v in graph[u]:
            if v in in_deg:
                in_deg[v] += 1

    q = deque([c for c in S if in_deg[c] == 0])
    order = []
    while q:
        if time.time() - start_time > 3:
            return None
        u = q.popleft()
        order.append(u)
        for v in graph.get(u, ()):
            if v in in_deg:
                in_deg[v] -= 1
                if v in in_deg and in_deg[v] == 0:
                    q.append(v)

    return order if len(order) == m else None


def is_lexicofeasible(A: np.ndarray) -> bool:
    for row in A:
        non_zero_elements = row[row != 0]
        if len(non_zero_elements) > 0 and non_zero_elements[0] <= 0:
            return False
    return True
