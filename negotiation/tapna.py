import math


def pareto_filter(actions, scores):
    non_dominated_actions = []

    for i, action in enumerate(actions):
        dominated = False
        current_scores = (
            scores["U_R"][i],
            scores["U_S"][i],
            scores["U_C"][i],
        )

        for j in range(len(actions)):
            if i == j:
                continue

            other_scores = (
                scores["U_R"][j],
                scores["U_S"][j],
                scores["U_C"][j],
            )

            if (
                all(other >= current for other, current in zip(other_scores, current_scores))
                and any(other > current for other, current in zip(other_scores, current_scores))
            ):
                dominated = True
                break

        if not dominated:
            non_dominated_actions.append(action)

    return non_dominated_actions


def compute_dynamic_weight(tss, tau=0.15, prev_wS=1/3, decay_lambda=0.3):
    if tss > tau:
        wS = 1/3 + (2/3) * min(1, (tss - tau) / (1 - tau))
    else:
        wS = 1/3 + (prev_wS - 1/3) * math.exp(-decay_lambda)

    return wS


def run_tapna(actions, U_R_list, U_S_list, U_C_list, tss, prev_wS):
    scores = {
        "U_R": U_R_list,
        "U_S": U_S_list,
        "U_C": U_C_list,
    }
    surviving_actions = pareto_filter(actions, scores)

    wS = compute_dynamic_weight(tss, prev_wS=prev_wS)
    wR = (1 - wS) / 2
    wC = (1 - wS) / 2
    weights = {
        "wR": wR,
        "wS": wS,
        "wC": wC,
    }

    best_action = None
    best_score = None

    for action in surviving_actions:
        index = actions.index(action)
        score = (
            wR * U_R_list[index] +
            wS * U_S_list[index] +
            wC * U_C_list[index]
        )

        if best_score is None or score > best_score:
            best_score = score
            best_action = action

    return best_action, weights
