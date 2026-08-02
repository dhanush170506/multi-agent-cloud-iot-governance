def greedy_cost_baseline(actions, U_C_list):
    best_index = max(range(len(actions)), key=lambda index: U_C_list[index])
    return actions[best_index]


def greedy_latency_baseline(actions, U_R_list):
    best_index = max(range(len(actions)), key=lambda index: U_R_list[index])
    return actions[best_index]


def static_weighted_baseline(
    actions,
    U_R_list,
    U_S_list,
    U_C_list
):
    best_index = max(
        range(len(actions)),
        key=lambda index: (
            (1 / 3) * U_R_list[index] +
            (1 / 3) * U_S_list[index] +
            (1 / 3) * U_C_list[index]
        ),
    )
    return actions[best_index]
