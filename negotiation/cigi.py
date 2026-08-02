def compute_cigi(U_R, U_S, U_C, weights) -> float:
    return float(
        weights["wR"] * U_R +
        weights["wS"] * U_S +
        weights["wC"] * U_C
    )
