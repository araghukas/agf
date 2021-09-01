import numpy as np
from numpy.linalg import inv
from agf.utility import fold_matrix, unfold_matrix


def decimate(arr: np.ndarray, d: int = 3) -> np.ndarray:
    if not arr.ndim == 2:
        raise ValueError("input array is not 2-dimensional")

    M = arr.shape[0]
    if not arr.shape[1] == M:
        raise ValueError("input array is not square.")
    if arr.shape[0] % d != 0:
        raise ValueError(f"can not divide {M}x{M} array into {d}x{d} blocks.")

    arr = fold_matrix(arr, d, d)  # returns a new array
    arr = _decimate(arr, M // d)
    return unfold_matrix(arr)


def _decimate(_arr: np.ndarray, N: int) -> np.ndarray:
    n = 1
    k = 2
    hi = 3
    lo = 1
    while hi < N:
        g_lo = inv(_arr[lo][lo])
        g_hi = inv(_arr[hi][hi])

        # connect top <-> before <-> center
        _arr[0][0] = _arr[0][0] - _arr[0][1] @ g_lo @ _arr[lo][lo + 1]
        _arr[0][1] = -_arr[0][1] @ g_lo @ _arr[lo][lo + 1]

        # connect before <-> center <-> after
        _arr[k][k] -= (_arr[k][k - 1] @ g_lo @ _arr[lo][lo + 1]
                       + _arr[k][k + 1] @ g_hi @ _arr[lo][lo - 1])
        _arr[k][k - 1] = -_arr[k][k - 1] @ _arr[lo][lo] @ _arr[lo][lo - 1]
        _arr[k][k + 1] = -_arr[k][k + 1] @ _arr[hi][hi] @ _arr[hi][hi + 1]

        # now have top <-> after
        # after becomes new center
        n += 1
        k = 2**n
        hi = k + k // 2
        lo = hi - k

    return _arr


if __name__ == "__main__":
    # global example array
    np.set_printoptions(linewidth=500, precision=1)
    N_atoms = 20
    H = np.zeros((N_atoms, N_atoms))
    ff = 5
    f = 3
    H[0][0] = ff
    for i in range(1, N_atoms):
        H[i][i] = ff
        H[i][i - 1] = -f
        H[i - 1][i] = -f

    Hd = decimate(H, 1)
    print(H)
    print(Hd)
