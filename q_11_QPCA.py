"""
QPCA - Quantum Principal Component Analysis
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/Users/4c/Desktop/GHQ/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
PCA_DIM = 3
LAMBDA_REG = 0.01


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def compute_quantum_kernel():
    n_states = 1 << NUM_QUBITS
    fmap = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)

    statevectors = []
    for v in range(n_states):
        feat = value_to_features(v)
        circ = fmap.assign_parameters(feat)
        sv = Statevector.from_instruction(circ)
        statevectors.append(sv)

    K = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i, n_states):
            fid = abs(statevectors[i].inner(statevectors[j])) ** 2
            K[i, j] = fid
            K[j, i] = fid

    return K


def kernel_pca(K, dim=PCA_DIM):
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    vals, vecs = np.linalg.eigh(K_centered)
    idx = np.argsort(vals)[::-1][:dim]
    components = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0))
    explained = vals[idx] / np.maximum(vals.sum(), 1e-15)

    return components, explained


def pca_regression(Z, y, lam=LAMBDA_REG):
    alpha = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    pred = Z @ alpha
    return pred


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- Kvantni kernel (ZZFeatureMap, {NUM_QUBITS}q, reps=1) ---")
    K = compute_quantum_kernel()
    print(f"  Kernel matrica: {K.shape}, rang: {np.linalg.matrix_rank(K)}")

    print(f"\n--- Kernel PCA (dim={PCA_DIM}) ---")
    Z, explained = kernel_pca(K, dim=PCA_DIM)
    print(f"  Komponente: {Z.shape}")
    for d in range(PCA_DIM):
        print(f"  PC{d+1}: objasnjeno = {explained[d]:.4f}")

    print(f"\n--- QPCA regresija po pozicijama ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred = pca_regression(Z, y)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QPCA, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Kvantni kernel (ZZFeatureMap, 5q, reps=1) ---
  Kernel matrica: (32, 32), rang: 32

--- Kernel PCA (dim=3) ---
  Komponente: (32, 3)
  PC1: objasnjeno = 0.0495
  PC2: objasnjeno = 0.0463
  PC3: objasnjeno = 0.0449

--- QPCA regresija po pozicijama ---
  Poz 1 [1-33]: 4:0.073 | 3:0.064 | 5:0.063
  Poz 2 [2-34]: 6:0.086 | 7:0.076 | 5:0.075
  Poz 3 [3-35]: 20:0.053 | 12:0.053 | 26:0.047
  Poz 4 [4-36]: 17:0.064 | 27:0.062 | 21:0.055
  Poz 5 [5-37]: 18:0.072 | 28:0.069 | 22:0.052
  Poz 6 [6-38]: 17:0.047 | 19:0.044 | 16:0.041
  Poz 7 [7-39]: 18:0.057 | 27:0.049 | 29:0.049

==================================================
Predikcija (QPCA, deterministicki, seed=39):
[4, 6, 20, 27, 28, 36, 38]
==================================================
"""


"""
QPCA - Quantum Principal Component Analysis

Kernel PCA u kvantnom prostoru: centrira kvantni kernel i radi spektralnu dekompoziciju
Izvlaci 3 glavne komponente koje nose najvise informacije iz kvantnog feature prostora
Regresija u PCA prostoru: fituje empirijsku distribuciju koristeci samo top 3 komponente
Pokazuje koliko svaka komponenta objasnjava varijanse (dijagnostika strukture)
Deterministicki, brz, bez iterativnog treniranja
"""
