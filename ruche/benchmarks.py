import os
import json
import argparse
import time
import math
import numpy as np
import tensorflow as tf
from pathlib import Path

from scipy.stats import wishart, invwishart, multivariate_normal, multivariate_t, geninvgauss
from scipy.linalg import sqrtm
from numpy.linalg import inv


from comoments import HOAO, comoment_tensor_tf

# ------------------------------------------------------------
# TensorFlow GPU setup
# ------------------------------------------------------------
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')  # lock to first GPU
            print(f"[INFO] Using GPU: {gpus[0].name}")
        except Exception as e:
            print(f"[WARN] Failed to set memory growth or device visibility: {e}")
    else:
        print("[WARN] No GPU detected. This script is intended for GPUs.")

# ------------------------------------------------------------
# Helpers: Frobenius norm (TensorFlow)
# ------------------------------------------------------------
def frob_norm_tf(tensor):
    t = tf.reshape(tensor, [-1])
    return tf.norm(t, ord='euclidean')

# ------------------------------------------------------------
# Ground truth tensors (Normal & Student-t)
# ------------------------------------------------------------
def true_coskew_normal(N):
    # zero for symmetric distributions
    return tf.zeros((N, N, N), dtype=tf.float32)

def true_cokurt_from_sigma(Sigma, factor=1.0):
    # Isserlis/Wick: E[x_i x_j x_k x_l] = Σ_ij Σ_kl + Σ_ik Σ_jl + Σ_il Σ_jk
    # Multiply by factor for Student-t.
    # Build with TensorFlow for device locality.
    Sigma_tf = tf.convert_to_tensor(Sigma, dtype=tf.float32)
    term1 = tf.einsum('ij,kl->ijkl', Sigma_tf, Sigma_tf)
    term2 = tf.einsum('ik,jl->ijkl', Sigma_tf, Sigma_tf)
    term3 = tf.einsum('il,jk->ijkl', Sigma_tf, Sigma_tf)
    K4 = term1 + term2 + term3
    if factor != 1.0:
        K4 = K4 * tf.constant(factor, dtype=tf.float32)
    return K4

def t_fourth_factor(nu):
    # For X ~ t_ν(0, Σ) (scale-param Σ), E[x_i x_j x_k x_l] =
    # [ν^2 / ((ν-2)(ν-4))] * (Σ_ij Σ_kl + Σ_ik Σ_jl + Σ_il Σ_jk), ν>4.
    return (nu * nu) / ((nu - 2.0) * (nu - 4.0))

# ------------------------------------------------------------
# GH target via two-pass streaming Monte Carlo
# X = W*gamma + sqrt(W)*Z A^T, with Z ~ N(0, I), W ~ GIG(p, b) (SciPy's geninvgauss)
# Steps:
#   1) Stream to compute mean μ_hat (N,)
#   2) Stream to accumulate central order-3 and order-4 tensors
# ------------------------------------------------------------
def gh_stream_generator(n_total, batch_size, N, A, p, b, gamma_vec, rng):
    # Generates batches of size <= batch_size, returns (batch: (B, N))
    batches = int(math.ceil(n_total / batch_size))
    for bi in range(batches):
        B = batch_size if (bi < batches - 1) else (n_total - batch_size * (batches - 1))
        # W ~ GIG(p, b); SciPy uses parameters (p, b)
        W = geninvgauss.rvs(p, b, size=B, random_state=rng)
        Z = rng.standard_normal((B, N))
        # X = W*gamma + sqrt(W) * Z @ A^T
        sqrtW = np.sqrt(W)[:, None]
        X = (W[:, None] * gamma_vec) + (sqrtW * (Z @ A.T))
        yield X

def gh_population_target(N, A, p, b, gamma_vec, n_mc=10_000_000, batch_size=200_000, seed=12345):
    # Two-pass streaming central comoment tensors
    rng = np.random.default_rng(seed)

    # Pass 1: mean
    count = 0
    mean = np.zeros(N, dtype=np.float64)
    for X in gh_stream_generator(n_mc, batch_size, N, A, p, b, gamma_vec, rng):
        mean += X.sum(axis=0)
        count += X.shape[0]
    mean /= count

    # Pass 2: central tensors
    # Accumulate sums of outer^3 and outer^4
    M3 = np.zeros((N, N, N), dtype=np.float64)
    M4 = np.zeros((N, N, N, N), dtype=np.float64)

    rng2 = np.random.default_rng(seed)  # same stream to regenerate
    for X in gh_stream_generator(n_mc, batch_size, N, A, p, b, gamma_vec, rng2):
        Xc = X - mean  # (B, N)
        # order-3: sum over samples of x ⊗ x ⊗ x
        # (B, N) -> (N,N,N) via einsum: sum_b x_b,i x_b,j x_b,k
        M3 += np.einsum('bi,bj,bk->ijk', Xc, Xc, Xc, optimize=True)
        # order-4: sum over samples of x ⊗ x ⊗ x ⊗ x
        M4 += np.einsum('bi,bj,bk,bl->ijkl', Xc, Xc, Xc, Xc, optimize=True)

    M3 /= count
    M4 /= count

    # Convert to tf
    K3 = tf.convert_to_tensor(M3, dtype=tf.float32)
    K4 = tf.convert_to_tensor(M4, dtype=tf.float32)
    return K3, K4

# ------------------------------------------------------------
# Experiment factories (fixed parameters per experiment)
# ------------------------------------------------------------
def sample_inverse_wishart_scale(N, df=None, rng=None):
    """
    Draw a covariance matrix Sigma ~ InvWishart(df, I_N) directly using scipy.stats.invwishart.
    """
    if df is None:
        df = N + 10  # comfortably > N; (df > N+1 gives finite mean)
    if rng is None:
        rng = np.random.default_rng(0)
    S0 = np.eye(N)
    Sigma = invwishart.rvs(df=df, scale=S0, random_state=rng)
    return Sigma

def make_A_from_sigma(Sigma):
    # A = sqrtm(Sigma) (GH requirement)
    A = sqrtm(Sigma)
    A = np.real_if_close(A)
    return A

def factory_mvn_identity(N, rng):
    Sigma = np.eye(N)
    A = sqrtm(Sigma)  # identity
    A = np.real_if_close(A)
    cfg = {"dist": "mvn_identity", "N": N, "Sigma": Sigma, "A": A}
    return cfg

def factory_mvn_invwishart(N, rng, df=None):
    Sigma = sample_inverse_wishart_scale(N, df=df, rng=rng)
    A = make_A_from_sigma(Sigma)
    cfg = {"dist": "mvn_invwishart", "N": N, "Sigma": Sigma, "A": A}
    return cfg

def factory_t_invwishart(N, rng, nu, df=None):
    Sigma = sample_inverse_wishart_scale(N, df=df, rng=rng)
    A = make_A_from_sigma(Sigma)
    cfg = {"dist": "t_invwishart", "N": N, "nu": nu, "Sigma": Sigma, "A": A}
    return cfg

def factory_gh_invwishart(N, rng, p, b, gamma_type="zero", gamma_std=0.1, df=None):
    Sigma = sample_inverse_wishart_scale(N, df=df, rng=rng)
    A = make_A_from_sigma(Sigma)
    if gamma_type == "zero":
        gamma_vec = np.zeros(N)
    elif gamma_type == "normal":
        gamma_vec = rng.normal(0.0, gamma_std, size=N)
    else:
        raise ValueError("gamma_type must be 'zero' or 'normal'")
    cfg = {
        "dist": "gh_invwishart",
        "N": N, "p": p, "b": b,
        "gamma_type": gamma_type,
        "gamma_std": gamma_std,
        "Sigma": Sigma, "A": A, "gamma_vec": gamma_vec
    }
    return cfg

# ------------------------------------------------------------
# Data generators (per replication) — parameters are fixed by factory
# ------------------------------------------------------------

def gen_data_mvn(T, A, rng):
    """
    Generate T samples from a multivariate normal distribution with zero mean
    and covariance Sigma = A A^T, using scipy.stats.multivariate_normal.
    """
    N = A.shape[0]
    Sigma = A @ A.T
    X = multivariate_normal.rvs(
        mean=np.zeros(N),
        cov=Sigma,
        size=T,
        random_state=rng
    )
    # Ensure X has shape (T, N) even when T=1
    if X.ndim == 1:
        X = X[None, :]
    return X


def gen_data_t(T, Sigma, nu, rng):
    # SciPy multivariate_t.rvs: shape=Sigma, df=nu, mean=0
    X = multivariate_t.rvs(loc=np.zeros(Sigma.shape[0]), shape=Sigma, df=nu, size=T, random_state=rng)
    if X.ndim == 1:  # T=1 guard
        X = X[None, :]
    return X

def gen_data_gh(T, A, p, b, gamma_vec, rng):
    W = geninvgauss.rvs(p, b, size=T, random_state=rng)
    Z = rng.standard_normal((T, A.shape[0]))
    X = (W[:, None] * gamma_vec) + (np.sqrt(W)[:, None] * (Z @ A.T))
    return X

# ------------------------------------------------------------
# Single replication run (returns errors for K3 and K4, both estimators)
# ------------------------------------------------------------
def run_replication(X_np, K3_true, K4_true, n_folds=20, tin_frac=0.5, seed=123):
    # Convert to TF on GPU
    X_tf = tf.convert_to_tensor(X_np.T, dtype=tf.float32)  # (N, T)

    # Sample estimator
    K3_samp = comoment_tensor_tf(X_tf, order=3, standardize=False, chunk=None, dtype=tf.float32)
    K4_samp = comoment_tensor_tf(X_tf, order=4, standardize=False, chunk=None, dtype=tf.float32)

    # HOAO estimator
    K3_hoao = HOAO(X_np, order=3, r=None, Tin_frac=tin_frac, n_folds=n_folds,
                   standardize=False, chunk=None, dtype=tf.float32, seed=seed)
    K4_hoao = HOAO(X_np, order=4, r=None, Tin_frac=tin_frac, n_folds=n_folds,
                   standardize=False, chunk=None, dtype=tf.float32, seed=seed)

    # Errors (Frobenius)
    e3_s = float(frob_norm_tf(K3_samp - K3_true).numpy())
    e3_h = float(frob_norm_tf(K3_hoao - K3_true).numpy())
    e4_s = float(frob_norm_tf(K4_samp - K4_true).numpy())
    e4_h = float(frob_norm_tf(K4_hoao - K4_true).numpy())

    return e3_s, e3_h, e4_s, e4_h

# ------------------------------------------------------------
# Per-experiment driver (fixed parameters; 500 replications)
# ------------------------------------------------------------
def benchmark_experiment(cfg, T, n_rep=500, seed=42,
                         hoao_folds=20, hoao_tin_frac=0.5,
                         gh_target_mc=10_000_000, gh_target_batch=200_000):
    rng = np.random.default_rng(seed)

    dist = cfg["dist"]
    N = cfg["N"]
    Sigma = cfg.get("Sigma", None)
    A = cfg.get("A", None)

    # Build targets
    if dist.startswith("mvn"):
        K3_true = true_coskew_normal(N)
        K4_true = true_cokurt_from_sigma(Sigma, factor=1.0)

        dof = np.nan
        gh_p = np.nan
        gh_b = np.nan
        gh_gamma_type = np.nan
        gh_gamma_std = np.nan

    elif dist == "t_invwishart":
        nu = cfg["nu"]
        if nu <= 4:
            raise ValueError("For Student-t benchmarks we require nu > 4.")
        K3_true = true_coskew_normal(N)
        fac = t_fourth_factor(nu)
        K4_true = true_cokurt_from_sigma(Sigma, factor=fac)

        dof = float(nu)
        gh_p = np.nan
        gh_b = np.nan
        gh_gamma_type = np.nan
        gh_gamma_std = np.nan

    elif dist == "gh_invwishart":
        p = float(cfg["p"])
        b = float(cfg["b"])
        gamma_vec = cfg["gamma_vec"].astype(float)

        # GH population target via streaming MC (two-pass)
        K3_true, K4_true = gh_population_target(
            N=N, A=A, p=p, b=b, gamma_vec=gamma_vec,
            n_mc=gh_target_mc, batch_size=gh_target_batch, seed=seed
        )

        dof = np.nan
        gh_p = p
        gh_b = b
        gh_gamma_type = cfg["gamma_type"]
        gh_gamma_std = float(cfg["gamma_std"])

    else:
        raise ValueError("Unknown distribution.")

    # Replications
    e3_s_list, e3_h_list, e4_s_list, e4_h_list = [], [], [], []
    for rpi in range(n_rep):
        # Fixed parameters per experiment – only data changes
        rseed = seed + 10_000 + rpi  # decorrelate replications
        rrng = np.random.default_rng(rseed)

        if dist == "mvn_identity":
            X = gen_data_mvn(T, A, rrng)
        elif dist == "mvn_invwishart":
            X = gen_data_mvn(T, A, rrng)
        elif dist == "t_invwishart":
            X = gen_data_t(T, Sigma, cfg["nu"], rrng)
        elif dist == "gh_invwishart":
            X = gen_data_gh(T, A, gh_p, gh_b, cfg["gamma_vec"], rrng)
        else:
            raise ValueError("Unknown distribution branch.")

        e3_s, e3_h, e4_s, e4_h = run_replication(
            X_np=X, K3_true=K3_true, K4_true=K4_true,
            n_folds=hoao_folds, tin_frac=hoao_tin_frac, seed=rseed
        )
        e3_s_list.append(e3_s)
        e3_h_list.append(e3_h)
        e4_s_list.append(e4_s)
        e4_h_list.append(e4_h)
    better3 = np.mean(np.array(e3_h_list) < np.array(e3_s_list))
    better4 = np.mean(np.array(e4_h_list) < np.array(e4_s_list))

    # Aggregate
    out = {
        "dist": dist, "N": int(N), "T": int(T),
        "dof": dof,
        "gh_p": gh_p, "gh_b": gh_b,
        "gh_gamma_type": gh_gamma_type, "gh_gamma_std": gh_gamma_std,
        "sample_coskew_err_mean": float(np.mean(e3_s_list)),
        "sample_coskew_err_std": float(np.std(e3_s_list, ddof=1)),
        "hoao_coskew_err_mean": float(np.mean(e3_h_list)),
        "hoao_coskew_err_std": float(np.std(e3_h_list, ddof=1)),
        "sample_cokurt_err_mean": float(np.mean(e4_s_list)),
        "sample_cokurt_err_std": float(np.std(e4_s_list, ddof=1)),
        "hoao_cokurt_err_mean": float(np.mean(e4_h_list)),
        "hoao_cokurt_err_std": float(np.std(e4_h_list, ddof=1)),
        "frac_hoao_better_coskew": float(better3),
        "frac_hoao_better_cokurt": float(better4),
    }
    return out

# ------------------------------------------------------------
# Experiment grid
# ------------------------------------------------------------
def build_experiments(N_list, T_list, seed_base=20251110):
    exps = []
    for N in N_list:
        # Fixed RNG per experiment constructor to lock parameters
        # (scale matrix, gamma, etc) once and for all
        # Note: We use distinct seeds per factory so each exp is fixed & reproducible.
        # ----- MVN -----
        exps.append(("mvn_identity", lambda N=N: factory_mvn_identity(N, np.random.default_rng(seed_base+N*101+1))))
        exps.append(("mvn_invwishart", lambda N=N: factory_mvn_invwishart(N, np.random.default_rng(seed_base+N*101+2))))

        # ----- Student-t (invwishart) -----
        for nu in [100.0, 20.0, 10.0, 8.0, 4.5]:
            exps.append((f"t_invwishart_nu{nu}", lambda N=N, nu=nu: factory_t_invwishart(N, np.random.default_rng(seed_base+N*101+int(nu*10)+3), nu=nu)))

        # ----- GH (invwishart): four scenarios -----
        # i) gamma=0, p=0.5, b=20
        exps.append(("gh_invwishart_p0.5_b20_gamma0",
                     lambda N=N: factory_gh_invwishart(N, np.random.default_rng(seed_base+N*101+4),
                                                       p=0.5, b=20.0, gamma_type="zero", gamma_std=0.0)))
        # ii) gamma ~ N(0, 0.1^2)
        exps.append(("gh_invwishart_p0.5_b20_gammaN",
                     lambda N=N: factory_gh_invwishart(N, np.random.default_rng(seed_base+N*101+5),
                                                       p=0.5, b=20.0, gamma_type="normal", gamma_std=0.1)))
        # iii) gamma=0, p=0.5, b=1
        exps.append(("gh_invwishart_p0.5_b1_gamma0",
                     lambda N=N: factory_gh_invwishart(N, np.random.default_rng(seed_base+N*101+6),
                                                       p=0.5, b=1.0, gamma_type="zero", gamma_std=0.0)))
        # iv) gamma ~ N(0, 0.1^2)
        exps.append(("gh_invwishart_p0.5_b1_gammaN",
                     lambda N=N: factory_gh_invwishart(N, np.random.default_rng(seed_base+N*101+7),
                                                       p=0.5, b=1.0, gamma_type="normal", gamma_std=0.1)))
    # Expand over T_list by duplicating
    expanded = []
    for name, ctor in exps:
        for T in T_list:
            expanded.append((name, ctor, T))
    return expanded

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_list", type=str, default="20",
                        help="Comma-separated list of N (e.g., 10,20)")
    parser.add_argument("--T_list", type=str, default="50000",
                        help="Comma-separated list of T (e.g., 20000,100000)")
    parser.add_argument("--replications", type=int, default=500,
                        help="Number of replications per experiment")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--out_csv", type=str, default="benchmark_results.csv")
    parser.add_argument("--hoao_folds", type=int, default=20)
    parser.add_argument("--hoao_tin_frac", type=float, default=0.5)
    parser.add_argument("--gh_target_mc", type=int, default=10_000_000,
                        help="MC sample count for GH population target")
    parser.add_argument("--gh_target_batch", type=int, default=200_000,
                        help="Batch size when streaming GH target")
    args = parser.parse_args()

    setup_gpu()

    N_list = [int(x) for x in args.N_list.split(",")]
    T_list = [int(x) for x in args.T_list.split(",")]

    exps = build_experiments(N_list, T_list, seed_base=args.seed)

    # Write CSV header
    out_path = Path(args.out_csv)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        with open(out_path, "w") as f:
            f.write(",".join([
                "dist","N","T","dof","gh_p","gh_b","gh_gamma_type","gh_gamma_std",
                "sample_coskew_err_mean","sample_coskew_err_std",
                "hoao_coskew_err_mean","hoao_coskew_err_std",
                "sample_cokurt_err_mean","sample_cokurt_err_std",
                "hoao_cokurt_err_mean","hoao_cokurt_err_std",
                "frac_hoao_better_coskew","frac_hoao_better_cokurt"
            ]) + "\n")

    # Run
    for name, ctor, T in exps:
        cfg = ctor()  # fixed params per experiment
        # Sanity align name and cfg["dist"] (informational only)
        print(f"\n[EXP] {name} | N={cfg['N']} | T={T}")
        res = benchmark_experiment(
            cfg=cfg, T=T, n_rep=args.replications, seed=args.seed,
            hoao_folds=args.hoao_folds, hoao_tin_frac=args.hoao_tin_frac,
            gh_target_mc=args.gh_target_mc, gh_target_batch=args.gh_target_batch
        )
        # Write row
        with open(out_path, "a") as f:
            f.write(",".join([
                str(res["dist"]), str(res["N"]), str(res["T"]),
                str(res["dof"]),
                str(res["gh_p"]), str(res["gh_b"]),
                str(res["gh_gamma_type"]), str(res["gh_gamma_std"]),
                f"{res['sample_coskew_err_mean']:.8e}",
                f"{res['sample_coskew_err_std']:.8e}",
                f"{res['hoao_coskew_err_mean']:.8e}",
                f"{res['hoao_coskew_err_std']:.8e}",
                f"{res['sample_cokurt_err_mean']:.8e}",
                f"{res['sample_cokurt_err_std']:.8e}",
                f"{res['hoao_cokurt_err_mean']:.8e}",
                f"{res['hoao_cokurt_err_std']:.8e}",
                f"{res['frac_hoao_better_coskew']:.4f}",
                f"{res['frac_hoao_better_cokurt']:.4f}",
            ]) + "\n")

    print(f"\n[DONE] Results saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
