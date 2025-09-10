# Install missing packages (required when running the notebook)
# Use notebook magic so packages are installed into the current kernel environment.
#%pip install "tensorflow-probability>=0.19" --quiet

import numpy as np
import sys
import tensorflow as tf
import tensorflow_probability as tfp

# provide alias/module shim for packages/code that do `import tf_keras`
# some code/tools expect a separate `tf_keras` package; make it point to tf.keras
sys.modules['tf_keras'] = tf.keras

tfk = tf.keras
tfm = tf.math
tfl = tf.linalg

# ---------- Utilità vincolo sum(w)=1 via parametrizzazione w = b + Q z ----------
def _householder_basis_sum_to_one(n: int, dtype=tf.float64):
    """Ritorna Q (n x (n-1)) e b (n,) tali che w = b + Q z e 1^T w = 1 per ogni z."""
    one = tf.ones([n], dtype)
    norm = tfl.norm(one)
    e1 = tf.concat([tf.ones([1], dtype), tf.zeros([n-1], dtype)], axis=0)
    v = one + norm * e1
    v = v / tfl.norm(v)
    # H = I - 2 v v^T
    H = tf.eye(n, dtype=dtype) - 2.0 * tf.tensordot(v, v, axes=0)
    Q = H[:, 1:]  # colonne 2..n: base del sottospazio ortogonale a 1
    b = one / tf.cast(n, dtype)
    return Q, b

# ---------- Momenti, obiettivo e gradiente analitico ----------
def _make_value_and_grad(Sigma_t, C_std_t, alpha_t, sigma_t, Q, b):
    """Costruisce value_and_gradients(z) -> (f, grad_z) per TFP L-BFGS."""
    @tf.function(jit_compile=True)
    def value_and_grad(z):
        # w = b + Q z
        w = b + tfl.matvec(Q, z)

        # var = w^T Sigma w
        Sw = tfl.matvec(Sigma_t, w)
        var = tf.tensordot(w, Sw, axes=1)

        # m4 con ws = w * sigma (std marginali)
        ws = w * sigma_t
        m4 = tf.einsum('ijkl,i,j,k,l->', C_std_t, ws, ws, ws, ws)

        # gradiente: g = 2 Sigma w + alpha * 4 * sigma * (C_std ⋅ (ws,ws,ws))
        core = tf.einsum('ijkl,j,k,l->i', C_std_t, ws, ws, ws)  # lunghezza n
        g_w = 2.0 * Sw + alpha_t * (4.0 * sigma_t * core)

        # gradiente in z: g_z = Q^T g_w (perché dw/dz = Q)
        g_z = tfl.matvec(tf.transpose(Q), g_w)

        f = var + alpha_t * m4
        return f, g_z
    return value_and_grad

# ---------- Solver L-BFGS sul sottospazio ----------
def min_var_with_cokurtosis_lbfgs_tf(
    Sigma: np.ndarray,
    C_std: np.ndarray,
    alpha: float,
    w0: np.ndarray = None,
    max_iter: int = 500,
    tol: float = 1e-9,
    dtype: str = "float64",
):
    """
    Minimizza f(w)=w^T Sigma w + alpha*m4(w) con vincolo sum(w)=1
    usando L-BFGS (TF Probability) sull’(n-1)-spazio dei pesi.
    """
    assert Sigma.ndim == 2 and Sigma.shape[0] == Sigma.shape[1]
    n = Sigma.shape[0]
    assert C_std.shape == (n, n, n, n)
    assert alpha >= 0

    tf_dtype = tf.float32 if dtype == "float32" else tf.float64
    Sigma_t  = tf.convert_to_tensor(Sigma, dtype=tf_dtype)
    C_std_t  = tf.convert_to_tensor(C_std, dtype=tf_dtype)
    alpha_t  = tf.convert_to_tensor(alpha, dtype=tf_dtype)

    # std marginali dai diagonali di Sigma (clippate a >=0)
    sigma_t = tf.sqrt(tf.maximum(tf.linalg.diag_part(Sigma_t), tf.constant(0.0, tf_dtype)))

    # Base Q e punto baricentrico b per il vincolo
    Q, b = _householder_basis_sum_to_one(n, tf_dtype)

    # Punto iniziale in coordinate z
    if w0 is None:
        w0_t = b
    else:
        w0_t = tf.convert_to_tensor(np.asarray(w0, float).reshape(n), dtype=tf_dtype)
        # forza il vincolo (in caso w0 non lo soddisfi numericamente)
        w0_t = w0_t - (tf.reduce_sum(w0_t) - 1.0) / tf.cast(n, tf_dtype)
    z0 = tfl.matvec(tf.transpose(Q), (w0_t - b))

    # Value & gradient per L-BFGS (analitici, compilati)
    value_and_grad = _make_value_and_grad(Sigma_t, C_std_t, alpha_t, sigma_t, Q, b)

    # Lancio L-BFGS
    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=value_and_grad,
        initial_position=z0,
        max_iterations=max_iter,
        tolerance=tol,
        parallel_iterations=1,   # deterministico
    )

    z_star = results.position
    w_star = b + tfl.matvec(Q, z_star)

    # Diagnostiche finali
    Sw = tfl.matvec(Sigma_t, w_star)
    m2 = float(tf.tensordot(w_star, Sw, axes=1).numpy())
    sigma_p = float(tf.sqrt(tf.maximum(m2, 0.)).numpy())
    ws = w_star * sigma_t
    m4 = float(tf.einsum('ijkl,i,j,k,l->', C_std_t, ws, ws, ws, ws).numpy())
    kappa = (m4 / (sigma_p**4)) if sigma_p > 0 else np.nan

    out = dict(
        w=w_star.numpy(),
        f=float(results.objective_value.numpy()),
        #grad_norm=float(tfl.norm(results.gradient).numpy()),
        it=int(results.num_iterations.numpy()),
        converged=bool(results.converged.numpy()),
        m2=m2, sigma_p=sigma_p, m4=m4, kappa=kappa,
    )
    return out
