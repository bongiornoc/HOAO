# -*- coding: utf-8-*-
import tensorflow as tf
import numpy as np

def _letters_needed(n):
    base = list("ijklmnopqrstuvwzabcdefgh") + list("bcdefghijkmnopqrstuvwxyz")
    return base[:n]

def _latent_letters(n):
    lat = list("abcdefghijklmnopopqrstuvwxyz")
    return lat[:n]

@tf.function(jit_compile=True)
def _scale_to_unit_diag(K, order, eps=1e-12,odd_sign_correction=True):
    K = tf.convert_to_tensor(K)
    N = tf.shape(K)[0]
    diag_idx = tf.range(N)
    diag_multi = tf.stack([diag_idx]*order,axis=1)
    mu_k = tf.gather_nd(K, diag_multi)
    abs_mu = tf.abs(mu_k)
    mag = tf.pow(tf.maximum(abs_mu, tf.constant(eps,K.dtype)), 1.0/order)
    
    denom_mag = tf.ones_like(K)
    for axis in range(order):
        shp = [1]*order; shp[axis] = tf.shape(K)[0]
        denom_mag *= tf.reshape(mag,shp)

    if (order%2) == 1 and odd_sign_correction:
        sgn = tf.sign(mu_k)
        sgn = tf.where(tf.equal(sgn,0), tf.ones_like(sgn), sgn)
        denom_sgn = tf.ones_like(K)
        for axis in range(order):
            shp = [1]*order; shp[axis] = tf.shape(K)[0]
        denom_sgn *= tf.reshape(sgn, shp)
        denom = denom_mag * denom_sgn
    else:
        denom = denom_mag
    K_std = tf.math.divide_no_nan(K, denom)
    return K_std, mag, (tf.sign(mu_k) if (order%2)==1 else None)

@tf.function(jit_compile=True)
def _unscale_from_unit_diag(K_std, order, mag, sign_vec=None):
    K_std = tf.convert_to_tensor(K_std)
    denom_mag = tf.ones_like(K_std)
    for axis in range(order):
        shp = [1]*order; shp[axis] = tf.shape(K_std)[0]
        denom_mag *= tf.reshape(mag, shp)

    if (order%2) == 1 and sign_vec is not None:
        sgn = tf.where(tf.equal(sign_vec, 0), tf.ones_like(sign_vec), sign_vec)
        denom_sgn = tf.ones_like(K_std)
        for axis in range(order):
            shp = [1]*order; shp[axis] = tf.shape(K_std)[0]
            denom_sgn *= tf.reshape(sgn, shp)
        K_raw = K_std * denom_mag * denom_sgn
    else:
        K_raw = K_std * denom_mag
    return K_raw 


@tf.function(jit_compile=True)
def comoment_tensor_tf(X, order=4, standardize = True, chunk = None, dtype = tf.float32, odd_sign_correction=True, eps=1e-12):
    X = tf.convert_to_tensor(X)
    if dtype is not None:
        X = tf.cast(X, dtype)
    tf.debugging.assert_rank(X,2,message="X must be 2D with shape (N,T).")
    N = tf.shape(X)[0]
    T = tf.shape(X)[1]

    Xm = X - tf.reduce_mean(X,axis=1, keepdims = True)
    data_letters = _letters_needed(order)
    ein_lhs = ','.join([f'{c}t' for c in data_letters])
    ein_rhs = ''.join(data_letters)
    ein = ein_lhs + '->' + ein_rhs

    if chunk is None:
        K = tf.einsum(ein, *([Xm]*order)) / tf.cast(T, X.dtype)
    else:
        K0 = tf.zeros([N]*order, dtype = X.dtype) 
        def cond(s, acc):
            return tf.less(s, T)
        def body(s, acc):
            e = tf.minimum(s + chunk, T)
            Xi = Xm[:,s:e]
            acc = acc + tf.einsum(ein, *([Xi]*order))
            return (e, acc)
        _, K = tf.while_loop(cond, body, (tf.constant(0, tf.int32), K0))
        K = K / tf.cast(T, X.dtype)

    if not standardize:
        return K 
    K_std, mag, sign_vec = _scale_to_unit_diag(K, order, eps=eps, odd_sign_correction = odd_sign_correction) 
    return K_std
        

@tf.function(jit_compile=True)
def symmetric_tucker_tf(K, order=4, r=None, dtype=tf.float32, return_factors=True):
    K = tf.cast(K, dtype)
    tf.debugging.assert_rank(K, order, message=f"K must have rank {order}.")
    for d in range(1, order):
        tf.debugging.assert_equal(tf.shape(K)[0], tf.shape(K)[d], message=f"K: dim 0 != dim {d}")

    N = tf.shape(K)[0]
    r_eff = N if (r is None) else tf.convert_to_tensor(r, tf.int32)

    K1 = tf.reshape(K, (N,-1))
    M = tf.matmul(K1, K1, transpose_b = True)
    w, V = tf.linalg.eigh(M)
    idx = tf.argsort(w, direction = "DESCENDING")
    idx_top = tf.gather(idx, tf.range(r_eff))
    U = tf.gather(V, idx_top, axis=1)

    data_letters = _letters_needed(order)
    latent_letters = _latent_letters(order)
    cur  = data_letters[:]

    G = K
    for mode in range(order):
        lhs = f"{data_letters[mode]}{latent_letters[mode]}"
        rhs_in = ''.join(cur)
        rhs_out = ''.join([latent_letters[mode] if j == mode else cur[j] for j in range(order)])
        G = tf.einsum(f"{lhs},{rhs_in}->{rhs_out}", U, G)
        cur[mode] = latent_letters[mode]

    return (G,U) if return_factors else G 

@tf.function(jit_compile=True)
def core_from_factors_tf(K, U, order=4, method = "sequential"):
    if method == "single":
        data_letters = _letters_needed(order)
        latent_letters = _latent_letters(order)
        lhs = ','.join([f"{data_letters[m]}{latent_letters[m]}" for m in range(order)])
        rhs_in = ''.join(data_letters)
        rhs_out = ''.join(latent_letters)
        return tf.einsum(f"{lhs},{rhs_in}->{rhs_out}", *([U]*order), K)
    
    data_letters = _letters_needed(order)
    latent_letters = _latent_letters(order)
    cur = data_letters[:]
    G = K
    for mode in range(order):
        lhs = f"{data_letters[mode]}{latent_letters[mode]}"
        rhs_in = ''.join(cur)
        rhs_out = ''.join([latent_letters[mode] if j == mode else cur[j] for j in range(order)])
        G = tf.einsum(f"{lhs},{rhs_in}->{rhs_out}", U, G)
        cur[mode] = latent_letters[mode]
    return G

@tf.function(jit_compile=True)
def reconstruct_comoment_tf(G,U,order=4,method="sequential", dtype=tf.float32):
    G = tf.cast(G,dtype)
    U = tf.cast(U,dtype)
    tf.debugging.assert_rank(G, order, message=f"G must have rank {order}")
    for d in range(1, order):
        tf.debugging.assert_equal(tf.shape(G)[0], tf.shape(G)[d], message = f"G has mismatched dims with {d}")
    
    if method == "single":
        data_letters = _letters_needed(order)
        latent_letters = _latent_letters(order)
        lhs = ','.join([f"{data_letters[m]}{latent_letters[m]}" for m in range(order)])
        rhs_in = ''.join(latent_letters)
        rhs_out = ''.join(data_letters)
        return tf.einsum(f"{lhs},{rhs_in}->{rhs_out}", *([U]*order), G)
    
    data_letters = _letters_needed(order)
    latent_letters = _latent_letters(order)
    cur = latent_letters[:]
    K = G
    for mode in range(order):
        lhs = f"{data_letters[mode]}{latent_letters[mode]}"
        rhs_in = ''.join(cur)
        rhs_out = ''.join([data_letters[mode] if j == mode else cur[j] for j in range(order)])
        K = tf.einsum(f"{lhs},{rhs_in}->{rhs_out}", U, K)
        cur[mode] = data_letters[mode]
    return K 


def mean_core_comoment(rin,rout,order=4,r=None, standardize = True, chunk = None, dtype = tf.float32, odd_sign_correction=True, eps=1e-12):
    rin = tf.cast(tf.convert_to_tensor(rin),dtype)
    rout = tf.cast(tf.convert_to_tensor(rout), dtype)
    tf.debugging.assert_rank(rin, 3, message="rin must be rank 3: batchn, N, T")
    tf.debugging.assert_rank(rout, 3, message="rout must be rank 3: batchn, N, T")
    tf.debugging.assert_equal(tf.shape(rin)[:2], tf.shape(rout)[:2], message = "rin and rout must have matching B and N")

    B = tf.shape(rin)[0]

    @tf.function(jit_compile=True)
    def _per_pair(X_in, X_out):
        Cin = comoment_tensor_tf(X_in, order = order, standardize=standardize, chunk=chunk, dtype=dtype, odd_sign_correction=odd_sign_correction, eps=eps)
        _,U = symmetric_tucker_tf(Cin, order=order, r=r, dtype=dtype, return_factors=True)
        Cout = comoment_tensor_tf(X_out, order = order, standardize=standardize, chunk=chunk, dtype=dtype, odd_sign_correction=odd_sign_correction, eps=eps)
        Gout = core_from_factors_tf(Cout, tf.transpose(U, [0,1]), order=order, method="sequential")
        return Gout 
    
    G_sum = None 
    for X_in, X_out in zip(tf.unstack(rin,axis=0), tf.unstack(rout,axis=0)):
        Gi = _per_pair(X_in, X_out)
        if G_sum is None:
            G_sum = tf.zeros_like(Gi)
        G_sum = G_sum + Gi
    return G_sum / tf.cast(B, dtype)


def HOAO(X, order=4, r=None, Tin_frac=0.5, n_folds=20, standardize=True, chunk=None, dtype=tf.float32, seed=None, odd_sign_correction=True, eps=1e-12, return_filtered=True, return_all=False):
    X = np.asarray(X)
    T, N = X.shape
    Tin = int(Tin_frac*T)
    Tout = T - Tin 
    rng = np.random.default_rng(seed)

    rin = []
    rout = []

    for _ in range(n_folds):
        idx = rng.permutation(T)
        idx_in = rng.choice(idx, size=Tin, replace=False)
        idx_out = rng.choice(idx, size=Tout, replace=False)
        rin.append(X[idx_in].T)
        rout.append(X[idx_out].T)
    rin = np.stack(rin, axis=0)
    rout = np.stack(rout, axis=0)

    clean_core_std = mean_core_comoment(rin, rout, order=order, r=r, standardize=standardize, chunk=chunk, dtype=dtype, odd_sign_correction=odd_sign_correction, eps=eps)

    Xtf = tf.convert_to_tensor(X.T, dtype=dtype)
    K_full = comoment_tensor_tf(Xtf, order=order, standardize=standardize, chunk=chunk, dtype=dtype, odd_sign_correction=odd_sign_correction, eps=eps)
    _, U_full = symmetric_tucker_tf(K_full, order=order, r=r, dtype=dtype, return_factors=True)

    K_clean = reconstruct_comoment_tf(clean_core_std, U_full, order=order, method="sequential", dtype=dtype)

    if not standardize:
        return K_clean 
    
    K_full_raw = comoment_tensor_tf(Xtf, order=order, standardize=False, chunk=chunk, dtype=dtype)
    K_full_std, mag, sign_vec = _scale_to_unit_diag(K_full_raw, order, eps=eps, odd_sign_correction = odd_sign_correction)

    K = _unscale_from_unit_diag(K_clean, order, mag, sign_vec if (order%2)==1 else None)
    return K


                    
    