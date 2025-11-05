# -*- coding: utf-8 -*-
import tensorflow as tf

# ============================================================
# 1) Cokurtosi: costruzione del tensore (ordine 4)
# ============================================================

@tf.function(jit_compile=True)
def cokurtosi_tensore_tf(X, standardizza=False, chunk=None, dtype=tf.float32):
    """
    Calcola il tensore di co-kurtosi K_{ijkl} = E[(X_i-μ_i)(X_j-μ_j)(X_k-μ_k)(X_l-μ_l)]
    a partire da X ∈ R^{N×T} usando TensorFlow.

    Parametri
    ---------
    X : Tensor o array-like, shape (N, T)
    standardizza : bool
        Se True, divide per σ_i σ_j σ_k σ_l (dimensionless).
    chunk : int o None
        Se non None, accumula a blocchi lungo T per ridurre l’uso di memoria.
    dtype : tf.DType
        Dtype dei calcoli (tf.float32 consigliato per GPU).

    Ritorna
    -------
    K : Tensor, shape (N, N, N, N)
    """
    X = tf.convert_to_tensor(X)
    if dtype is not None:
        X = tf.cast(X, dtype)

    # Controlli di forma
    tf.debugging.assert_rank(X, 2, message="X deve essere 2D con shape (N, T).")
    N = tf.shape(X)[0]
    T = tf.shape(X)[1]

    # Centra per rimuovere le medie lungo il tempo
    Xm = X - tf.reduce_mean(X, axis=1, keepdims=True)

    # Accumulo (eventualmente chunked) del momento 4°
    if chunk is None:
        K = tf.einsum('it,jt,kt,lt->ijkl', Xm, Xm, Xm, Xm) / tf.cast(T, X.dtype)
    else:
        # while_loop per lavorare a blocchi e ridurre picchi di memoria
        K0 = tf.zeros((N, N, N, N), dtype=X.dtype)

        def cond(s, acc):
            return tf.less(s, T)

        def body(s, acc):
            e = tf.minimum(s + chunk, T)
            Xi = Xm[:, s:e]  # (N, t_blk)
            acc = acc + tf.einsum('it,jt,kt,lt->ijkl', Xi, Xi, Xi, Xi)
            return (e, acc)

        _, K = tf.while_loop(cond, body, (tf.constant(0, dtype=tf.int32), K0))
        K = K / tf.cast(T, X.dtype)

    if standardizza:
        # σ_i σ_j σ_k σ_l con outer product; divisione no-nan
        s = tf.math.reduce_std(Xm, axis=1)  # popolazione
        denom = tf.einsum('i,j,k,l->ijkl', s, s, s, s)
        K = tf.math.divide_no_nan(K, denom)

    return K


# ============================================================
# 2) Decomposizione Tucker simmetrica (HOSVD) e core
# ============================================================
@tf.function(jit_compile=True)
def tucker_simmetrico_tf(K, r=None, dtype=tf.float32, return_fattori=True):
    """
    Estrae U (N×r) via HOSVD simmetrico dal tensore K (N,N,N,N) e calcola G (r,r,r,r).
    Se r è None, usa r = N.
    """
    K = tf.cast(K, dtype)
    tf.debugging.assert_rank(K, 4, message="K deve avere rank 4.")
    N = tf.shape(K)[0]
    tf.debugging.assert_equal(tf.shape(K)[0], tf.shape(K)[1], message="K: dim 0 != dim 1")
    tf.debugging.assert_equal(tf.shape(K)[0], tf.shape(K)[2], message="K: dim 0 != dim 2")
    tf.debugging.assert_equal(tf.shape(K)[0], tf.shape(K)[3], message="K: dim 0 != dim 3")

    # Gestione di r=None in Python (consente a tf.function di tracciare correttamente)
    if r is None:
        r_eff = N                              # Tensor (dimensione dipende da K)
    else:
        r_eff = tf.convert_to_tensor(r, tf.int32)  # Tensor scalare

    # Gram della matricizzazione K_(1): M = K1 @ K1^T
    K1 = tf.reshape(K, (N, -1))                   # (N, N^3)
    M  = tf.matmul(K1, K1, transpose_b=True)      # (N, N) PSD

    # Autovettori principali (ordinati decrescente)
    w, V = tf.linalg.eigh(M)                      # ascendente
    idx = tf.argsort(w, direction='DESCENDING')   # (N,)

    # Primi r_eff indici e colonne
    idx_top = tf.gather(idx, tf.range(r_eff))     # (r,)
    U = tf.gather(V, idx_top, axis=1)             # (N, r)

    # Core simmetrico: G = K ×1 U^T ×2 U^T ×3 U^T ×4 U^T
    T1 = tf.einsum('ia,ijkl->ajkl', U, K)         # mode-1
    T2 = tf.einsum('jb,ajkl->abkl', U, T1)        # mode-2
    T3 = tf.einsum('kc,abkl->abcl', U, T2)        # mode-3
    G  = tf.einsum('ld,abcl->abcd', U, T3)        # mode-4  -> (r,r,r,r)

    return (G, U) if return_fattori else G


@tf.function(jit_compile=True)
def core_da_fattori_tf(K, U, metodo="sequenziale"):
    """
    Applica i fattori U (N×r) al tensore K (N,N,N,N) per ottenere il core G (r,r,r,r).

    Parametri
    ---------
    K : Tensor (N,N,N,N)
    U : Tensor (N,r)
    metodo : 'sequenziale' (default) oppure 'singolo' (einsum unico)

    Ritorna
    -------
    G : Tensor (r,r,r,r)
    """
    K = tf.convert_to_tensor(K)
    U = tf.convert_to_tensor(U)

    if metodo == "singolo":
        return tf.einsum('ia,jb,kc,ld,ijkl->abcd', U, U, U, U, K)

    # Sequenziale: spesso più parsimonioso in memoria
    T1 = tf.einsum('ia,ijkl->ajkl', U, K)
    T2 = tf.einsum('jb,ajkl->abkl', U, T1)
    T3 = tf.einsum('kc,abkl->abcl', U, T2)
    G  = tf.einsum('ld,abcl->abcd', U, T3)
    return G

@tf.function(jit_compile=True)
def ricostruisci_cokurtosi_tf(G, U, metodo="sequenziale", dtype=tf.float32):
    """
    Ricostruisce K a partire dal core G (r,r,r,r) e dal fattore U (N,r) uguale su tutte le modalità.
    Implementa: K = G ×1 U ×2 U ×3 U ×4 U

    Parametri
    ---------
    G : Tensor (r,r,r,r)
    U : Tensor (N,r)
    metodo : 'sequenziale' (default) oppure 'singolo' (einsum unico)
    dtype : tf.DType

    Ritorna
    -------
    K : Tensor (N,N,N,N)
    """
    G = tf.cast(G, dtype)
    U = tf.cast(U, dtype)

    # Controlli di forma
    tf.debugging.assert_rank(G, 4, message="G deve avere rank 4 (r,r,r,r).")
    r0 = tf.shape(G)[0]
    tf.debugging.assert_equal(r0, tf.shape(G)[1], message="G: dim 0 != dim 1")
    tf.debugging.assert_equal(r0, tf.shape(G)[2], message="G: dim 0 != dim 2")
    tf.debugging.assert_equal(r0, tf.shape(G)[3], message="G: dim 0 != dim 3")
    tf.debugging.assert_equal(r0, tf.shape(U)[1], message="U seconda dimensione deve eguagliare r di G.")

    if metodo == "singolo":
        # Un'unica einsum
        # Indici: U (i,a), U (j,b), U (k,c), U (l,d), G (a,b,c,d) -> K (i,j,k,l)
        return tf.einsum('ia,jb,kc,ld,abcd->ijkl', U, U, U, U, G)

    # Sequenziale (meno picchi di memoria)
    T1 = tf.einsum('ia,abcd->ibcd', U, G)   # ×1 U
    T2 = tf.einsum('jb,ibcd->ijcd', U, T1)  # ×2 U
    T3 = tf.einsum('kc,ijcd->ijkd', U, T2)  # ×3 U
    K  = tf.einsum('ld,ijkd->ijkl', U, T3)  # ×4 U
    return K


# ============================================================
# 3) Pipeline batch: media del core su coppie (rin[i], rout[i])
# ============================================================

def media_core_cokurtosi_tf(rin, rout, r=None, standardizza=True, chunk=None, dtype=tf.float32):
    """
    Replica il loop:
        C = cokurtosi(rin[i]); (G,U)=tucker(C); Cout=cokurtosi(rout[i]); acc += core(Cout,U)
    e restituisce la media su i.

    Parametri
    ---------
    rin, rout : Tensor o array-like, shape (B, N, T)
    r : int o None
    standardizza : bool
    chunk : int o None
    dtype : tf.DType

    Ritorna
    -------
    ccore : Tensor (r,r,r,r)  (o (N,N,N,N) se r=None)
    """
    rin = tf.cast(tf.convert_to_tensor(rin), dtype)
    rout = tf.cast(tf.convert_to_tensor(rout), dtype)

    tf.debugging.assert_rank(rin, 3, message="rin deve essere (B,N,T).")
    tf.debugging.assert_rank(rout, 3, message="rout deve essere (B,N,T).")
    tf.debugging.assert_equal(tf.shape(rin)[:2], tf.shape(rout)[:2], message="rin e rout devono avere la stessa shape.")

    B = tf.shape(rin)[0]
    N = tf.shape(rin)[1]
    r_eff = N if r is None else int(r)

    @tf.function(jit_compile=True)
    def _per_coppia(X_in, X_out):
        C_in = cokurtosi_tensore_tf(X_in, standardizza=standardizza, chunk=chunk, dtype=dtype)
        _, U  = tucker_simmetrico_tf(C_in, r=r, dtype=dtype, return_fattori=True)
        C_out = cokurtosi_tensore_tf(X_out, standardizza=standardizza, chunk=chunk, dtype=dtype)
        G_out = core_da_fattori_tf(C_out, U, metodo="sequenziale")
        return G_out

    # Accumula sui batch (usa tensorflow per il corpo pesante)
    # Python loop esterno è ok: il lavoro vero sta dentro funzioni compilate.
    ccore = None
    for i in range(int(B.numpy()) if isinstance(B, tf.Tensor) and not tf.executing_eagerly() else B):
        G_i = _per_coppia(rin[i], rout[i])
        if ccore is None:
            ccore = tf.zeros_like(G_i)
        ccore = ccore + G_i

    ccore = ccore / tf.cast(B, dtype)
    return ccore


