import numpy as np
from typing import Sequence, Optional, List

try:
    import numba as nb
    NUMBA_AVAILABLE = True
except Exception:
    nb = None
    NUMBA_AVAILABLE = False


# ------------------ Numba helpers (se disponibili) ------------------

if NUMBA_AVAILABLE:
    @nb.njit(inline='always')
    def _build_allowed_candidates(bi: np.ndarray, kept_mask: np.ndarray, out_idx: np.ndarray) -> int:
        """
        bi, kept_mask: uint8 (0/1) lunghi universe.
        Scrive in out_idx gli indici ammessi e ritorna il conteggio.
        """
        cnt = 0
        n = bi.shape[0]
        for v in range(n):
            if bi[v] == 1 and kept_mask[v] == 0:
                out_idx[cnt] = v
                cnt += 1
        return cnt

    @nb.njit(inline='always')
    def _partial_shuffle_pick_first_k(a: np.ndarray, n: int, k: int):
        """
        Fisher–Yates parziale: a[0:k] diventa un k-subset uniforme di a[0:n].
        """
        for i in range(k):
            j = np.random.randint(i, n)  # [i, n)
            tmp = a[i]
            a[i] = a[j]
            a[j] = tmp

    @nb.njit
    def _chain_window_numba(bitsets_u8_window: np.ndarray, m: int,
                            kept_mask: np.ndarray, candidates: np.ndarray,
                            out_int32: np.ndarray) -> None:
        """
        bitsets_u8_window: (k, universe) uint8 0/1 per la finestra contigua.
        out_int32: (k, m) dove scriviamo il risultato.
        """
        k = bitsets_u8_window.shape[0]
        universe = bitsets_u8_window.shape[1]

        # Step 0: campiona m dalla prima lista
        for v in range(universe):
            kept_mask[v] = 0
        cnt = _build_allowed_candidates(bitsets_u8_window[0], kept_mask, candidates)
        if cnt < m:
            raise ValueError("Prima lista insufficiente per m (no replacement).")
        _partial_shuffle_pick_first_k(candidates, cnt, m)
        for t in range(m):
            out_int32[0, t] = candidates[t]

        # Step 1..k-1
        for i in range(1, k):
            bi = bitsets_u8_window[i]
            kept_count = 0
            # tieni i sopravvissuti
            for t in range(m):
                val = out_int32[i-1, t]
                if bi[val] == 1:
                    out_int32[i, kept_count] = val
                    kept_count += 1

            need = m - kept_count
            if need > 0:
                # escludi i già tenuti
                for v in range(universe):
                    kept_mask[v] = 0
                for t in range(kept_count):
                    kept_mask[out_int32[i, t]] = 1

                cnt = _build_allowed_candidates(bi, kept_mask, candidates)
                if cnt < need:
                    raise ValueError("Lista della finestra insufficiente per arrivare a m.")
                _partial_shuffle_pick_first_k(candidates, cnt, need)
                for t in range(need):
                    out_int32[i, kept_count + t] = candidates[t]

    @nb.njit
    def _numba_seed(seed: int) -> None:
        np.random.seed(seed)


# ------------------ Fallback NumPy (molto veloce comunque) ------------------

def _chain_window_numpy(bitsets_u8_window: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """
    bitsets_u8_window: (k, universe) uint8 0/1
    Ritorna (k, m) uint16.
    """
    k, universe = bitsets_u8_window.shape
    bitsets = bitsets_u8_window.astype(bool, copy=False)

    out = np.empty((k, m), dtype=np.uint16)

    # step 0
    b0 = bitsets[0]
    base_vals = np.flatnonzero(b0)
    if base_vals.size < m:
        raise ValueError("Prima lista insufficiente per m (no replacement).")
    sample = rng.choice(base_vals, size=m, replace=False).astype(np.uint16)
    out[0] = sample

    # step 1..k-1
    kept_mask = np.zeros(universe, dtype=bool)
    for i in range(1, k):
        bi = bitsets[i]
        kept = sample[bi[sample]]
        need = m - kept.size
        if need:
            kept_mask[:] = False
            kept_mask[kept] = True
            allowed = np.logical_and(bi, np.logical_not(kept_mask))
            candidates = np.flatnonzero(allowed)
            if candidates.size < need:
                raise ValueError("Lista della finestra insufficiente per arrivare a m.")
            repl = rng.choice(candidates, size=need, replace=False).astype(np.uint16)
            sample = np.concatenate([kept, repl])
        else:
            sample = kept
        out[i] = sample
    return out


# ------------------ Classe principale ------------------

class WindowBitsetSampler:
    """
    Preprocessa una volta le tue K liste (valori interi >=0) in bitset uint8
    e permette di fare sampling su qualsiasi finestra contigua di lunghezza k:
      - step 0: campiona m senza replacement dalla prima lista della finestra
      - step i>0: tieni i sopravvissuti e rimpiazzi i mancanti da lista_i evitando duplicati
    Usa Numba JIT (compilato in __init__) se disponibile, con fallback NumPy.
    """

    def __init__(self,
                 lists: Sequence[Sequence[int] | np.ndarray],
                 universe: Optional[int] = None,
                 use_numba: bool = True,
                 warmup: bool = True,
                 seed: Optional[int] = None):
        """
        lists: sequenza di K liste/array con valori interi >=0 (non serve che siano unici).
        universe:
            - None -> inferito automaticamente come max_val + 1 dai dati
            - int  -> usato come limite superiore esclusivo (valori fuori range vengono ignorati)
        use_numba: prova ad usare Numba (se installato).
        warmup: se True, compila il kernel Numba in __init__ per evitare il costo alla prima chiamata.
        seed: per riproducibilità (NumPy + Numba).
        """
        self.K = len(lists)

        # --- inferenza universe se non fornito ---
        if universe is None:
            max_val = None
            for arr in lists:
                a = np.asarray(arr)
                if a.size:
                    # accetto solo non-negativi per definire il range
                    a_nonneg = a[a >= 0]
                    if a_nonneg.size:
                        amax = int(a_nonneg.max())
                        max_val = amax if max_val is None else max(max_val, amax)
            if max_val is None:
                raise ValueError("Impossibile inferire 'universe': tutte le liste sono vuote o negative. Passa universe esplicitamente.")
            universe = max_val + 1  # [0..max_val]
        self.universe = int(universe)

        # --- selezione backend ---
        self._numba_enabled = bool(use_numba and NUMBA_AVAILABLE)

        # --- preprocess: K x universe uint8 (0/1), dedup automatico via assegnazione a 1 ---
        bitsets = np.zeros((self.K, self.universe), dtype=np.uint8)
        for i, arr in enumerate(lists):
            a = np.asarray(arr, dtype=np.int64).ravel()
            a = a[(a >= 0) & (a < self.universe)]
            if a.size:
                bitsets[i, np.unique(a)] = 1
        self._bitsets = bitsets

        # RNG e stato
        self._rng = np.random.default_rng(seed)
        if self._numba_enabled:
            _numba_seed(int(seed if seed is not None else np.random.SeedSequence().entropy))
            # workspace riutilizzabili
            self._kept_mask_u8 = np.zeros(self.universe, dtype=np.uint8)
            self._candidates_i32 = np.empty(self.universe, dtype=np.int32)
            if warmup:
                # Compila per le dimensioni reali dell'universo
                dummy_out = np.empty((1, 1), dtype=np.int32)
                _chain_window_numba(self._bitsets[:1], 1,
                                    self._kept_mask_u8, self._candidates_i32, dummy_out)

    def seed(self, seed: int) -> None:
        """Reimposta il seed per NumPy e Numba."""
        self._rng = np.random.default_rng(seed)
        if self._numba_enabled:
            _numba_seed(int(seed))

    def sample_chain_window(self, start: int, k: int, m: int) -> np.ndarray:
        """
        Esegue il sampling sulla finestra contigua [start, start+k-1].
        Ritorna un array (k, m) uint16.
        """
        if not (0 <= start < self.K) or start + k > self.K:
            raise IndexError("Finestra fuori dai limiti: start + k > K.")
        if m <= 0:
            raise ValueError("m deve essere > 0.")

        window = self._bitsets[start:start + k]  # (k, universe)

        if self._numba_enabled:
            out_i32 = np.empty((k, m), dtype=np.int32)
            _chain_window_numba(window, int(m),
                                self._kept_mask_u8, self._candidates_i32, out_i32)
            return out_i32.astype(np.uint16, copy=False)
        else:
            return _chain_window_numpy(window, int(m), self._rng)

    def sample_many_windows(self, starts: Sequence[int], k: int, m: int) -> List[np.ndarray]:
        """Comodo per calcolare più finestre in serie; restituisce una lista di (k, m)."""
        return [self.sample_chain_window(int(s), k, m) for s in starts]

    @property
    def K_lists(self) -> int:
        return self.K

    @property
    def bitsets_view(self) -> np.ndarray:
        """Accesso read-only ai bitset preprocessati (K, universe)."""
        return self._bitsets
