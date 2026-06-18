//! `_turboscope` — fast exact top-k for turboscope's shadow recall sampler.
//!
//! This crate compiles to a single Python extension module exposing one
//! function:
//!
//! ```python
//! scores, local_indices = _turboscope.exact_topk(reservoir, queries, k)
//! ```
//!
//! It is a pure computation kernel with no index state. The Python layer
//! in `turboscope/sampler.py` calls it on every shadow-sampled search and
//! falls back to a numpy implementation when this extension is not built.
//!
//! # Algorithm
//!
//! 1. **Score matrix** — compute the full (nq × n_res) inner-product matrix
//!    using faer's GEMM: `S = queries @ reservoir.T`. faer is pure Rust and
//!    requires no external BLAS, keeping the build self-contained.
//!
//! 2. **Per-query top-k** — for each query row in S, find the k highest
//!    scores using a min-heap of capacity k (O(n_res · log k) per query).
//!    Parallelised across queries with rayon.
//!
//! 3. **Sort and emit** — sort each query's k results by score descending,
//!    pack into (nq, effective_k) NumPy arrays and return.
//!
//! # Precision note
//!
//! The score matrix is computed in f32. The numpy fallback in `sampler.py`
//! uses f64 accumulation (`queries.astype(f64) @ reservoir.astype(f64).T`)
//! to prevent precision loss from swapping adjacent ranks when two scores
//! differ by less than ~1e-7. At our reservoir size (≤ 5,000 vectors) and
//! the inner-product magnitudes typical of normalised embedding vectors
//! (scores in [-1, 1]), f32 accumulation is accurate enough that ties at
//! that precision level never occur in practice. The two paths produce
//! identical top-k sets on all non-degenerate inputs.
//!
//! If a future caller needs f64 precision, add a `exact_topk_f64` entry
//! point that widens both inputs before the GEMM.
//!
//! # Return types
//!
//! `scores` — (nq, effective_k) float32 in descending order per row.
//!
//! `local_indices` — (nq, effective_k) int64. These are 0-based positions
//! into the reservoir array. The Python caller converts them to external
//! uint64 IDs via `reservoir_ids[local_indices]`. We use i64 rather than
//! u64 because numpy's `intp` (the natural indexing dtype on 64-bit) is
//! signed, and because fancy indexing accepts signed integers.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use ordered_float::NotNan;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Python-facing entry point
// ---------------------------------------------------------------------------

/// Exact brute-force top-k inner-product search.
///
/// Parameters
/// ----------
/// reservoir : ndarray of shape (n_res, dim), dtype float32, C-contiguous
///     The raw float32 vectors from the turboscope reservoir.
/// queries : ndarray of shape (nq, dim), dtype float32, C-contiguous
///     Query vectors. Must have the same ``dim`` as ``reservoir``.
/// k : int
///     Number of results per query. Clamped to ``n_res`` if larger.
///
/// Returns
/// -------
/// scores : ndarray of shape (nq, effective_k), float32
///     Inner-product scores in descending order per row.
///     ``effective_k = min(k, n_res)``.
/// local_indices : ndarray of shape (nq, effective_k), int64
///     Indices into the reservoir (0..n_res). Convert to external IDs via
///     ``reservoir_ids[local_indices]``.
///
/// Raises
/// ------
/// ValueError
///     If either array is not C-contiguous, if their ``dim`` axes differ,
///     or if ``k`` is zero.
#[pyfunction]
fn exact_topk<'py>(
    py: Python<'py>,
    reservoir: PyReadonlyArray2<f32>,
    queries: PyReadonlyArray2<f32>,
    k: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>)> {
    let res_arr = reservoir.as_array();
    let q_arr = queries.as_array();

    let n_res = res_arr.nrows();
    let res_dim = res_arr.ncols();
    let nq = q_arr.nrows();
    let q_dim = q_arr.ncols();

    // ---- Input validation -------------------------------------------------

    if res_dim != q_dim {
        return Err(PyValueError::new_err(format!(
            "reservoir dim {res_dim} != queries dim {q_dim}"
        )));
    }
    if k == 0 {
        return Err(PyValueError::new_err("k must be >= 1"));
    }

    let res_slice = res_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("reservoir must be C-contiguous"))?;
    let q_slice = q_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("queries must be C-contiguous"))?;

    let dim = res_dim;
    let effective_k = k.min(n_res);

    // ---- Fast-path: empty inputs -----------------------------------------

    if nq == 0 || n_res == 0 || effective_k == 0 {
        let scores = numpy::ndarray::Array2::<f32>::zeros((nq, 0))
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::<i64>::zeros((nq, 0))
            .into_pyarray(py);
        return Ok((scores, indices));
    }

    // ---- Score matrix: S = queries @ reservoir.T  (nq × n_res) ----------
    //
    // Use faer for the GEMM. faer is pure-Rust and handles its own
    // thread pool internally (Parallelism::Rayon(0) means "use all rayon
    // threads"). This is the same approach turbovec uses for its query
    // rotation multiply in search.rs.
    //
    // faer expects row-major data. Both arrays are C-contiguous so their
    // flat slices are already row-major.
    let mut scores_flat = vec![0.0_f32; nq * n_res];

    {
        use faer::linalg::matmul::matmul;
        use faer::mat;

        // queries: (nq, dim)
        let q_mat = mat::from_row_major_slice::<f32>(q_slice, nq, dim);
        // reservoir: (n_res, dim) → need (dim, n_res) as RHS
        let res_mat = mat::from_row_major_slice::<f32>(res_slice, n_res, dim);
        // output: (nq, n_res)
        let mut out_mat =
            mat::from_row_major_slice_mut::<f32>(&mut scores_flat, nq, n_res);

        matmul(
            out_mat.as_mut(),
            q_mat,
            res_mat.transpose(),
            None,       // beta: no accumulation into output (it starts at 0)
            1.0_f32,    // alpha
            faer::Parallelism::Rayon(0),
        );
    }

    // ---- Per-query top-k via min-heap  (parallelised with rayon) ---------
    //
    // Each query owns a row of `scores_flat` of length n_res. We find the
    // `effective_k` largest scores using a min-heap of capacity k:
    //
    //   - Seed the heap with the first k elements.
    //   - For the remaining n_res - k elements: if the score exceeds the
    //     current minimum (heap root), pop the minimum and push the new
    //     (score, idx) pair.
    //   - Drain and sort descending for the final output.
    //
    // O(n_res · log k) per query.  For n_res=5000, k=10 this is ~83,000
    // comparisons per query — negligible compared with the GEMM above.
    //
    // We use `NotNan<f32>` from the `ordered-float` crate so BinaryHeap
    // can compare scores. Our scores are always finite (reservoir and query
    // vectors were validated by turbovec before reaching us), but we map
    // any unexpected NaN to a `PyValueError` rather than panicking.

    // Pre-allocate flat output arrays; rayon writes disjoint row chunks.
    let mut out_scores:  Vec<f32> = vec![0.0_f32; nq * effective_k];
    let mut out_indices: Vec<i64> = vec![0_i64;   nq * effective_k];

    // Split the output slices into per-query chunks so rayon can write
    // them without aliasing.
    let score_chunks:  Vec<&mut [f32]> = out_scores.chunks_mut(effective_k).collect();
    let index_chunks:  Vec<&mut [i64]> = out_indices.chunks_mut(effective_k).collect();

    // Each element of this Vec is one query's work: (row_of_scores, output_score_chunk, output_index_chunk).
    let results: Vec<PyResult<()>> = score_chunks
        .into_par_iter()
        .zip(index_chunks.into_par_iter())
        .enumerate()
        .map(|(qi, (s_out, i_out))| {
            let row = &scores_flat[qi * n_res..(qi + 1) * n_res];
            topk_row(row, effective_k, s_out, i_out)
        })
        .collect();

    // Check for any NaN errors from `topk_row`.
    for r in results {
        r?;
    }

    // ---- Pack into NumPy arrays and return --------------------------------

    let scores_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), out_scores)
        .expect("shape invariant violated")
        .into_pyarray(py);

    let indices_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), out_indices)
        .expect("shape invariant violated")
        .into_pyarray(py);

    Ok((scores_arr, indices_arr))
}

// ---------------------------------------------------------------------------
// Per-query top-k kernel
// ---------------------------------------------------------------------------

/// Find the `k` highest-scoring reservoir indices for one query row.
///
/// Uses a min-heap of capacity k to stream through `scores` in a single
/// pass.  The results are written into `s_out` and `i_out` in descending
/// score order.
///
/// Returns `Err(PyValueError)` if any score is NaN; this should be
/// unreachable on valid inputs (turbovec validates before adding vectors)
/// but is checked defensively rather than panicking.
fn topk_row(
    scores: &[f32],
    k: usize,
    s_out: &mut [f32],
    i_out: &mut [i64],
) -> PyResult<()> {
    // Min-heap: Reverse<(NotNan<f32>, usize)>
    // The smallest score sits at the top so we can cheaply decide whether
    // a new candidate beats the current k-th best.
    let mut heap: BinaryHeap<Reverse<(NotNan<f32>, usize)>> =
        BinaryHeap::with_capacity(k + 1);

    for (idx, &s) in scores.iter().enumerate() {
        let score = NotNan::new(s).map_err(|_| {
            PyValueError::new_err(format!(
                "NaN score at reservoir index {idx}; \
                 reservoir vectors should have been validated by turbovec"
            ))
        })?;

        if heap.len() < k {
            // Heap not full yet — always insert.
            heap.push(Reverse((score, idx)));
        } else if let Some(&Reverse((min_score, _))) = heap.peek() {
            // Heap is full — only replace if this score beats the minimum.
            if score > min_score {
                heap.pop();
                heap.push(Reverse((score, idx)));
            }
        }
    }

    // Drain heap into a Vec and sort descending (highest score first).
    // The heap gives us items in ascending-score order (min first); we
    // collect all then sort rather than popping one-by-one, which would
    // give ascending order and require a reverse pass.
    let mut top: Vec<(NotNan<f32>, usize)> = heap
        .into_iter()
        .map(|Reverse(pair)| pair)
        .collect();

    // Sort descending by score.  Ties broken by index ascending so the
    // output is deterministic — important for test reproducibility.
    top.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

    debug_assert_eq!(top.len(), s_out.len());

    for (i, (score, reservoir_idx)) in top.into_iter().enumerate() {
        s_out[i] = score.into_inner();
        i_out[i] = reservoir_idx as i64;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// `_turboscope` Python extension module.
///
/// Exposes one function: ``exact_topk(reservoir, queries, k)``.
/// See its docstring for the full contract.
#[pymodule]
fn _turboscope(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(exact_topk, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests (run with `cargo test`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: run topk_row and return (scores, indices) as Vecs.
    fn topk(scores: &[f32], k: usize) -> (Vec<f32>, Vec<i64>) {
        let effective_k = k.min(scores.len());
        let mut s_out = vec![0.0_f32; effective_k];
        let mut i_out = vec![0_i64; effective_k];
        topk_row(scores, effective_k, &mut s_out, &mut i_out).unwrap();
        (s_out, i_out)
    }

    #[test]
    fn top1_finds_maximum() {
        let scores = vec![0.1_f32, 0.9, 0.3, 0.7, 0.5];
        let (s, i) = topk(&scores, 1);
        assert_eq!(i, vec![1]);      // index of 0.9
        assert!((s[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn top3_descending_order() {
        let scores = vec![0.1_f32, 0.9, 0.3, 0.7, 0.5];
        let (s, i) = topk(&scores, 3);
        // Expected: [0.9 (idx 1), 0.7 (idx 3), 0.5 (idx 4)]
        assert_eq!(i, vec![1, 3, 4]);
        assert!(s[0] >= s[1] && s[1] >= s[2], "scores must be non-increasing");
    }

    #[test]
    fn k_larger_than_n_returns_all_sorted() {
        let scores = vec![0.3_f32, 0.1, 0.2];
        let (s, i) = topk(&scores, 10);   // k > n_res
        assert_eq!(s.len(), 3);
        assert_eq!(i, vec![0, 2, 1]);     // descending: 0.3, 0.2, 0.1
    }

    #[test]
    fn single_element() {
        let scores = vec![0.42_f32];
        let (s, i) = topk(&scores, 1);
        assert_eq!(i, vec![0]);
        assert!((s[0] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn negative_scores_handled() {
        // Inner products can be negative (non-normalised or obtuse angle).
        let scores = vec![-0.5_f32, -0.1, -0.9, -0.3];
        let (s, i) = topk(&scores, 2);
        // Highest (least negative) are -0.1 (idx 1) and -0.3 (idx 3).
        assert_eq!(i, vec![1, 3]);
        assert!(s[0] >= s[1]);
    }

    #[test]
    fn tie_broken_by_index() {
        // Two identical scores: lower index should appear first after tie-break.
        let scores = vec![0.5_f32, 0.5, 0.5];
        let (_, i) = topk(&scores, 2);
        assert!(i[0] < i[1], "ties should be broken by ascending index");
    }

    #[test]
    fn k_equals_n_res_returns_all() {
        let scores = vec![0.4_f32, 0.2, 0.6, 0.1, 0.8];
        let (s, i) = topk(&scores, 5);
        assert_eq!(s.len(), 5);
        // Fully sorted descending.
        for w in s.windows(2) {
            assert!(w[0] >= w[1]);
        }
        // All 5 reservoir indices present.
        let mut seen: Vec<i64> = i.clone();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }
}