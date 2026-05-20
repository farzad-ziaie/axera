/// axera_core — Rust-accelerated primitives for Axera.
///
/// Exports:
///   combinations_all(n)         -> list[list[int]]
///   combinations_k(n, k)        -> list[list[int]]
///   grid_find_index(cost, ubs)  -> int
///   pareto_dominates(a, b)      -> bool
///   pareto_fast(costs)          -> list[bool]   (True = non-dominated)
use pyo3::prelude::*;
use rayon::prelude::*;

// ── combinatorics ────────────────────────────────────────────────────────────

/// All non-empty subsets of 0..n as sorted index lists.
/// Equivalent to itertools.combinations for L in 1..=n.
#[pyfunction]
fn combinations_all(n: usize) -> Vec<Vec<usize>> {
    let total = (1usize << n).saturating_sub(1);
    let mut result: Vec<Vec<usize>> = Vec::with_capacity(total);
    for mask in 1usize..(1 << n) {
        let mut subset = Vec::with_capacity(mask.count_ones() as usize);
        for bit in 0..n {
            if mask & (1 << bit) != 0 {
                subset.push(bit);
            }
        }
        result.push(subset);
    }
    result
}

/// All k-combinations of 0..n.
#[pyfunction]
fn combinations_k(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 || k > n {
        return vec![];
    }
    let mut result = Vec::new();
    let mut indices: Vec<usize> = (0..k).collect();
    loop {
        result.push(indices.clone());
        // find rightmost index that can be incremented
        let mut i = k;
        loop {
            if i == 0 {
                return result;
            }
            i -= 1;
            if indices[i] != i + n - k {
                break;
            }
        }
        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

// ── MOPSO grid ───────────────────────────────────────────────────────────────

/// Find which grid cell a scalar cost value falls into given upper-bound
/// breakpoints for one objective dimension.
/// Returns the first index j such that cost < upper_bounds[j].
#[pyfunction]
fn grid_compare(cost: f64, upper_bounds: Vec<f64>) -> usize {
    for (j, &ub) in upper_bounds.iter().enumerate() {
        if cost < ub {
            return j;
        }
    }
    upper_bounds.len()
}

/// Compute a single integer grid index for a particle whose cost vector
/// lives in an n_obj-dimensional grid.
///
/// `cost`        : Vec<f64> of length n_obj
/// `upper_bounds`: Vec<Vec<f64>> — for each objective, the sorted breakpoints
/// `n_grid`      : number of grid cells per dimension
///
/// Returns the flat integer grid index used for leader selection.
#[pyfunction]
fn grid_find_index(cost: Vec<f64>, upper_bounds: Vec<Vec<f64>>, n_grid: usize) -> u64 {
    let n_obj = cost.len();
    assert_eq!(upper_bounds.len(), n_obj, "cost and upper_bounds length mismatch");
    let sub: Vec<u64> = cost
        .iter()
        .zip(upper_bounds.iter())
        .map(|(&c, ubs)| grid_compare(c, ubs.clone()) as u64)
        .collect();
    // Bijective mapping: index = sub[0] * (G+2)^(n-1) + sub[1] * (G+2)^(n-2) + ...
    let base = (n_grid as u64) + 2;
    sub.iter().fold(0u64, |acc, &s| acc * base + s)
}

// ── Pareto dominance ──────────────────────────────────────────────────────────

/// True if cost vector `a` weakly dominates `b`:
///   all a[i] <= b[i]  AND  exists i: a[i] < b[i]
#[pyfunction]
fn pareto_dominates(a: Vec<f64>, b: Vec<f64>) -> bool {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    let all_le = a.iter().zip(b.iter()).all(|(&ai, &bi)| ai <= bi);
    let any_lt = a.iter().zip(b.iter()).any(|(&ai, &bi)| ai < bi);
    all_le && any_lt
}

/// Determine non-dominated front for a matrix of cost vectors (one per row).
/// Returns a bool vec: True = particle is non-dominated.
/// Runs pairwise comparisons in parallel via Rayon.
#[pyfunction]
fn pareto_fast(costs: Vec<Vec<f64>>) -> Vec<bool> {
    let n = costs.len();
    let dominated: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| {
            (0..n).any(|j| {
                if i == j {
                    return false;
                }
                pareto_dominates(costs[j].clone(), costs[i].clone())
            })
        })
        .collect();
    dominated.iter().map(|&d| !d).collect()
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
fn _core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(combinations_all, m)?)?;
    m.add_function(wrap_pyfunction!(combinations_k, m)?)?;
    m.add_function(wrap_pyfunction!(grid_compare, m)?)?;
    m.add_function(wrap_pyfunction!(grid_find_index, m)?)?;
    m.add_function(wrap_pyfunction!(pareto_dominates, m)?)?;
    m.add_function(wrap_pyfunction!(pareto_fast, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
