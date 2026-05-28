//! Statistical manifold operations.

/// A statistical manifold with its metric tensor.
pub struct StatisticalManifold {
    pub dim: usize,
    pub metric: Vec<Vec<f64>>,
    pub christoffel: Vec<Vec<Vec<f64>>>,
}

impl StatisticalManifold {
    /// Create a flat manifold (Euclidean metric).
    pub fn euclidean(dim: usize) -> Self {
        let mut metric = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            metric[i][i] = 1.0;
        }
        Self {
            dim,
            metric,
            christoffel: vec![vec![vec![0.0; dim]; dim]; dim],
        }
    }

    /// Create from a Fisher information matrix (the metric tensor).
    pub fn from_fisher(fisher: &[Vec<f64>]) -> Self {
        let dim = fisher.len();
        Self {
            dim,
            metric: fisher.to_vec(),
            christoffel: vec![vec![vec![0.0; dim]; dim]; dim],
        }
    }

    /// Geodesic distance between two points.
    pub fn geodesic_distance(&self, p1: &[f64], p2: &[f64]) -> f64 {
        let diff: Vec<f64> = p1.iter().zip(p2).map(|(a, b)| a - b).collect();
        let mut dist_sq = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                dist_sq += diff[i] * self.metric[i][j] * diff[j];
            }
        }
        dist_sq.max(0.0).sqrt()
    }

    /// Volume element: sqrt(det(g)).
    pub fn volume_element(&self) -> f64 {
        let det = compute_determinant(&self.metric);
        det.max(0.0).sqrt()
    }

    /// Parallel transport of a vector along a geodesic.
    /// For flat manifolds, this is the identity.
    pub fn parallel_transport(&self, v: &[f64], _from: &[f64], _to: &[f64]) -> Vec<f64> {
        v.to_vec() // Flat approximation
    }

    /// Exponential map: maps a tangent vector to a point on the manifold.
    pub fn exp_map(&self, base: &[f64], tangent: &[f64]) -> Vec<f64> {
        base.iter().zip(tangent).map(|(b, t)| b + t).collect()
    }

    /// Logarithmic map: maps a point back to a tangent vector.
    pub fn log_map(&self, base: &[f64], point: &[f64]) -> Vec<f64> {
        point.iter().zip(base).map(|(p, b)| p - b).collect()
    }

    /// Riemannian mean (Fréchet mean) of a set of points.
    pub fn riemannian_mean(&self, points: &[Vec<f64>], iterations: usize) -> Vec<f64> {
        if points.is_empty() {
            return vec![0.0; self.dim];
        }
        // Start with Euclidean mean
        let mut mean = vec![0.0; self.dim];
        for p in points {
            for i in 0..self.dim {
                mean[i] += p[i];
            }
        }
        for v in mean.iter_mut() {
            *v /= points.len() as f64;
        }

        // Iterate: mean = exp(1/n * Σ log(mean, p_i))
        for _ in 0..iterations {
            let mut tangent_sum = vec![0.0; self.dim];
            for p in points {
                let log = self.log_map(&mean, p);
                for i in 0..self.dim {
                    tangent_sum[i] += log[i];
                }
            }
            for v in tangent_sum.iter_mut() {
                *v /= points.len() as f64;
            }
            mean = self.exp_map(&mean, &tangent_sum);
        }
        mean
    }
}

fn compute_determinant(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return m[0][0];
    }
    if n == 2 {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }
    let mut a = m.to_vec();
    let mut det = 1.0;
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[max_row][col].abs() {
                max_row = row;
            }
        }
        if max_row != col {
            a.swap(col, max_row);
            det *= -1.0;
        }
        if a[col][col].abs() < 1e-15 {
            return 0.0;
        }
        det *= a[col][col];
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            for j in (col + 1)..n {
                a[row][j] -= f * a[col][j];
            }
        }
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_manifold() {
        let m = StatisticalManifold::euclidean(3);
        assert_eq!(m.dim, 3);
        assert_eq!(m.metric[0][0], 1.0);
    }

    #[test]
    fn test_geodesic_distance() {
        let m = StatisticalManifold::euclidean(2);
        let d = m.geodesic_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_element() {
        let m = StatisticalManifold::euclidean(2);
        let v = m.volume_element();
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_log_map() {
        let m = StatisticalManifold::euclidean(2);
        let base = vec![1.0, 2.0];
        let point = vec![4.0, 6.0];
        let log = m.log_map(&base, &point);
        assert!((log[0] - 3.0).abs() < 1e-10);
        let exp = m.exp_map(&base, &log);
        assert!((exp[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_riemannian_mean() {
        let m = StatisticalManifold::euclidean(2);
        let points = vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![0.0, 2.0]];
        let mean = m.riemannian_mean(&points, 10);
        assert!((mean[0] - 2.0 / 3.0).abs() < 0.1);
    }
}
