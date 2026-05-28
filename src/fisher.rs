//! Fisher information matrix computation.

/// Fisher information matrix for statistical models.
///
/// F_ij = E[∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j]
#[derive(Debug, Clone)]
pub struct FisherInformation {
    /// The Fisher matrix.
    pub matrix: Vec<Vec<f64>>,
    /// Parameter dimension.
    pub dim: usize,
}

impl FisherInformation {
    /// Compute from score function samples.
    /// Each sample is a gradient of the log-likelihood w.r.t. parameters.
    pub fn from_scores(scores: &[Vec<f64>]) -> Self {
        let n = scores.len();
        if n == 0 {
            return Self {
                matrix: vec![],
                dim: 0,
            };
        }
        let d = scores[0].len();
        let mut fisher = vec![vec![0.0; d]; d];
        for s in scores {
            for i in 0..d {
                for j in 0..d {
                    fisher[i][j] += s[i] * s[j];
                }
            }
        }
        for row in &mut fisher {
            for v in row.iter_mut() {
                *v /= n as f64;
            }
        }
        Self {
            matrix: fisher,
            dim: d,
        }
    }

    /// Fisher information for a Gaussian distribution N(μ, σ²).
    /// F_μμ = 1/σ², F_σσ = 2/σ².
    pub fn gaussian(mu: f64, sigma: f64) -> Self {
        let s2 = sigma * sigma;
        Self {
            matrix: vec![vec![1.0 / s2, 0.0], vec![0.0, 2.0 / s2]],
            dim: 2,
        }
    }

    /// Fisher information for a categorical distribution with probabilities p.
    pub fn categorical(p: &[f64]) -> Self {
        let n = p.len();
        let mut f = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    f[i][j] = 1.0 / p[i].max(1e-10);
                }
            }
        }
        Self { matrix: f, dim: n }
    }

    /// Compute the Cramér-Rao bound: the inverse of the Fisher information.
    /// This gives the minimum variance of any unbiased estimator.
    pub fn cramer_rao_bound(&self) -> Vec<Vec<f64>> {
        invert_matrix(&self.matrix)
    }

    /// Fisher-Rao distance between two parameter values.
    /// Approximate for most distributions.
    pub fn fisher_rao_distance(&self, theta1: &[f64], theta2: &[f64]) -> f64 {
        let diff: Vec<f64> = theta1.iter().zip(theta2).map(|(a, b)| a - b).collect();
        let mut dist_sq = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                dist_sq += diff[i] * self.matrix[i][j] * diff[j];
            }
        }
        dist_sq.max(0.0).sqrt()
    }

    /// Determinant of the Fisher matrix (related to model complexity).
    pub fn determinant(&self) -> f64 {
        // Simple determinant for small matrices
        if self.dim == 1 {
            return self.matrix[0][0];
        }
        if self.dim == 2 {
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0];
        }
        // General: LU-based
        let mut m = self.matrix.clone();
        let n = self.dim;
        let mut det = 1.0;
        for i in 0..n {
            let mut max_val = m[i][i].abs();
            let mut max_row = i;
            for k in (i + 1)..n {
                if m[k][i].abs() > max_val {
                    max_val = m[k][i].abs();
                    max_row = k;
                }
            }
            if max_row != i {
                m.swap(i, max_row);
                det *= -1.0;
            }
            if m[i][i].abs() < 1e-15 {
                return 0.0;
            }
            det *= m[i][i];
            for k in (i + 1)..n {
                let factor = m[k][i] / m[i][i];
                for j in (i + 1)..n {
                    m[k][j] -= factor * m[i][j];
                }
            }
        }
        det
    }
}

/// Simple matrix inversion via Gauss-Jordan elimination.
fn invert_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = matrix[i][j];
        }
        aug[i][n + i] = 1.0;
    }
    for col in 0..n {
        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            continue;
        }
        for j in 0..2 * n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..2 * n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    (0..n).map(|i| aug[i][n..2 * n].to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_scores_identity() {
        let scores = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let f = FisherInformation::from_scores(&scores);
        assert!((f.matrix[0][0] - 0.5).abs() < 1e-10);
        assert!((f.matrix[0][1]).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_fisher() {
        let f = FisherInformation::gaussian(0.0, 1.0);
        assert!((f.matrix[0][0] - 1.0).abs() < 1e-10);
        assert!((f.matrix[1][1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cramer_rao_bound() {
        let f = FisherInformation::gaussian(0.0, 1.0);
        let cr = f.cramer_rao_bound();
        assert!(cr[0][0] > 0.0);
    }

    #[test]
    fn test_fisher_rao_distance() {
        let f = FisherInformation::gaussian(0.0, 1.0);
        let d = f.fisher_rao_distance(&[0.0, 1.0], &[1.0, 1.0]);
        assert!(d > 0.0);
    }

    #[test]
    fn test_determinant() {
        let f = FisherInformation::gaussian(0.0, 1.0);
        let det = f.determinant();
        assert!((det - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_categorical_fisher() {
        let f = FisherInformation::categorical(&[0.5, 0.5]);
        assert!((f.matrix[0][0] - 2.0).abs() < 1e-10);
    }
}
