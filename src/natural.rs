//! Natural gradient descent on statistical manifolds.

use crate::fisher::FisherInformation;
use crate::manifold::StatisticalManifold;

/// Natural gradient optimizer using Fisher information as the Riemannian metric.
pub struct NaturalGradient {
    pub learning_rate: f64,
    pub damping: f64,
}

impl NaturalGradient {
    pub fn new(lr: f64) -> Self {
        Self {
            learning_rate: lr,
            damping: 1e-3,
        }
    }

    /// Compute the natural gradient: F^{-1} ∇L.
    pub fn compute(&self, fisher: &FisherInformation, gradient: &[f64]) -> Vec<f64> {
        let n = gradient.len();
        if n == 0 {
            return vec![];
        }
        // Damped inverse: (F + λI)^{-1} g
        let mut f_damped = fisher.matrix.clone();
        for i in 0..n {
            f_damped[i][i] += self.damping;
        }
        let inv = invert(&f_damped);
        let mut result = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                result[i] += inv[i][j] * gradient[j];
            }
        }
        result
    }

    /// One optimization step on a statistical manifold.
    pub fn step(
        &self,
        manifold: &StatisticalManifold,
        params: &[f64],
        gradient: &[f64],
    ) -> Vec<f64> {
        let fisher = FisherInformation {
            matrix: manifold.metric.clone(),
            dim: manifold.dim,
        };
        let ng = self.compute(&fisher, gradient);
        manifold.exp_map(
            params,
            &ng.iter()
                .map(|g| -self.learning_rate * g)
                .collect::<Vec<_>>(),
        )
    }

    /// Adam-like adaptive natural gradient.
    pub fn adaptive_step(
        &self,
        params: &[f64],
        gradient: &[f64],
        fisher: &FisherInformation,
        moment1: &mut Vec<f64>,
        moment2: &mut Vec<f64>,
        t: usize,
        beta1: f64,
        beta2: f64,
    ) -> Vec<f64> {
        let ng = self.compute(fisher, gradient);
        let n = params.len();
        for i in 0..n {
            moment1[i] = beta1 * moment1[i] + (1.0 - beta1) * ng[i];
            moment2[i] = beta2 * moment2[i] + (1.0 - beta2) * ng[i] * ng[i];
        }
        let bias_corr1 = 1.0 - beta1.powi(t as i32);
        let bias_corr2 = 1.0 - beta2.powi(t as i32);
        params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let m_hat = moment1[i] / bias_corr1;
                let v_hat = moment2[i] / bias_corr2;
                p - self.learning_rate * m_hat / (v_hat.sqrt() + 1e-8)
            })
            .collect()
    }
}

fn invert(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i][j];
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
            let f = aug[row][col];
            for j in 0..2 * n {
                aug[row][j] -= f * aug[col][j];
            }
        }
    }
    (0..n).map(|i| aug[i][n..2 * n].to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_gradient_identity() {
        let ng = NaturalGradient::new(0.01);
        let fisher = FisherInformation {
            matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            dim: 2,
        };
        let g = vec![1.0, 2.0];
        let result = ng.compute(&fisher, &g);
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_step() {
        let ng = NaturalGradient::new(0.1);
        let m = StatisticalManifold::euclidean(2);
        let p = vec![5.0, 3.0];
        let g = vec![1.0, 1.0];
        let new_p = ng.step(&m, &p, &g);
        assert!(new_p[0] < p[0]);
    }

    #[test]
    fn test_adaptive_step() {
        let ng = NaturalGradient::new(0.01);
        let fisher = FisherInformation {
            matrix: vec![vec![1.0]],
            dim: 1,
        };
        let mut m1 = vec![0.0];
        let mut m2 = vec![0.0];
        let p = vec![10.0];
        let g = vec![1.0];
        let new_p = ng.adaptive_step(&p, &g, &fisher, &mut m1, &mut m2, 1, 0.9, 0.999);
        assert!(new_p[0] < p[0]);
    }
}
