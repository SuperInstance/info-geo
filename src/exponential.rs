//! Exponential family distributions on statistical manifolds.

/// An exponential family distribution: p(x|θ) = h(x) exp(η(θ)·T(x) - A(η))
pub struct ExponentialFamily {
    pub name: String,
    pub natural_params: Vec<f64>,
    pub sufficient_stats: Vec<f64>,
}

impl ExponentialFamily {
    /// Gaussian as exponential family.
    pub fn gaussian(mu: f64, sigma: f64) -> Self {
        Self {
            name: "Gaussian".into(),
            natural_params: vec![mu / (sigma * sigma), -1.0 / (2.0 * sigma * sigma)],
            sufficient_stats: vec![mu, mu * mu + sigma * sigma],
        }
    }

    /// Bernoulli as exponential family.
    pub fn bernoulli(p: f64) -> Self {
        let p = p.clamp(1e-10, 1.0 - 1e-10);
        Self {
            name: "Bernoulli".into(),
            natural_params: vec![(p / (1.0 - p)).ln()],
            sufficient_stats: vec![p],
        }
    }

    /// Poisson as exponential family.
    pub fn poisson(lambda: f64) -> Self {
        Self {
            name: "Poisson".into(),
            natural_params: vec![lambda.ln()],
            sufficient_stats: vec![lambda],
        }
    }

    /// Log-partition function A(η).
    pub fn log_partition(&self) -> f64 {
        match self.name.as_str() {
            "Gaussian" => {
                let eta1 = self.natural_params[0];
                let eta2 = self.natural_params[1];
                -eta1 * eta1 / (4.0 * eta2) - 0.5 * (-2.0 * eta2).ln() / 2.0
            }
            "Bernoulli" => (1.0 + self.natural_params[0].exp()).ln(),
            "Poisson" => self.natural_params[0].exp(),
            _ => 0.0,
        }
    }

    /// KL divergence between two members of the same exponential family.
    pub fn kl_divergence(&self, other: &Self) -> f64 {
        let d_eta: Vec<f64> = self
            .natural_params
            .iter()
            .zip(&other.natural_params)
            .map(|(a, b)| a - b)
            .collect();
        let dot: f64 = d_eta
            .iter()
            .zip(&self.sufficient_stats)
            .map(|(d, t)| d * t)
            .sum();
        let kl = other.log_partition() - self.log_partition() + dot;
        kl.max(0.0)
    }

    /// Bregman divergence (generalizes KL for exponential families).
    pub fn bregman_divergence(&self, other: &Self) -> f64 {
        self.kl_divergence(other)
    }

    /// Maximum entropy distribution matching given moment constraints.
    pub fn max_entropy(mean: f64, variance: f64) -> Self {
        Self::gaussian(mean, variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_params() {
        let g = ExponentialFamily::gaussian(0.0, 1.0);
        assert_eq!(g.name, "Gaussian");
        assert_eq!(g.natural_params.len(), 2);
    }

    #[test]
    fn test_bernoulli_params() {
        let b = ExponentialFamily::bernoulli(0.7);
        assert!(b.natural_params[0] > 0.0); // log(0.7/0.3) > 0
    }

    #[test]
    fn test_kl_same_distribution() {
        let g1 = ExponentialFamily::gaussian(0.0, 1.0);
        let g2 = ExponentialFamily::gaussian(0.0, 1.0);
        let kl = g1.kl_divergence(&g2);
        assert!(kl < 0.1);
    }

    #[test]
    fn test_kl_different_means() {
        let g1 = ExponentialFamily::gaussian(0.0, 1.0);
        let g2 = ExponentialFamily::gaussian(5.0, 1.0);
        let kl = g1.kl_divergence(&g2);
        assert!(kl > 1.0);
    }

    #[test]
    fn test_max_entropy() {
        let me = ExponentialFamily::max_entropy(0.0, 1.0);
        assert_eq!(me.name, "Gaussian");
    }

    #[test]
    fn test_log_partition() {
        let g = ExponentialFamily::gaussian(0.0, 1.0);
        let lp = g.log_partition();
        assert!(lp.is_finite());
    }
}
