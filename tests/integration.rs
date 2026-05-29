use info_geo::*;

#[test]
fn test_gaussian_log_partition() {
    let g = ExponentialFamily::gaussian(0.0, 1.0);
    let lp = g.log_partition();
    assert!(lp.is_finite());
}

#[test]
fn test_gaussian_kl_symmetric_offset() {
    let g1 = ExponentialFamily::gaussian(0.0, 1.0);
    let g2 = ExponentialFamily::gaussian(1.0, 1.0);
    let kl12 = g1.kl_divergence(&g2);
    let kl21 = g2.kl_divergence(&g1);
    // KL is not symmetric in general, but for Gaussians with same variance: KL(p||q) = KL(q||p)
    assert!((kl12 - kl21).abs() < 0.1, "same-variance Gaussian KL should be ~symmetric: {kl12} vs {kl21}");
}

#[test]
fn test_kl_same_distribution_near_zero() {
    let g1 = ExponentialFamily::gaussian(2.5, 1.5);
    let g2 = ExponentialFamily::gaussian(2.5, 1.5);
    let kl = g1.kl_divergence(&g2);
    assert!(kl < 0.01, "KL(p||p) should be ~0, got {kl}");
}

#[test]
fn test_kl_different_distributions_positive() {
    let g1 = ExponentialFamily::gaussian(0.0, 1.0);
    let g2 = ExponentialFamily::gaussian(10.0, 1.0);
    let kl = g1.kl_divergence(&g2);
    assert!(kl > 0.0, "KL of different distributions should be > 0");
}

#[test]
fn test_bernoulli_creation() {
    let b = ExponentialFamily::bernoulli(0.5);
    assert_eq!(b.name, "Bernoulli");
    assert_eq!(b.natural_params.len(), 1);
}

#[test]
fn test_poisson_creation() {
    let p = ExponentialFamily::poisson(5.0);
    assert_eq!(p.name, "Poisson");
    assert_eq!(p.natural_params.len(), 1);
}

#[test]
fn test_max_entropy_is_gaussian() {
    let g = ExponentialFamily::max_entropy(0.0, 1.0);
    assert_eq!(g.name, "Gaussian");
}

#[test]
fn test_bregman_matches_kl() {
    let g1 = ExponentialFamily::gaussian(0.0, 1.0);
    let g2 = ExponentialFamily::gaussian(3.0, 2.0);
    let kl = g1.kl_divergence(&g2);
    let bd = g1.bregman_divergence(&g2);
    assert!((kl - bd).abs() < 1e-10, "Bregman should match KL for exponential families");
}
