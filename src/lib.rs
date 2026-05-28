#![allow(
    clippy::needless_range_loop,
    clippy::new_without_default,
    clippy::type_complexity,
    dead_code
)]
//! # Info-Geo
//!
//! Information geometry — statistical manifolds, Fisher information, natural gradient, KL divergence.
//!
//! Treats probability distributions as points on a Riemannian manifold where the Fisher information
//! matrix defines the metric tensor. This provides a principled foundation for optimization in
//! probability space.

mod exponential;
mod fisher;
mod manifold;
mod natural;

pub use exponential::ExponentialFamily;
pub use fisher::FisherInformation;
pub use manifold::StatisticalManifold;
pub use natural::NaturalGradient;
