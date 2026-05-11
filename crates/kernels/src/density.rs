pub trait LogDensity {
    type Point: ?Sized;
    /// .
    fn log_prob<'a>(&'a self, x: &'a Self::Point) -> f64;
}

pub trait GradLogDensity: LogDensity {
    type Gradient: ?Sized;
    fn grad_log_prob<'a>(&'a self, x: &'a Self::Point, grad: &mut Self::Gradient);
}

pub trait FusedLogDensity: GradLogDensity {
    fn log_prob_and_grad<'a>(&'a self, x: &'a Self::Point, grad: &mut Self::Gradient) -> f64;
}

pub trait HessianLogDensity: GradLogDensity {
    type Hessian;
    fn hessian(&self, x: &Self::Point, h: &mut Self::Hessian);
}
