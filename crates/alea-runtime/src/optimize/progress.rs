//! Best-effort terminal reporting for an explicitly opted-in fit.
use super::{MaximizeError, OptimizationResult, OptimizationTermination};
use crate::output::progress::{Terminal, bar};
use std::{
    error::Error,
    io::{self, Write},
};

pub(super) struct FitProgress<W: Write> {
    terminal: Terminal<W>,
    begun: bool,
}

impl FitProgress<io::Stderr> {
    pub(super) fn stderr() -> Self {
        Self {
            terminal: Terminal::stderr(),
            begun: false,
        }
    }
}

impl<W: Write> FitProgress<W> {
    pub(super) fn update(&mut self, result: &OptimizationResult) {
        if !self.begun {
            self.begun = true;
            self.terminal.heading(
                "Fitting model",
                &format!(
                    "L-BFGS · up to {} steps, {} evaluations",
                    result.options.iteration_limit(),
                    result.options.evaluation_limit()
                ),
            );
        }
        if !self.terminal.ready() {
            return;
        }
        let steps = bar(
            result.iterations as u128,
            result.options.iteration_limit() as u128,
        );
        let evaluations = bar(
            result.evaluations as u128,
            result.options.evaluation_limit() as u128,
        );
        let best = result
            .candidate
            .as_ref()
            .map_or_else(|| "—".to_owned(), |p| format!("{:.4e}", p.value));
        let (style, reset) = self.terminal.style("\x1b[36m");
        self.terminal.render(&format!(
            "{style}Steps {steps}{reset} {}  {style}Evals {evaluations}{reset} {}  Best {best}",
            result.iterations, result.evaluations,
        ));
    }

    pub(super) fn finish<E: Error + 'static>(
        &mut self,
        outcome: &Result<OptimizationResult, MaximizeError<E>>,
    ) {
        let result = match outcome {
            Ok(result) => Some(result),
            Err(MaximizeError::Evaluation(error)) => Some(error.result.as_ref()),
            Err(MaximizeError::Invalid(_)) => None,
        };
        let mut detail = String::new();
        let status = if let Some(result) = result {
            use std::fmt::Write as _;
            let _ = write!(
                detail,
                "  {} steps · {} evaluations",
                result.iterations, result.evaluations
            );
            if let Some(candidate) = &result.candidate {
                let _ = write!(detail, " · best {:.6e}", candidate.value);
            }
            match result.termination {
                OptimizationTermination::ScoreTolerance => "Score tolerance reached",
                OptimizationTermination::IterationLimit => "Step limit reached",
                OptimizationTermination::EvaluationLimit => "Evaluation limit reached",
                OptimizationTermination::Cancelled => "Cancelled",
                OptimizationTermination::LineSearchFailed => "Line search failed",
                OptimizationTermination::NumericalDirection => "Numerical direction failed",
                OptimizationTermination::EvaluationFailed => "Evaluation failed",
            }
        } else {
            "Invalid fit configuration"
        };
        self.terminal.finish(status, &detail);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::{
        LbfgsOptions, ObjectiveDescription, ObjectiveKind, OptimizationCandidate,
    };
    use std::convert::Infallible;

    fn result() -> OptimizationResult {
        OptimizationResult {
            dimension: 1,
            options: LbfgsOptions::new(8, 16).unwrap(),
            objective: ObjectiveDescription {
                kind: ObjectiveKind::Other,
                measure: "real".into(),
                model: "test".into(),
                data: "none".into(),
                source: "test".into(),
            },
            candidate: Some(OptimizationCandidate {
                position: vec![1.],
                gradient: vec![0.],
                value: -0.5,
            }),
            termination: OptimizationTermination::Cancelled,
            line_search_failure: None,
            iterations: 2,
            evaluations: 4,
            domain_trials: 0,
            skipped_updates: 0,
        }
    }

    #[test]
    fn terminal_updates_budgets_and_clears_line_on_completion() {
        let result = result();
        let mut output = Vec::new();
        {
            let mut display = FitProgress {
                terminal: Terminal::new(&mut output, true, false),
                begun: false,
            };
            display.update(&result);
            display.finish::<Infallible>(&Ok(result));
        }
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("Steps [━━──────] 2  Evals [━━──────] 4"));
        assert!(text.contains("\r\x1b[2KCancelled  2 steps · 4 evaluations · best -5.000000e-1"));
        assert!(text.ends_with('\n'));
        assert!(!text.contains("\x1b[36m"));
    }

    #[test]
    fn redirected_display_writes_only_plain_lifecycle_lines() {
        let result = result();
        let mut output = Vec::new();
        {
            let mut display = FitProgress {
                terminal: Terminal::new(&mut output, false, false),
                begun: false,
            };
            display.update(&result);
            display.update(&result);
            display.finish::<Infallible>(&Ok(result));
        }
        let text = String::from_utf8(output).unwrap();
        assert_eq!(text.lines().count(), 2);
        assert!(!text.contains('\x1b'));
        assert!(text.starts_with("Fitting model"));
        assert!(text.contains("\nCancelled"));
    }
}
