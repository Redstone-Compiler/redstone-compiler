use std::collections::BTreeMap;
use std::time::Duration;

use crate::world::position::Position;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvaluationStage {
    Placement,
    RouteProbe,
    Routing,
    ShortVerification,
    FullVerification,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GlobalFailure {
    IllegalPlacement { reason: String },
    RouteSearchExhausted { net: String, source: Position, sink: Position },
    ForbiddenSignalContact { net: String, position: Position },
    PoweredPositionContract { net: String, source: Position, sink: Position },
    Assembly { reason: String },
    SemanticMismatch { phase: String, step: usize, expected: usize, actual: usize },
    BudgetExhausted { stage: EvaluationStage },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlobalAttemptOutcome {
    Passed,
    Failed(GlobalFailure),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageTiming {
    pub stage: EvaluationStage,
    pub elapsed: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalAttemptRecord {
    pub signature: String,
    pub timings: Vec<StageTiming>,
    pub outcome: GlobalAttemptOutcome,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GlobalSearchReport {
    pub attempts: Vec<GlobalAttemptRecord>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StageTimingSummary {
    pub count: usize,
    pub total: Duration,
}

impl GlobalSearchReport {
    pub fn failure_counts(&self) -> BTreeMap<GlobalFailure, usize> {
        let mut counts = BTreeMap::new();
        for attempt in &self.attempts {
            if let GlobalAttemptOutcome::Failed(failure) = &attempt.outcome {
                *counts.entry(failure.clone()).or_default() += 1;
            }
        }
        counts
    }

    pub fn stage_timings(&self) -> BTreeMap<EvaluationStage, StageTimingSummary> {
        let mut summaries = BTreeMap::new();
        for timing in self.attempts.iter().flat_map(|attempt| &attempt.timings) {
            let summary = summaries.entry(timing.stage).or_insert_with(StageTimingSummary::default);
            summary.count += 1;
            summary.total += timing.elapsed;
        }
        summaries
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        EvaluationStage, GlobalAttemptOutcome, GlobalAttemptRecord, GlobalFailure,
        GlobalSearchReport, StageTiming,
    };
    use crate::world::position::Position;

    fn failed_attempt(signature: &str, failure: GlobalFailure) -> GlobalAttemptRecord {
        GlobalAttemptRecord {
            signature: signature.to_owned(),
            timings: Vec::new(),
            outcome: GlobalAttemptOutcome::Failed(failure),
        }
    }

    #[test]
    fn report_groups_failures_by_structured_signature() {
        let repeated = GlobalFailure::RouteSearchExhausted {
            net: "carry".to_owned(),
            source: Position(1, 2, 3),
            sink: Position(4, 5, 6),
        };
        let distinct = GlobalFailure::RouteSearchExhausted {
            net: "carry".to_owned(),
            source: Position(1, 2, 3),
            sink: Position(7, 8, 9),
        };
        let report = GlobalSearchReport {
            attempts: vec![
                failed_attempt("a", repeated.clone()),
                failed_attempt("b", repeated),
                failed_attempt("c", distinct),
            ],
        };

        let counts = report.failure_counts();
        assert_eq!(counts.len(), 2);
        let mut grouped = counts.values().copied().collect::<Vec<_>>();
        grouped.sort_unstable();
        assert_eq!(grouped, vec![1, 2]);
    }

    #[test]
    fn report_sums_stage_timings() {
        let report = GlobalSearchReport {
            attempts: vec![GlobalAttemptRecord {
                signature: "a".to_owned(),
                timings: vec![
                    StageTiming {
                        stage: EvaluationStage::Routing,
                        elapsed: Duration::from_millis(10),
                    },
                    StageTiming {
                        stage: EvaluationStage::Routing,
                        elapsed: Duration::from_millis(20),
                    },
                ],
                outcome: GlobalAttemptOutcome::Passed,
            }],
        };

        let routing = &report.stage_timings()[&EvaluationStage::Routing];
        assert_eq!(routing.count, 2);
        assert_eq!(routing.total, Duration::from_millis(30));
    }
}
