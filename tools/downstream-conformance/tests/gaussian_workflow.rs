#![cfg(feature = "faer")]
use alea::analysis::AnalysisError;
use alea_downstream_conformance::gaussian_workflow::{CASES, write_case};
use std::{
    io::{self, Write},
    process::{Command, Stdio},
};

#[test]
fn native_gaussian_workflow_retains_all_six_outcomes_and_round_trips_to_independent_consumer() {
    let mut bytes = Vec::new();
    for case in CASES {
        write_case(case, &mut bytes).unwrap();
    }
    assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), CASES.len());
    let mut first = Vec::new();
    let mut repeated = Vec::new();
    write_case("qualified", &mut first).unwrap();
    write_case("qualified", &mut repeated).unwrap();
    assert_eq!(
        first, repeated,
        "same model/data/options must replay exactly"
    );
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("python/gaussian_workflow.py");
    let mut child = Command::new("python3")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["completed"], 6);
    assert_eq!(
        summary["native_reference_consistent"],
        serde_json::json!(["qualified"])
    );
    assert_eq!(
        summary["intervals_available_for_separate_ols"],
        serde_json::json!(["qualified", "budget", "cancelled", "initial_transform"])
    );
}

#[test]
fn unknown_cases_write_nothing_and_original_partial_io_failures_propagate() {
    let mut bytes = Vec::new();
    assert!(write_case("unknown", &mut bytes).is_err());
    assert!(bytes.is_empty());
    struct FailAfterPrefix {
        prefix: Vec<u8>,
    }
    impl Write for FailAfterPrefix {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.prefix.is_empty() {
                let n = bytes.len().min(13);
                self.prefix.extend_from_slice(&bytes[..n]);
                Ok(n)
            } else {
                Err(io::Error::from_raw_os_error(28))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut writer = FailAfterPrefix { prefix: Vec::new() };
    let error = write_case("qualified", &mut writer).unwrap_err();
    let Some(AnalysisError::Io(error)) = error.downcast_ref::<AnalysisError>() else {
        panic!("lost original I/O error")
    };
    assert_eq!(error.raw_os_error(), Some(28));
    assert_eq!(writer.prefix.len(), 13);
    // Recovery uses a new output, not appending to the partial document.
    write_case("qualified", &mut bytes).unwrap();
    assert!(serde_json::from_slice::<serde_json::Value>(&bytes).is_ok());
}
