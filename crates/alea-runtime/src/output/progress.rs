//! Shared best-effort stderr rendering, separate from inference records.
use std::{
    env,
    io::{self, IsTerminal, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const INTERVAL: Duration = Duration::from_millis(100);
const CLEAR_LINE: &str = "\r\x1b[2K";

pub(crate) struct Terminal<W: Write> {
    output: W,
    live: bool,
    color: bool,
    started: Instant,
    last_update: Option<Instant>,
    line_visible: bool,
    writable: bool,
}

impl Terminal<io::Stderr> {
    pub(crate) fn stderr() -> Self {
        let output = io::stderr();
        let term = env::var_os("TERM");
        let live = cfg!(unix)
            && output.is_terminal()
            && term.is_some_and(|term| !term.is_empty() && term != "dumb");
        Self::new(output, live, live && env::var_os("NO_COLOR").is_none())
    }
}

impl<W: Write> Terminal<W> {
    pub(crate) fn new(output: W, live: bool, color: bool) -> Self {
        Self {
            output,
            live,
            color,
            started: Instant::now(),
            last_update: None,
            line_visible: false,
            writable: true,
        }
    }

    fn write(&mut self, message: &str) {
        if self.writable {
            self.writable = self
                .output
                .write_all(message.as_bytes())
                .and_then(|()| self.output.flush())
                .is_ok();
        }
    }

    pub(crate) fn heading(&mut self, title: &str, detail: &str) {
        let (start, end) = self.style("\x1b[1;36m");
        self.write(&format!("{start}{title}{end}  {detail}\n"));
    }

    pub(crate) fn style(&self, code: &'static str) -> (&'static str, &'static str) {
        if self.color {
            (code, "\x1b[0m")
        } else {
            ("", "")
        }
    }

    pub(crate) fn ready(&self) -> bool {
        self.live
            && self.writable
            && self
                .last_update
                .is_none_or(|last| last.elapsed() >= INTERVAL)
    }

    pub(crate) fn render(&mut self, line: &str) {
        self.last_update = Some(Instant::now());
        self.line_visible = true;
        self.write(&format!("{CLEAR_LINE}{line}"));
    }

    pub(crate) fn finish(&mut self, status: &str, detail: &str) {
        let prefix = if self.line_visible { CLEAR_LINE } else { "" };
        let (start, end) = self.style("\x1b[1m");
        let elapsed = self.started.elapsed().as_secs_f64();
        self.write(&format!(
            "{prefix}{start}{status}{end}{detail} · {elapsed:.2}s\n"
        ));
        if self.writable {
            self.line_visible = false;
        }
    }
}

impl<W: Write> Drop for Terminal<W> {
    fn drop(&mut self) {
        if self.line_visible {
            let _ = self.output.write_all(CLEAR_LINE.as_bytes());
            let _ = self.output.flush();
        }
    }
}

pub(crate) fn bar(done: u128, limit: u128) -> String {
    const WIDTH: usize = 8;
    // The displayed counts remain exact. Round down using the thresholds
    // rather than multiplying potentially large ensemble totals by WIDTH.
    let filled = (1..=WIDTH as u128)
        .filter(|&i| {
            done >= (limit / WIDTH as u128) * i
                + ((limit % WIDTH as u128) * i).div_ceil(WIDTH as u128)
        })
        .count();
    format!("[{}{}]", "━".repeat(filled), "─".repeat(WIDTH - filled))
}

#[derive(Default)]
struct Counts {
    warmup: AtomicUsize,
    draws: AtomicUsize,
}

pub(crate) struct LiveSampling {
    stop: AtomicBool,
    chains: Box<[Counts]>,
}

impl LiveSampling {
    pub(crate) fn update(&self, index: usize, warmup: usize, draws: usize) {
        let chain = &self.chains[index];
        chain.warmup.store(warmup, Relaxed);
        chain.draws.store(draws, Relaxed);
    }

    fn totals(&self) -> (u128, u128) {
        counters(
            self.chains
                .iter()
                .map(|chain| (chain.warmup.load(Relaxed), chain.draws.load(Relaxed))),
        )
    }
}

pub(crate) fn counters(pairs: impl Iterator<Item = (usize, usize)>) -> (u128, u128) {
    pairs.fold((0, 0), |(w, d), (warmup, draws)| {
        (w + warmup as u128, d + draws as u128)
    })
}

pub(crate) struct SamplingProgress {
    terminal: Option<Terminal<io::Stderr>>,
    worker: Option<(Arc<LiveSampling>, JoinHandle<Terminal<io::Stderr>>)>,
}

impl SamplingProgress {
    pub(crate) fn start(
        kernel: &str,
        warmup: usize,
        draws: usize,
        initial_draws: impl ExactSizeIterator<Item = usize>,
    ) -> Self {
        let mut terminal = Terminal::stderr();
        let count = initial_draws.len();
        let chains = if count == 1 { "chain" } else { "chains" };
        let detail =
            format!("{kernel} · {count} {chains} · {warmup} warmup + {draws} draws per chain");
        terminal.heading("Sampling", &detail);
        if !terminal.ready() {
            return Self {
                terminal: Some(terminal),
                worker: None,
            };
        }

        let live = Arc::new(LiveSampling {
            stop: AtomicBool::new(false),
            chains: initial_draws
                .map(|draws| Counts {
                    warmup: AtomicUsize::new(0),
                    draws: AtomicUsize::new(draws),
                })
                .collect(),
        });
        let shared = Arc::clone(&live);
        let warmup = count as u128 * warmup as u128;
        let draws = count as u128 * draws as u128;
        // Render the initial counts even for a run that completes between ticks.
        render_sampling(&mut terminal, live.totals(), (warmup, draws));
        match thread::Builder::new()
            .name("alea-progress".into())
            .spawn(move || {
                while !shared.stop.load(Relaxed) && terminal.writable {
                    thread::park_timeout(INTERVAL);
                    if !shared.stop.load(Relaxed) && terminal.ready() {
                        render_sampling(&mut terminal, shared.totals(), (warmup, draws));
                    }
                }
                terminal
            }) {
            Ok(worker) => Self {
                terminal: None,
                worker: Some((live, worker)),
            },
            Err(_) => Self {
                terminal: Some(Terminal::stderr()),
                worker: None,
            },
        }
    }

    pub(crate) fn live(&self) -> Option<&LiveSampling> {
        self.worker.as_ref().map(|(live, _)| live.as_ref())
    }

    fn stop(&mut self) {
        if let Some((live, worker)) = self.worker.take() {
            live.stop.store(true, Relaxed);
            worker.thread().unpark();
            self.terminal = worker.join().ok();
        }
    }

    pub(crate) fn finish(&mut self, status: &str, (warmup, draws): (u128, u128)) {
        self.stop();
        if let Some(terminal) = &mut self.terminal {
            terminal.finish(status, &format!("  {warmup} warmup · {draws} draws"));
        }
    }
}

impl Drop for SamplingProgress {
    fn drop(&mut self) {
        self.stop();
    }
}

fn render_sampling<W: Write>(
    terminal: &mut Terminal<W>,
    (warmup, draws): (u128, u128),
    (warmup_limit, draw_limit): (u128, u128),
) {
    let (style, reset) = terminal.style("\x1b[36m");
    let d = bar(draws, draw_limit);
    if warmup_limit == 0 {
        terminal.render(&format!("{style}Draws {d}{reset} {draws}/{draw_limit}"));
        return;
    }
    let w = bar(warmup, warmup_limit);
    terminal.render(&format!(
        "{style}Warmup {w}{reset} {warmup}/{warmup_limit}  {style}Draws {d}{reset} {draws}/{draw_limit}"
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_display_uses_actual_totals_and_clears_its_line() {
        let mut output = Vec::new();
        {
            let mut terminal = Terminal::new(&mut output, true, false);
            render_sampling(&mut terminal, (5, 3), (10, 12));
            terminal.finish("Sampling cancelled", "  5 warmup · 3 draws");
        }
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("Warmup [━━━━────] 5/10  Draws [━━──────] 3/12"));
        assert!(text.contains("\r\x1b[2KSampling cancelled  5 warmup · 3 draws"));
        assert!(text.ends_with('\n'));
        assert_eq!(bar(u128::MAX / 2, u128::MAX), "[━━━─────]");
    }
}
