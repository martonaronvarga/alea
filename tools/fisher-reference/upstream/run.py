"""Execute pinned nuts-rs Fisher fits in isolation; never modify Alea dependencies.

Run inside the stable Nix shell. --check compares numerically (not eigenvectors
or bytes); --write explicitly replaces the checked-in fixture. --lock prepares
the separate dependency lockfile and does not generate fixtures.
"""
import argparse
import csv
import io
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REV = "a762aae513bf7bb5ccb27ffc9a938a923fc837ec"
FIXTURE = ROOT / "crates/alea-math/tests/fixtures/fisher-nuts-rs.csv"


def compare(actual, expected, updates=False):
    def rows(text):
        return list(csv.reader(io.StringIO('\n'.join(
            line for line in text.splitlines() if not line.startswith('#')))))
    actual, expected = rows(actual), rows(expected)
    assert len(actual) == len(expected) == (29 if updates else 19)
    assert actual[0] == expected[0]
    for a, e in zip(actual[1:], expected[1:]):
        assert len(a) == len(e) == (15 if updates else 11)
        if updates:
            assert a[:8] == e[:8] and a[9:12] == e[9:12] and a[14] == e[14]
        else:
            assert a[:5] == e[:5] and a[6:9] == e[6:9]
        for field in ([8, 12, 13] if updates else [5, 9, 10]):
            av, ev = a[field].split(';'), e[field].split(';')
            assert len(av) == len(ev)
            for x, y in zip(av, ev):
                x, y = float(x), float(y)
                assert math.isfinite(x) and math.isfinite(y)
                assert abs(x - y) <= 1e-9 * (1 + abs(y)), (a[0], field, x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--lock', action='store_true')
    parser.add_argument('--updates', action='store_true', help='execute collector/window/installed-metric traces')
    args = parser.parse_args()
    fixture = ROOT / 'crates/alea-mcmc/tests/fixtures/fisher-nuts-updates.csv' if args.updates else FIXTURE
    # All patches apply exclusively to this freshly created temporary checkout.
    with tempfile.TemporaryDirectory(prefix='alea-fisher-upstream-') as temp:
        work = Path(temp)
        upstream = work / 'upstream'
        subprocess.run(['git', 'init', '-q', str(upstream)], check=True)
        subprocess.run(['git', '-C', str(upstream), 'fetch', '-q', '--depth=1',
                        'https://github.com/pymc-devs/nuts-rs.git', REV], check=True)
        subprocess.run(['git', '-C', str(upstream), 'checkout', '-q', '--detach', 'FETCH_HEAD'], check=True)
        head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
        assert head == REV
        shutil.copyfile(HERE / 'Cargo.toml', work / 'Cargo.toml')
        (work / 'src').mkdir()
        shutil.copyfile(HERE / 'main.rs', work / 'src/main.rs')
        if args.lock:
            subprocess.run(['cargo', 'generate-lockfile', '--manifest-path', str(work / 'Cargo.toml')], check=True)
            shutil.copyfile(work / 'Cargo.lock', HERE / 'Cargo.lock')
            return
        shutil.copyfile(HERE / 'Cargo.lock', work / 'Cargo.lock')
        source = upstream / 'src/transform/adapt/low_rank.rs'
        source.write_text(source.read_text() + '\n' + (HERE / 'probe.rs').read_text()
                          + '\n' + (HERE / 'update_probe.rs').read_text())
        math_module = upstream / 'src/math/mod.rs'
        original = math_module.read_text()
        needle = '#[cfg(test)]\npub mod test_logps;'
        assert original.count(needle) == 1
        math_module.write_text(original.replace(needle, 'pub mod test_logps;'))
        library = upstream / 'src/lib.rs'
        library.write_text(library.read_text() + '\npub use transform::LowRankMassMatrixStrategy as AleaReference;\n')
        env = os.environ.copy()
        env['CARGO_TARGET_DIR'] = str(ROOT / 'target/fisher-upstream')
        output = subprocess.check_output(['cargo', 'run', '--quiet', '--locked',
            '--manifest-path', str(work / 'Cargo.toml'), '--', *(['--updates'] if args.updates else [])], env=env, text=True)
        compare(output, output, args.updates)  # Validate even on --write.
        if args.write:
            fixture.write_text(output)
        else:
            compare(output, fixture.read_text(), args.updates)
        print('validated pinned upstream Fisher ' + ('updates' if args.updates else 'fits'))


if __name__ == '__main__':
    main()
