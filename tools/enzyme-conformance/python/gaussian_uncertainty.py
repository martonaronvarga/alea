"""Independent exact-dyadic and optional pivoted-QR qualification.

Consumes the three JSONL records emitted by examples/gaussian_uncertainty.rs.
Not a general linear-model API. Also checks the public Gaussian interval output.
Run --scipy in the analysis Nix shell
to execute LAPACK column-pivoted QR and triangular solves as a second oracle.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys

CASES = {"base": (1., 1.), "large": (8., 4.), "small": (.125, .25)}
FIELDS = {
    "case", "rows", "columns", "rank", "df", "design", "response", "offsets",
    "batch_estimates", "generated_estimates", "svd_coefficients", "mle_variance",
    "residual_variance", "joint_covariance", "coefficient_covariance", "corrected_covariance",
    "intervals",
}
CRITICAL_DF4 = {0.9:2.131846786326651, 0.95:2.7764451051977934, 0.99:4.604094871349992}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numbers(value, size):
    require(isinstance(value, list) and len(value) == size, "array shape")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in value), "finite numeric array")
    return value


def close(actual, expected, tolerance=3e-6):
    require(type(actual) in (int, float) and math.isfinite(actual), "finite numeric value")
    require(abs(actual - float(expected)) <= tolerance * (1 + abs(float(expected))),
            f"reference mismatch: {actual} != {float(expected)}")


def check(record, scipy=False):
    require(isinstance(record, dict) and set(record) == FIELDS, "record fields")
    require(type(record["case"]) is str and record["case"] in CASES, "case")
    for key, expected in (("rows", 6), ("columns", 2), ("rank", 2), ("df", 4)):
        require(type(record[key]) is int and record[key] == expected, key)
    x = numbers(record["design"], 12)
    y = numbers(record["response"], 6)
    o = numbers(record["offsets"], 6)
    xs, ys = CASES[record["case"]]
    for i, (predictor, error) in enumerate(zip(range(-2, 4), (1, -2, 1, 1, -2, 1))):
        offset = (i + 1) * .25 * (-1 if i % 2 else 1)
        require(x[2*i:2*i+2] == [1., predictor * xs], "design/orientation")
        require(o[i] == offset * ys, "offsets")
        require(y[i] == (1.5 - .75 * predictor + error + offset) * ys, "responses")
    # Exact independent fixture oracle; no floating normal-equation solver is
    # used by Alea or the QR reference. Only this two-column Fraction calculation.
    z = list(map(F, x[1::2]))
    adjusted = [F(a) - F(b) for a, b in zip(y, o)]
    sx, sxx = sum(z), sum(v*v for v in z)
    sy, sxy = sum(adjusted), sum(a*b for a, b in zip(z, adjusted))
    determinant = 6*sxx - sx*sx
    require(determinant > 0, "exact fixture rank")
    beta = [(sxx*sy - sx*sxy)/determinant, (6*sxy - sx*sy)/determinant]
    rss = sum((a - beta[0] - beta[1]*b)**2 for a, b in zip(adjusted, z))
    require(rss > 0, "positive residual")
    mle, residual = rss/6, rss/4
    unscaled = [sxx/determinant, -sx/determinant, -sx/determinant, F(6)/determinant]
    expected_ml = [mle*a for a in unscaled]
    expected_corrected = [residual*a for a in unscaled]
    close(record["mle_variance"], mle, 1e-12)
    close(record["residual_variance"], residual, 1e-12)
    for name in ("batch_estimates", "generated_estimates"):
        values = numbers(record[name], 3)
        for actual, expected in zip(values[:2], beta):
            close(actual, expected)
        close(values[2], math.sqrt(float(mle)))
    for actual, expected in zip(numbers(record["svd_coefficients"], 2), beta):
        close(actual, expected, 1e-12)
    for name, expected in (("coefficient_covariance", expected_ml), ("corrected_covariance", expected_corrected)):
        for actual, reference in zip(numbers(record[name], 4), expected):
            close(actual, reference)
    joint = numbers(record["joint_covariance"], 9)
    for index, expected in zip((0, 1, 3, 4), expected_ml):
        close(joint[index], expected)
    for index in (2, 5, 6, 7):
        close(joint[index], 0)
    close(joint[8], mle/12)
    for actual, expected in zip((joint[0], joint[1], joint[3], joint[4]), record["coefficient_covariance"]):
        close(actual, expected, 0.)
    require(isinstance(record["intervals"],list) and len(record["intervals"])==3,"three interval levels")
    for row,(level,critical) in zip(record["intervals"],CRITICAL_DF4.items()):
        numbers(row,6)
        require(row[0]==level,"interval levels/order")
        close(row[1],critical,1e-10)
        for j in range(2):
            width=critical*math.sqrt(float(expected_corrected[j*3]))
            close(row[2+2*j],float(beta[j])-width,1e-9)
            close(row[3+2*j],float(beta[j])+width,1e-9)
    result = {"case": record["case"], "oracle": "exact stored-dyadic Fraction",
              "maximum_coefficient_error": max(abs(record[name][i]-float(beta[i]))
                  for name in ("batch_estimates", "generated_estimates") for i in range(2)),
              "maximum_corrected_covariance_error": max(abs(a-float(b))
                  for a, b in zip(record["corrected_covariance"], expected_corrected))}
    if scipy:
        result.update(check_qr(record, beta, expected_corrected))
    return result


def check_qr(record, beta, expected_corrected):
    import numpy as np
    import scipy
    from scipy import linalg
    from scipy.stats import t
    x = np.asarray(record["design"]).reshape(6, 2)
    y = np.asarray(record["response"]) - record["offsets"]
    q, r, pivot = linalg.qr(x, mode="economic", pivoting=True)
    require(np.all(np.abs(np.diag(r)) > 1e-12*np.abs(r[0, 0])), "QR rank")
    fit = np.empty(2)
    fit[pivot] = linalg.solve_triangular(r, q.T @ y)
    rss = float(np.linalg.norm(y - x @ fit)**2)
    # R W = identity; unpivot the covariance. This is NOT inv(X.T @ X).
    w = linalg.solve_triangular(r, np.eye(2))
    covariance = np.empty((2, 2))
    covariance[np.ix_(pivot, pivot)] = (w @ w.T) * (rss/4)
    for actual, expected in zip(fit, beta):
        close(float(actual), expected, 1e-12)
    for actual, expected in zip(covariance.ravel(), expected_corrected):
        close(float(actual), expected, 1e-12)
    for row in record["intervals"]:
        critical=float(t.isf((1-row[0])/2,4))
        close(row[1],critical,1e-10)
        for j in range(2):
            width=critical*math.sqrt(float(covariance[j,j]))
            close(row[2+2*j],float(fit[j])-width,1e-9)
            close(row[3+2*j],float(fit[j])+width,1e-9)
    return {"case": record["case"], "oracle": "exact dyadic + LAPACK pivoted QR/triangular solve",
            "numpy": np.__version__, "scipy": scipy.__version__, "pivot": pivot.tolist()}


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate field")
        result[key] = value
    return result


def read(text):
    records = [json.loads(line, object_pairs_hook=unique_pairs) for line in text.splitlines()]
    require(len(records) == 3, "planned three cases")
    require(all(isinstance(r, dict) and type(r.get("case")) is str for r in records), "case records")
    require({r["case"] for r in records} == set(CASES), "case identities/duplicates")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scipy", action="store_true")
    args = parser.parse_args()
    raw = sys.stdin.buffer.read()
    results = [check(r, args.scipy) for r in read(raw.decode())]
    print(json.dumps({"planned": 3, "completed": len(results), "checks": results,
                      "input_sha256": hashlib.sha256(raw).hexdigest(),
                      "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "scope": "fixed-design Gaussian covariance/marginal-interval reference cohort; no general coverage or speed claim"}))


if __name__ == "__main__":
    main()
