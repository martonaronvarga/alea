"""Alea's package-license boundary and required redistributed notices."""
from pathlib import Path

PERMISSIVE = "MIT OR Apache-2.0"
NATIVE = "GPL-3.0-or-later"
NATIVE_PACKAGES = frozenset({"alea-ffi", "alea-distributions", "alea-cli"})


def expected_license(name: str) -> str:
    return NATIVE if name in NATIVE_PACKAGES else PERMISSIVE


def required_notices(root: Path, name: str) -> dict[str, bytes]:
    files = {"THIRD_PARTY_NOTICES.md": (root / "THIRD_PARTY_NOTICES.md").read_bytes()}
    if name in NATIVE_PACKAGES:
        files["LICENSE"] = (root / "LICENSE-GPL").read_bytes()
    else:
        for filename in ("LICENSE-MIT", "LICENSE-APACHE"):
            files[filename] = (root / filename).read_bytes()
    return files


def check_graph(packages: list[dict]) -> None:
    """Reject a permissive declaration hiding a production native dependency."""
    by_name = {p["name"]: p for p in packages}
    def uses_native(name, seen):
        if name == "alea-ffi":
            return True
        if name in seen:
            return False
        seen.add(name)
        return any(uses_native(dep["name"], seen)
                   for dep in by_name[name]["dependencies"]
                   if dep["name"] in by_name and dep["kind"] != "dev")
    for name, package in by_name.items():
        if package.get("license") != expected_license(name):
            raise ValueError(f"{name}: expected {expected_license(name)}")
        if uses_native(name, set()) != (name in NATIVE_PACKAGES):
            raise ValueError(f"{name}: production native dependency changed; licensing review required")


def check_archive(root: Path, package: dict, archive) -> None:
    prefix = package["name"] + "-" + package["version"] + "/"
    for filename, expected in required_notices(root, package["name"]).items():
        try:
            member = archive.getmember(prefix + filename)
            with archive.extractfile(member) as stream:
                actual = stream.read()
        except (KeyError, AttributeError) as error:
            raise ValueError(f"{package['name']}: packaged notice missing: {filename}") from error
        if actual != expected:
            raise ValueError(f"{package['name']}: packaged notice differs: {filename}")
