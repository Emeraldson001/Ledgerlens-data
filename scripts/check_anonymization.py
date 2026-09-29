import argparse
import json
import re
import sys
from pathlib import Path

STELLAR_PUB_REGEX = re.compile(r"\bG[A-Z2-7]{55}\b")
STELLAR_SEC_REGEX = re.compile(r"\bS[A-Z2-7]{55}\b")
IPV4_REGEX = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")

# Repeating pattern exception
REPEATING_PUB_REGEX = re.compile(r"^(G)([A-Z2-7])\2{54}$")
SYNTHETIC_PREFIXES = ("GBTEST", "GSYNTH")


def is_valid_ipv4(ip: str) -> bool:
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    return all(0 <= int(p) <= 255 for p in parts)


def is_exempt_ipv4(ip: str) -> bool:
    if ip in ("127.0.0.1", "0.0.0.0"):
        return True
    if (
        ip.startswith("192.168.")
        or ip.startswith("10.")
        or ip.startswith("172.16.")
        or ip.startswith("172.31.")
    ):
        return True
    return False


def load_allowlist(repo_root: Path) -> set[str]:
    allowlist_path = repo_root / "data" / "allowlist.json"
    if allowlist_path.exists():
        try:
            with open(allowlist_path, encoding="utf-8") as f:
                return set(json.load(f))
        except Exception as e:
            print(f"Warning: could not load allowlist.json: {e}")
            return set()
    return set()


def check_file(file_path: Path, allowlist: set[str]) -> list[tuple[int, str, str]]:
    violations = []
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        return violations

    for i, line in enumerate(lines, 1):
        # Check Public Keys
        for match in STELLAR_PUB_REGEX.findall(line):
            if match in allowlist:
                continue
            if REPEATING_PUB_REGEX.match(match):
                continue
            if any(match.startswith(prefix) for prefix in SYNTHETIC_PREFIXES):
                continue
            violations.append((i, "Stellar Public Key", match))

        # Check Secret Keys
        for match in STELLAR_SEC_REGEX.findall(line):
            # No exceptions for secret keys
            violations.append((i, "Stellar Secret Key", match))

        # Check IPv4
        for match in IPV4_REGEX.findall(line):
            if not is_valid_ipv4(match):
                continue
            if is_exempt_ipv4(match):
                continue
            violations.append((i, "IPv4 Address", match))

    return violations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check for unanonymized PII in shared example data."
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=["data", "tests/fixtures", "tests/fuzz/corpus"],
        help="Directories to scan.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    allowlist = load_allowlist(repo_root)

    extensions = {".json", ".csv", ".md", ".yaml", ".yml", ".txt"}
    has_violations = False

    for target in args.target:
        target_path = repo_root / target
        if not target_path.exists():
            continue

        if target_path.is_file():
            files = [target_path]
        else:
            files = [
                p for p in target_path.rglob("*") if p.is_file() and p.suffix.lower() in extensions
            ]

        for file_path in files:
            violations = check_file(file_path, allowlist)
            for line_num, type_, val in violations:
                has_violations = True
                masked_val = val[:6] + "..." + val[-4:] if len(val) > 10 else "***"
                try:
                    rel_path = file_path.relative_to(repo_root)
                except ValueError:
                    rel_path = file_path
                print(f"[{type_}] {rel_path}:{line_num} -> {masked_val}")

    if has_violations:
        print("ERROR: Unanonymized PII found. Please mask or add to data/allowlist.json.")
        sys.exit(1)
    else:
        print("SUCCESS: No unanonymized PII found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
