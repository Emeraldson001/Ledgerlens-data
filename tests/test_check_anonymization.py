from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_anonymization import (
    check_file,
    is_exempt_ipv4,
    is_valid_ipv4,
    load_allowlist,
)


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    # Setup mock repo root with allowlist
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    allowlist_path = data_dir / "allowlist.json"
    with open(allowlist_path, "w") as f:
        json.dump(["GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"], f)
    return tmp_path


def test_is_valid_ipv4():
    assert is_valid_ipv4("192.168.1.1")
    assert is_valid_ipv4("0.0.0.0")
    assert not is_valid_ipv4("256.0.0.1")
    assert not is_valid_ipv4("192.168.1")
    assert not is_valid_ipv4("not.an.ip")


def test_is_exempt_ipv4():
    assert is_exempt_ipv4("127.0.0.1")
    assert is_exempt_ipv4("0.0.0.0")
    assert is_exempt_ipv4("192.168.1.50")
    assert is_exempt_ipv4("10.0.0.5")
    assert not is_exempt_ipv4("8.8.8.8")
    assert not is_exempt_ipv4("1.1.1.1")


def test_load_allowlist(repo_root: Path):
    allowlist = load_allowlist(repo_root)
    assert len(allowlist) == 1
    assert "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN" in allowlist


def test_check_file_flags_real_public_keys(tmp_path: Path):
    test_file = tmp_path / "test.json"
    with open(test_file, "w") as f:
        f.write('{"account": "GCEZWKCA5VLDNRLN3RPRJMRZOX3Z6G5CHCGYWDEAVJJCSBVALM2XVKXB"}')

    violations = check_file(test_file, set())
    assert len(violations) == 1
    assert violations[0][0] == 1
    assert violations[0][1] == "Stellar Public Key"
    assert violations[0][2] == "GCEZWKCA5VLDNRLN3RPRJMRZOX3Z6G5CHCGYWDEAVJJCSBVALM2XVKXB"


def test_check_file_flags_secret_keys(tmp_path: Path):
    test_file = tmp_path / "test.json"
    with open(test_file, "w") as f:
        f.write('{"secret": "SDQGBA4BLXU4P2XG66Z3L5E7XZ3F76Y6V5WZV4F2GZYX5V4F2GZYX5VA"}')

    violations = check_file(test_file, set())
    assert len(violations) == 1
    assert violations[0][1] == "Stellar Secret Key"


def test_check_file_ignores_allowlisted_keys(tmp_path: Path):
    test_file = tmp_path / "test.json"
    allowlist_key = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
    with open(test_file, "w") as f:
        f.write(f'{{"account": "{allowlist_key}"}}')

    violations = check_file(test_file, {allowlist_key})
    assert len(violations) == 0


def test_check_file_ignores_synthetic_keys(tmp_path: Path):
    test_file = tmp_path / "test.json"
    with open(test_file, "w") as f:
        f.write('{"account": "GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ234567ABCDEFGHIJKLMNOPQ"}\n')
        f.write('{"account": "GBTESTABCDEFGHIJKLMNOPQRSTUVWXYZ234567ABCDEFGHIJKLMNOPQ"}\n')
        f.write('{"account": "GSYNTHABCDEFGHIJKLMNOPQRSTUVWXYZ234567ABCDEFGHIJKLMNOPQ"}\n')

    violations = check_file(test_file, set())
    assert len(violations) == 0


def test_check_file_flags_real_ips_but_ignores_local(tmp_path: Path):
    test_file = tmp_path / "test.json"
    with open(test_file, "w") as f:
        f.write('{"local": "127.0.0.1", "external": "8.8.8.8"}')

    violations = check_file(test_file, set())
    assert len(violations) == 1
    assert violations[0][1] == "IPv4 Address"
    assert violations[0][2] == "8.8.8.8"
