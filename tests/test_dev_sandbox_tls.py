"""Behavioral contracts for the dev sandbox HTTPS proxy."""

import importlib.util
from pathlib import Path
import re
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE1 = _REPO_ROOT / "scripts" / "dev-sandbox.sh"
_STAGE2 = _REPO_ROOT / "scripts" / "sandbox" / "stage2-run.sh"
_PROXY = _REPO_ROOT / "scripts" / "sandbox" / "proxy.py"


def _sandbox_environment() -> dict[str, str]:
    source = _STAGE2.read_text(encoding="utf-8")
    return dict(re.findall(r"--setenv ([A-Z0-9_]+) ([^\\\s]+)", source))


def test_node_trusts_the_same_mitm_ca_as_other_sandbox_clients():
    env = _sandbox_environment()

    # The fake-internet proxy presents certificates minted by ca.pem. Node must
    # trust that MITM CA just like curl, Python/OpenSSL, and Git do; real-ca.pem
    # is reserved for the proxy's own upstream verification.
    assert env["NODE_EXTRA_CA_CERTS"] == env["CURL_CA_BUNDLE"]
    assert env["NODE_EXTRA_CA_CERTS"] == env["SSL_CERT_FILE"]
    assert env["NODE_EXTRA_CA_CERTS"] == env["GIT_SSL_CAINFO"]
    assert env["NODE_EXTRA_CA_CERTS"] == "/work/certs/client-ca.pem"

    stage1 = _STAGE1.read_text(encoding="utf-8")
    assert 'root/certs/ca.pem" "$SANDBOX_ROOT/root/certs/real-ca.pem"' in stage1
    assert '> "$SANDBOX_ROOT/root/certs/client-ca.pem"' in stage1


def test_https_hosts_without_fixtures_use_a_connect_tunnel(tmp_path, monkeypatch):
    fixture_root = tmp_path / "http"
    certs = tmp_path / "certs"
    fixture_root.mkdir()
    certs.mkdir()
    real_ca = certs / "real-ca.pem"
    real_ca.touch()

    monkeypatch.setattr(
        sys, "argv", [str(_PROXY), str(fixture_root), str(certs), str(real_ca)]
    )
    spec = importlib.util.spec_from_file_location("dev_sandbox_proxy", _PROXY)
    assert spec is not None and spec.loader is not None
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)

    calls = []
    monkeypatch.setattr(
        proxy,
        "tunnel_https",
        lambda conn, host, port: calls.append((conn, host, port)),
    )
    connection = object()
    proxy.handle_connect(connection, "registry.npmjs.org:443")

    assert calls == [(connection, "registry.npmjs.org", 443)]
