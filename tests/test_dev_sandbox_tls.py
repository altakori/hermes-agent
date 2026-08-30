"""Behavioral contracts for the dev sandbox TLS trust split."""

from pathlib import Path
import re


_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE2 = _REPO_ROOT / "scripts" / "sandbox" / "stage2-run.sh"


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
