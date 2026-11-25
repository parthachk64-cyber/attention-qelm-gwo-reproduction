"""Label contract, class aliases and the leakage-safe column rules.

Canonical class names are the paper's (Table 1/8). Aliases record spellings seen in the provider's files or in the
paper itself (for example the paper prints ``DDos_HTTP`` in Table 1 and ``DDoS_HTTP`` elsewhere). Alias provenance
is noted per entry; entries marked ``provider`` come from the Edge-IIoTset naming convention and must be verified
against the downloaded CSV by ``iiot-qelm data check``.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

from ..errors import LeakageError, SchemaError

CLASSES: Sequence[str] = (
    "Normal", "DDoS_UDP", "DDoS_ICMP", "SQL_Injection", "DDoS_TCP", "DDoS_HTTP", "Password", "Vulnerability_Scanner",
    "Port_Scanning", "XSS", "Uploading", "Backdoor", "OS_Fingerprinting", "Ransomware", "MITM",
)
NORMAL = "Normal"

# canonical -> [(alias, where it comes from)]
ALIAS_SOURCES: Dict[str, List[tuple]] = {
    "DDoS_HTTP": [("DDos_HTTP", "paper Table 1 print"), ("DDoS_Http", "case variant")],
    "SQL_Injection": [("SQL_injection", "provider"), ("SQLInjection", "case/underscore variant")],
    "Vulnerability_Scanner": [("Vulnerability_scanner", "provider")],
    "OS_Fingerprinting": [("Fingerprinting", "provider; also the label printed in paper Fig. 7")],
    "MITM": [("MTTM", "upstream preprocessing typo map"), ("Man-in-the-Middle", "paper text"), ("Man_in_the_Middle", "variant")],
}

LABEL_COLUMNS = ("Attack_label", "Attack_type")      # provider's binary and 15-class columns


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(s).strip().lower())


def alias_map() -> Dict[str, str]:
    m = {_norm(c): c for c in CLASSES}
    for canon, items in ALIAS_SOURCES.items():
        for a, _ in items:
            m[_norm(a)] = canon
    return m


_ALIAS = alias_map()


def canonical_label(raw: str) -> str:
    key = _norm(raw)
    if key not in _ALIAS:
        raise SchemaError(f"unknown class label {raw!r}; known canonical classes: {list(CLASSES)}; add an alias in data/schema.py only after checking the source file")
    return _ALIAS[key]


def canonicalise(labels: Iterable[str]) -> List[str]:
    return [canonical_label(x) for x in labels]


def class_index(name: str) -> int:
    return CLASSES.index(name)


def multiclass_target(canon_labels: Sequence[str]) -> List[int]:
    return [class_index(c) for c in canon_labels]


def binary_target(canon_labels: Sequence[str]) -> List[int]:
    """Normal = 0, any attack = 1."""
    return [0 if c == NORMAL else 1 for c in canon_labels]


def target_for(task: str, canon_labels: Sequence[str]) -> List[int]:
    if task == "binary":
        return binary_target(canon_labels)
    if task == "multiclass":
        return multiclass_target(canon_labels)
    raise SchemaError(f"task must be 'binary' or 'multiclass', got {task!r}")


def class_names_for(task: str) -> List[str]:
    return ["Normal", "Attack"] if task == "binary" else list(CLASSES)


def assert_no_label_leak(feature_columns: Iterable[str]) -> None:
    """A binary or multiclass label column must never be a feature for either task."""
    bad = [c for c in feature_columns if c in LABEL_COLUMNS or _norm(c) in {_norm(x) for x in LABEL_COLUMNS}]
    if bad:
        raise LeakageError(f"label column(s) present among features: {bad}")


# Columns the paper removes as identifiers / metadata (Sec. 2.2). Matching is EXACT, not substring: the upstream
# code matches substrings, so its pattern 'mqtt.msg' also removes 'mqtt.msgtype' and 'mqtt.msg_decoded_as'
# (unverified against the file; see docs/paper-audit.md).
METADATA_EXACT = (
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4", "http.file_data",
    "http.request.full_uri", "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options", "tcp.payload",
    "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
)
