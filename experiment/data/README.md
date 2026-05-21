# DongTing Dataset

**Source:** G. Duan, Y. Fu, M. Cai, H. Chen, and J. Sun, "DongTing: A large-scale dataset for anomaly detection of the Linux kernel," *J. Syst. Softw.*, vol. 203, 2023.

## Dataset Overview

- **18,966** syscall sequences, **2.6M+** individual syscalls
- **6,850** benign sequences (web servers, databases — `open`, `read`, `mmap`, `futex`)
- **12,116** malicious sequences (privilege escalation, filesystem tampering, network intrusion)
- Each sequence: syscall IDs + timestamps + contextual metadata (PID, namespace)

## Download

Option 1 — GitHub release:
```
https://github.com/DongTingDingData/DongTing
```

Option 2 — Zenodo/paper supplementary:
Search DOI of the paper on https://zenodo.org or the journal page.

## Expected Format

Place downloaded files under `experiment/data/dongting/`:

```
data/dongting/
├── benign/          # JSON or CSV per-container trace
│   ├── mysql.json
│   ├── nginx.json
│   └── ...
└── malicious/       # Labeled attack traces
    ├── priv_esc.json
    ├── container_escape.json
    └── ...
```

Each trace file should contain records with fields:
```json
{
  "syscall_id": 3,
  "syscall_name": "read",
  "timestamp_ns": 1680000000000000000,
  "pid": 1234,
  "ppid": 1,
  "comm": "nginx",
  "container_name": "web-server",
  "namespace": "default",
  "return_value": 512
}
```

## Fallback: Synthetic Dataset

If the dataset is unavailable, the notebook auto-generates a synthetic dataset
with the same statistical properties (18,966 sequences, same class ratio) so
all model code can still be exercised.
