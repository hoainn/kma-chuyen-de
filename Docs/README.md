# Docs — reference PDFs for the SyscallAD report (upload to NotebookLM)

Recent (2023–2026) research for *"Phát hiện tấn công leo thang đặc quyền trong Kubernetes
dựa trên lời gọi hệ thống"*. Upload these to the NotebookLM notebook
**"DeSFAM — Syscall Privilege-Escalation Detection"** (drag-drop into the notebook UI).

| File | Paper | Year | arXiv/DOI | Why it's relevant | suggested citekey |
|---|---|---|---|---|---|
| `00` | DeSFAM: Adaptive eBPF + AI framework (Zehra et al.) | 2025 | 10.1109/ACCESS.2025.3592192 | The framework we study (SyscallAD = VAE+iForest) | `desfam_original` |
| `01` | eBPF-PATROL: runtime threat recognition & overreach limit | 2025 | arXiv:2511.18155 | eBPF agent detecting **privilege escalation / container escape** via syscalls — direct SOTA peer | `ebpfpatrol2025` |
| `02` | Programmable System Call Security with eBPF | 2023 | arXiv:2302.10366 | syscall security enforcement with eBPF (vs seccomp) | `progsyscall2023` |
| `03` | FedMon: Federated eBPF monitoring, distributed anomaly detection | 2025 | arXiv:2510.10126 | eBPF anomaly detection across multi-cluster cloud | `fedmon2025` |
| `04` | LIGHT-HIDS: lightweight ML host IDS | 2025 | arXiv:2509.13464 | compact-feature **syscall** HIDS + **novelty detection** (one-class, like SyscallAD) | `lighthids2025` |
| `05` | Language Models for Novelty Detection in System Call Traces | 2023 | arXiv:2309.02206 | LM-based **novelty detection on syscall traces** — modern analogue to our anomaly core | `lmnovelty2023` |
| `06` | Deep Learning-based IDS: A Survey | 2025 | arXiv:2504.07839 | recent survey to anchor related-work | `dlidssurvey2025` |
| `07` | Transformers & LLMs for Efficient IDS: A Survey | 2024 | arXiv:2408.07583 | recent survey (seq models for IDS) | `translmidssurvey2024` |
| `08` | Adversarial-Robust Behavior-Sequence Anomaly Detection | 2025 | arXiv:2509.15756 | robustness of sequence anomaly detection (threats-to-validity / evasion) | `advrobustseq2025` |

## Next steps (after you upload to the notebook)
Tell me when they're in, and I'll: query NotebookLM for a grounded synthesis → build the
Obsidian KB (`obsidian/CaoHoc/desfam-syscall-kb/`) → refresh `ReportV2` related-work (§2) and
discussion (§8) with recent SOTA + add the new bib entries → rebuild.

> Note: foundational method citations the report must keep regardless of age (VAE — Kingma &
> Welling 2013; Isolation Forest — Liu et al. 2008; the syscall-IDS lineage) stay in
> `references.bib`; these Docs are the *recent* related-work layer.
