# 09 — Corpus tài liệu (để import NotebookLM + làm giàu báo cáo)

Lược khảo có hệ thống cho đề tài *phát hiện leo thang đặc quyền trong multi-tenant Kubernetes dựa
system call*. **Ưu tiên nguồn chất lượng & MỚI (2024–2026).** citekey khớp 1–1 với
`../Reference/README.md` và `archive/paper/src/references.bib`. PDF tải sẵn trong `../Reference/`.

## Tiêu chí lựa chọn (inclusion)
- **Chủ đề**: phát hiện bất thường/xâm nhập dựa **system call**; eBPF runtime security; privilege
  escalation / container escape; anomaly detection cho container/cloud/K8s; robustness/evasion.
- **Thời gian**: **ưu tiên 2024–2026** (≈2 năm gần nhất) làm lõi trích dẫn; 2023 = "cận mới" giữ nếu
  giá trị cao; <2020 chỉ giữ khi **nền tảng bắt buộc** (Forrest 1996, Creech 2014, VAE, iForest).
- **Chất lượng**: ưu tiên **peer-reviewed** (USENIX, IEEE, ACM, Elsevier, Springer, Nature); arXiv chỉ
  bổ sung SOTA mới nhất và **phải ghi rõ là preprint**.

## Corpus (import-ready)

### Tier 0 — MỚI 2024–2026, peer-reviewed (TRÍCH DẪN CHÍNH)
| # | Paper | Venue (peer-reviewed) | DOI | Dùng cho | citekey |
|---|---|---|---|---|---|
| 1 | **Privilege Escalation Detection & Prediction via eBPF + ML** (Li et al.) | **IEEE CCC 2025** | 10.23919/CCC64809.2025.11178545 | **đúng trọng tâm**: privesc + eBPF + ML — peer gần nhất | `li2025privescebpf` |
| 2 | **DeSFAM**: Adaptive eBPF + AI framework (Zehra et al.) | **IEEE Access 2025** | 10.1109/ACCESS.2025.3592192 | **cơ chế cơ sở** (SyscallAD = VAE+iForest) | `desfam_original` |
| 3 | **eBPF-Guard**: container escape detection, multi-level monitoring (Lin et al.) | **Empirical Software Eng.** (Springer) 2026 | 10.1007/s10664-025-10784-1 | phát hiện **container escape** (paywalled) | `ebpfguard2026` |
| 4 | **EIDS**: cloud IDS, high performance & maintainability (Hu et al.) | **ACM Trans. Computer Systems** 2026 | 10.1145/3787966 | so sánh với Falco; IDS cloud hiệu năng cao (paywalled) | `hu2026eids` |
| 5 | **Adaptive Runtime Security for Kubernetes** via eBPF + ML (Joshi & Segireddy) | **IEEE ICOECA 2026** | 10.1109/ICOECA68095.2026.11485567 | eBPF telemetry + ML cho K8s runtime (paywalled) | `joshi2026adaptivek8s` |
| 6 | **Anomaly Detection in Containerized Systems** (syscall histogram + autoencoder) (Kotenko & Melnik) | **IEEE EDM 2025** | 10.1109/EDM65517.2025.11096840 | autoencoder + syscall trong container (gần DeSFAM) (paywalled) | `kotenko2025containerae` |
| 7 | **Real-time multi-class threat detection & adaptive deception in K8s** (Aly et al.) | **Scientific Reports** (Nature) 2025 | 10.1038/s41598-025-91606-8 | đa lớp đe doạ + deception trong K8s (open access, PDF) | `aly2025k8sthreat` |
| 8 | **Comparative Analysis of eBPF Runtime Security Monitoring Tools on K8s** (Syairozi & Arizal) | **SciTePress RITECH 2025** | 10.5220/0014272700004928 | đối chiếu Falco/Tetragon/Tracee (PDF) | `syairozi2025ebpfcompare` |
| 9 | **LIGHT-HIDS**: lightweight ML host IDS (Gungor et al.) | **IEEE ICMLA 2025** | arXiv:2509.13464 | syscall HIDS + **novelty one-class** (như SyscallAD) | `lighthids2025` |
| 10 | **SafeBPF**: hardware-assisted defense-in-depth for eBPF (Lim et al.) | **ACM CCSW 2024** | 10.1145/3689938.3694781 | an ninh chính eBPF | `safebpf2024` |
| 11 | Transformers & LLMs for Efficient IDS: A Survey (Kheddar) | **Information Fusion** (Elsevier) 2025 | 10.1016/j.inffus.2025.103347 | survey seq-model IDS | `translmidssurvey` |

### Tier 1 — Cận mới (2023) & nền tảng peer-reviewed (bổ trợ, không phải lõi "mới")
| Paper | Venue | citekey |
|---|---|---|
| Cross Container Attacks: The Bewildered eBPF on Clouds (He et al.) | **USENIX Security 2023** | `crosscontainer2023` |
| Container Privilege Escalation & Escape Detection (security-first arch.) (Zhou et al.) | IEEE HPCC/DSS 2023 | `zhou2023cpeed` |
| **DongTing**: dataset cho Linux kernel anomaly detection (Duan et al.) | J. Systems & Software (Elsevier) 2023 | `ref26_dongtingdataset` |
| A Survey of IDS Leveraging Host Data (Bridges et al.) | ACM Comput. Surv. 2019 | `bridges2019hostdata` |
| HIDS with System Calls: Review & Future Trends (Liu et al.) | ACM Comput. Surv. 2018 | `liu2018hids` |
| Forrest et al., *A Sense of Self for Unix Processes* | IEEE S&P 1996 | `forrest1996sense` |
| Creech & Hu, semantic HIDS (ADFA-LD) | IEEE Trans. Computers 2014 | `creech2014semantic` |
| Kingma & Welling, *Auto-Encoding Variational Bayes* (VAE) | ICLR 2014 | `vae_kingma` |
| Liu, Ting, Zhou, *Isolation Forest* | IEEE ICDM 2008 | `iforest_liu` |
| Edgar & Manz, *Research Methods for Cyber Security* | Syngress 2017 (PP luận — bắt buộc) | — |

### Tier 2 — Preprint arXiv (recent nhưng CHƯA bình duyệt; ghi rõ "preprint")
| # | Paper | id | Dùng cho | citekey |
|---|---|---|---|---|
| 12 | eBPF-PATROL: runtime threat recognition & overreach limit | arXiv:2511.18155 | eBPF **privesc/container escape** | `ebpfpatrol2025` |
| 13 | FedMon: Federated eBPF monitoring (nhóm DeSFAM) | arXiv:2510.10126 | eBPF anomaly đa cluster (multi-tenant) | `fedmon2025` |
| 14 | Programmable System Call Security with eBPF (Jia et al.) | arXiv:2302.10366 | cưỡng chế an ninh syscall (vs seccomp) | `progsyscall2023` |
| 15 | Language Models for Novelty Detection in System Call Traces | arXiv:2309.02206 | LM novelty detection trên syscall | `lmnovelty2023` |
| 16 | Adversarial-Robust Behavior-Sequence Anomaly Detection | arXiv:2509.15756 | **né tránh/đối kháng** (Discussion) | `advrobustseq2025` |
| 17 | Deep Learning-based IDS: A Survey | arXiv:2504.07839 | neo related-work | `dlidssurvey2025` |

## Quy trình import + dựng báo cáo
1. **Import** PDF Tier 0 trước (lõi mới), sau đó Tier 1/2 chọn lọc, vào NotebookLM (notebook
   *"DeSFAM — Syscall Privilege-Escalation Detection"*). Ưu tiên #1–8.
2. Chờ NotebookLM xử lý xong (icon source sáng).
3. **Query grounded**: (a) đối chiếu cách tiếp cận eBPF+ML mới nhất; (b) trích gap (privesc live,
   multi-tenant, dataset↔vận hành); (c) thước đo & robustness; (d) định vị đóng góp.
4. **Tổng hợp** vào báo cáo; citekey đã có sẵn trong `archive/paper/src/references.bib`.

> Lưu ý chặt chẽ: **ưu tiên Tier 0 (2024–2026)**; Tier 1 chỉ bổ trợ/nền tảng; preprint (Tier 2) phải
> ghi rõ "preprint chưa bình duyệt". Với paper có cả preprint lẫn xuất bản, **trích bản xuất bản
> (DOI)**. Chỉ đưa claim **đã đọc/kiểm chứng**; **KHÔNG** dùng headline tổng hợp DeSFAM (AUC≈0.94…)
> làm mục tiêu per-model (CLAUDE.md).
