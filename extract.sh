#!/bin/bash
# Extract each section of the DeSFAM paper from paper.txt into clean English LaTeX.
# Uses claude -p (Claude Code CLI) — no SDK or API key management needed.
# Output: latex/sections_en/*.tex  (English, IEEEtran-compatible LaTeX)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPER="$SCRIPT_DIR/paper.txt"
OUT="$SCRIPT_DIR/latex/sections_en"
mkdir -p "$OUT"

CLAUDE="claude -p --dangerously-skip-permissions --output-format text"

BASE_INSTRUCTIONS="You are a LaTeX expert. Your task is to extract ONE specific section \
from the DeSFAM IEEE paper (file: $PAPER) and format it as clean LaTeX source code \
compatible with the IEEEtran document class.

Rules:
- Output ONLY the LaTeX source for the requested section — no preamble, no \\begin{document}, no explanations.
- Use proper LaTeX environments: \\section, \\subsection, \\subsubsection for headings.
- Equations: use equation, align, or multline environments with \\label{eq:N}.
- Algorithms: use the algorithm + algpseudocode package environments.
- Tables: use tabular with booktabs (\\toprule, \\midrule, \\bottomrule).
- Code listings: use lstlisting environment.
- Figures: use \\begin{figure}[t]...\\end{figure} with \\caption and \\label. \
  Since we have no image files, write \\includegraphics[width=\\linewidth]{figN} as a placeholder.
- Mathematical notation: reproduce all formulas exactly from the paper.
- The source paper text was extracted from a two-column PDF and may be interleaved — \
  reconstruct the correct reading order.
- Use \\cite{refN} for citations (keep citation numbers from the paper, e.g. [1] → \\cite{ref1}).
- Do NOT invent content. If something is unclear in the extraction, reproduce it as best you can."

echo "==> Extracting sections from: $PAPER"
echo "==> Output directory: $OUT"
echo ""

run_extract() {
  local section_name="$1"
  local output_file="$2"
  local section_hint="$3"

  echo "--- Extracting: $section_name → $output_file"
  $CLAUDE "$BASE_INSTRUCTIONS

Extract the following section: $section_hint

Read the paper at: $PAPER" > "$OUT/$output_file"
  echo "    Done ($(wc -l < "$OUT/$output_file") lines)"
}

run_extract "Abstract" \
  "00_abstract.tex" \
  "The ABSTRACT section and INDEX TERMS keywords list. \
   For abstract: use the IEEEtran abstract environment \\begin{abstract}...\\end{abstract}. \
   For keywords: use \\begin{IEEEkeywords}...\\end{IEEEkeywords}."

run_extract "Section I - Introduction" \
  "01_introduction.tex" \
  "Section I: INTRODUCTION. \
   Start with \\section{Introduction}\\label{sec:intro}. \
   Include all subsections, the motivation, the list of contributions, and the paper organization paragraph."

run_extract "Section II - Background" \
  "02_background.tex" \
  "Section II: BACKGROUND. \
   Start with \\section{Background}\\label{sec:background}. \
   Include all subsections: containerization and the shared kernel, syscalls and the attack surface, \
   limitations of traditional mitigation approaches, eBPF for syscall monitoring and enforcement."

run_extract "Section III - Related Work" \
  "03_related_work.tex" \
  "Section III: RELATED WORK. \
   Start with \\section{Related Work}\\label{sec:related}. \
   Include all subsections: syscall filtering approaches, anomaly detection approaches, \
   provenance-based detection, kernel-level security enhancements, common limitations, \
   and the DeSFAM: Advancing the State of the Art subsection. \
   Include Table 1 (comparison of related work) as a LaTeX table."

run_extract "Section IV - DeSFAM Architecture" \
  "04_desfam.tex" \
  "Section IV: DESFAM: DYNAMIC EBPF-DRIVEN SYSCALL FILTERING AND ANOMALY MITIGATION. \
   Start with \\section{DeSFAM: Dynamic eBPF-Driven Syscall Filtering and Anomaly Mitigation}\\label{sec:desfam}. \
   Include ALL subsections covering: \
   (A) Overview and architecture, \
   (B) Phase 1 - Hybrid Syscall Access Listing (with Algorithm 1 as algorithm environment), \
   (C) Phase 2 - SyscallAD anomaly detection (VAE + Isolation Forest, with Algorithm 2, \
       all equations for VAE loss, anomaly score computation, sliding window logic), \
   (D) Phase 3 - Adaptive Policy Management (CVE parser, MITRE ATT&CK risk scoring, \
       tiered response, with Algorithm 3, all equations for risk score). \
   Reproduce ALL mathematical equations with proper numbering."

run_extract "Section V - Evaluation" \
  "05_evaluation.tex" \
  "Section V: EXPERIMENT EVALUATION AND RESULTS. \
   Start with \\section{Experiment Evaluation and Results}\\label{sec:evaluation}. \
   Include all subsections: experimental setup, dataset description (DongTing), \
   evaluation metrics, detection performance results, overhead analysis, \
   ablation study, comparison with baselines. \
   Include ALL tables with full data (detection performance table, overhead table, \
   ablation study table, comparison table)."

run_extract "Section VI - Discussion" \
  "06_discussion.tex" \
  "Section VI: DISCUSSION. \
   Start with \\section{Discussion}\\label{sec:discussion}. \
   Include all subsections covering strengths, limitations, and broader implications."

run_extract "Section VII - Conclusion" \
  "07_conclusion.tex" \
  "Section VII: CONCLUSION. \
   Start with \\section{Conclusion}\\label{sec:conclusion}. \
   Include the full conclusion text and any future work discussion."

run_extract "Appendix A" \
  "appendix_a.tex" \
  "APPENDIX A: the CVE-based rule parser for syscall extraction. \
   Start with \\appendix\\section{...}\\label{appendix:cve_parser}. \
   Include all subsections: scope and design principles, syscall dictionary construction \
   (with bash code listing), extraction and filtering rules (with Python regex code listing), \
   algorithm pseudocode (with Python code listing), practical filtering examples."

echo ""
echo "==> Extraction complete. Review files in: $OUT/"
ls -lh "$OUT/"
