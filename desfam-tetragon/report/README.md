# Experience Report (LaTeX, English + Vietnamese)

Self-contained IEEEtran reports covering the DeSFAM reproduction on
DongTing and the live-cluster deployment via Cilium Tetragon. Mirrors
the parent paper's bilingual layout (`latex/sections_en/` + `latex/sections/`).

## Files

| File | Purpose |
|---|---|
| `main_en.tex`        | English entry — preamble, title, abstract, `\input{experience_en}` |
| `experience_en.tex`  | English body (intro → reproduction → deployment → experiment → reflection → conclusion) |
| `main_vi.tex`        | Vietnamese entry — same preamble + `\usepackage[vietnamese]{babel}`, `\input{experience_vi}` |
| `experience_vi.tex`  | Vietnamese body, mirrors the English structure section-by-section |
| `references.bib`     | Shared bibliography: DeSFAM paper + DongTing + Tetragon |
| `build.sh`           | One-shot Docker build via `texlive/texlive:latest`; builds both PDFs by default |

## Build

```bash
bash build.sh            # builds main_en.pdf and main_vi.pdf
bash build.sh en         # English only
bash build.sh vi         # Vietnamese only
```

If you have a local TeX Live (LuaLaTeX + IEEEtran):

```bash
latexmk -lualatex -interaction=nonstopmode main_en.tex
latexmk -lualatex -interaction=nonstopmode main_vi.tex
```

## Including in the parent paper instead

Either of the `experience_*.tex` files is `\input`-safe — they only use
standard sectioning commands and the `booktabs` + `listings` packages.
To attach the English version as an appendix to the parent English
paper at `../latex/main_en.tex`:

```latex
\input{../desfam-tetragon/report/experience_en}
```

For the Vietnamese parent paper (`../latex/main.tex`):

```latex
\input{../desfam-tetragon/report/experience_vi}
```
