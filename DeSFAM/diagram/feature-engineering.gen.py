"""Generator for feature-engineering.excalidraw.

Layout: numbered 7-step snake pipeline.
  Row 1 (left→right): Steps 1, 2, 3 (Step 3 expanded to show 6 feature groups).
  Row 2 (right→left): Steps 4, 5, 6, 7.

Run: python3 feature-engineering.gen.py > feature-engineering.excalidraw
"""
import json, itertools

_seed = itertools.count(300_000)
def n(): return next(_seed)

COMMON = {
    "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
    "roughness": 0, "opacity": 100, "angle": 0,
    "isDeleted": False, "groupIds": [], "boundElements": [],
    "link": None, "locked": False, "version": 1,
}

def rect(id, x, y, w, h, stroke, fill, dashed=False, sw=2):
    return {**COMMON, "type": "rectangle", "id": id, "x": x, "y": y,
            "width": w, "height": h,
            "strokeColor": stroke, "backgroundColor": fill,
            "strokeWidth": sw,
            "strokeStyle": "dashed" if dashed else "solid",
            "seed": n(), "versionNonce": n(),
            "roundness": {"type": 3}}

def text(id, x, y, w, h, t, container=None, size=18, color="#0f172a",
         align="center", valign="middle"):
    return {**COMMON, "type": "text", "id": id, "x": x, "y": y,
            "width": w, "height": h,
            "strokeColor": color, "backgroundColor": "transparent",
            "strokeWidth": 1,
            "seed": n(), "versionNonce": n(),
            "text": t, "originalText": t,
            "fontSize": size, "fontFamily": 3,
            "textAlign": align, "verticalAlign": valign,
            "containerId": container, "lineHeight": 1.25}

def arrow(id, x, y, dx, dy, color="#475569", sw=2):
    return {**COMMON, "type": "arrow", "id": id, "x": x, "y": y,
            "width": abs(dx), "height": abs(dy),
            "strokeColor": color, "backgroundColor": "transparent",
            "strokeWidth": sw,
            "seed": n(), "versionNonce": n(),
            "points": [[0, 0], [dx, dy]], "lastCommittedPoint": None,
            "startBinding": None, "endBinding": None,
            "startArrowhead": None, "endArrowhead": "arrow", "elbowed": False}

def step_card(id, x, y, w, h, step_num, title, body,
              accent="#1e40af", body_color="#0f172a"):
    """A 'card' with a coloured strip at the top showing STEP N, a bold title,
    and a body description below."""
    out = []
    # Outer box (white background, accent border)
    card = rect(id, x, y, w, h, accent, "#ffffff", sw=2)
    out.append(card)
    # Step badge strip
    badge_h = 32
    badge = rect(id + "_badge", x, y, w, badge_h, accent, accent, sw=0)
    badge["roundness"] = {"type": 3}
    out.append(badge)
    out.append(text(id + "_step", x + 12, y, w - 24, badge_h,
                    f"STEP {step_num}", size=14, color="#ffffff",
                    align="left"))
    # Title (centred, just under badge)
    title_h = 32
    out.append(text(id + "_title", x + 8, y + badge_h, w - 16, title_h,
                    title, size=17, color=accent, align="center"))
    # Body (multi-line, centred)
    body_y = y + badge_h + title_h
    body_h = h - badge_h - title_h
    out.append(text(id + "_body", x + 10, body_y + 4, w - 20, body_h - 8,
                    body, size=12, color=body_color, align="center"))
    return out

def feature_box(id, x, y, w, h, name, dims, blurb, accent="#0f172a",
                fill="#e2e8f0", is_new=False):
    """A small box for one feature group inside Step 3."""
    out = []
    out.append(rect(id, x, y, w, h, accent, fill, sw=2))
    # Top: name (e.g. "freq_60")
    out.append(text(id + "_n", x + 6, y + 6, w - 12, 24, name,
                    size=15, color=accent, align="center"))
    # Middle: dims line (e.g. "60 dims")
    out.append(text(id + "_d", x + 6, y + 32, w - 12, 18,
                    f"{dims} dims", size=11, color="#475569", align="center"))
    # Bottom: blurb
    out.append(text(id + "_b", x + 6, y + 52, w - 12, h - 58,
                    blurb, size=10, color="#1e293b", align="center"))
    if is_new:
        out.append(text(id + "_new", x + 6, y + h - 22, w - 12, 18,
                        "★ paper §IV.B.1", size=11,
                        color="#15803d", align="center"))
    return out

elements: list = []

# ── Title bar ────────────────────────────────────────────────────────────
elements.append(text("title", 200, 14, 1500, 40,
                     "DeSFAM Feature Engineering Pipeline — 7 Steps",
                     size=28, color="#1e40af"))
elements.append(text("subtitle", 200, 56, 1500, 22,
                     "From raw syscall stream to per-window anomaly decision",
                     size=14, color="#475569"))

# ── ROW 1 (left → right): Step 1, Step 2, Step 3 ─────────────────────────
ROW1_Y = 110

# STEP 1 — Ingest
elements += step_card("s1", 40, ROW1_Y, 220, 160, 1,
                      "Ingest syscall stream",
                      "Tetragon gRPC (live)\nor DongTing MongoDB\n[s₁, s₂, s₃, … sₙ]",
                      accent="#1e3a5f")

# Arrow + shape label between Step 1 and Step 2
elements.append(arrow("a12", 268, ROW1_Y + 80, 56, 0))
elements.append(text("lbl12", 268, ROW1_Y + 50, 56, 18,
                     "raw IDs", size=11, color="#64748b"))

# STEP 2 — Slide windows
elements += step_card("s2", 332, ROW1_Y, 220, 160, 2,
                      "Slide windows",
                      "length = 15\nstride = 3 (80 % overlap)\nemit window every 3 syscalls",
                      accent="#7c2d12")

# Arrow + shape label between Step 2 and Step 3
elements.append(arrow("a23", 560, ROW1_Y + 80, 56, 0))
elements.append(text("lbl23", 560, ROW1_Y + 50, 56, 18,
                     "W × 15", size=11, color="#64748b"))

# STEP 3 — Feature Engineering (wide container)
S3_X, S3_Y, S3_W, S3_H = 624, ROW1_Y, 1136, 280
elements.append(rect("s3_outer", S3_X, S3_Y, S3_W, S3_H,
                     "#1e40af", "#ffffff", sw=2))
# Top badge
elements.append(rect("s3_badge", S3_X, S3_Y, S3_W, 32,
                     "#1e40af", "#1e40af", sw=0))
elements.append(text("s3_step", S3_X + 12, S3_Y, S3_W - 24, 32,
                     "STEP 3   ·   Feature Engineering — per-window 159-dim vector",
                     size=15, color="#ffffff", align="left"))
elements.append(text("s3_sub", S3_X + 8, S3_Y + 36, S3_W - 16, 22,
                     "for each window, compute 6 feature groups in parallel and concatenate",
                     size=13, color="#1e40af", align="center"))

# 6 feature group boxes inside Step 3 (single row)
FG_Y, FG_H, FG_W = S3_Y + 70, 180, 175
fg_gap = (S3_W - (FG_W * 6)) // 7
fg_specs = [
    ("freq60",   "freq_60",     60,
     "top-60 syscall\nfrequencies\n(count / win_len)",
     "#0f172a", "#e2e8f0", False),
    ("disc40",   "disc_40",     40,
     "discriminative-40\npresence flags\n(0 / 1)",
     "#0f172a", "#e2e8f0", False),
    ("stats8",   "stats_8",      8,
     "entropy, log-len,\nmax-freq, p75,\nstd, coverage, …",
     "#0f172a", "#e2e8f0", False),
    ("bigram40", "bigrams_40",  40,
     "top-40 syscall\nbigram frequencies\n(co-occurrence)",
     "#0f172a", "#e2e8f0", False),
    ("ver1",     "ver_1",        1,
     "kernel version\none-hot\n(DongTing → 5.12)",
     "#0f172a", "#e2e8f0", False),
    ("cat10",    "cat_10",      10,
     "functional category\nfrequencies\n(file / mem / net / …)",
     "#166534", "#bbf7d0", True),
]
for i, (id_, name, dims, blurb, stroke, fill, is_new) in enumerate(fg_specs):
    fx = S3_X + fg_gap + i * (FG_W + fg_gap)
    elements += feature_box(id_, fx, FG_Y, FG_W, FG_H, name, dims, blurb,
                            accent=stroke, fill=fill, is_new=is_new)

# ── Down arrow from Step 3 to Row 2 ──────────────────────────────────────
# Step 3 bottom-centre → Step 4 top-centre
S3_BOT_X = S3_X + S3_W // 2
S3_BOT_Y = S3_Y + S3_H
ROW2_Y = 470
elements.append(arrow("a3_4", S3_BOT_X, S3_BOT_Y + 4, 0, ROW2_Y - S3_BOT_Y - 8))
elements.append(text("lbl34", S3_BOT_X + 12, S3_BOT_Y + 30, 180, 18,
                     "concat → W × 159", size=11, color="#64748b", align="left"))

# ── ROW 2 (right → left): Step 4, 5, 6, 7 ────────────────────────────────
# Step 4 directly under Step 3's centre
elements += step_card("s4", S3_BOT_X - 110, ROW2_Y, 220, 160, 4,
                      "Concatenate vector",
                      "x ∈ ℝ¹⁵⁹\nper window\nfeature_groups → row",
                      accent="#1e40af")

# Step 5
S5_X = S3_BOT_X - 110 - 280
elements += step_card("s5", S5_X, ROW2_Y, 220, 160, 5,
                      "Robust scale",
                      "RobustScaler(p1, p99)\nfit on normal-only train\nremove outlier influence",
                      accent="#7c3aed")
elements.append(arrow("a4_5", S3_BOT_X - 114, ROW2_Y + 80, -56, 0))
elements.append(text("lbl45", S3_BOT_X - 168, ROW2_Y + 50, 56, 18,
                     "W × 159", size=11, color="#64748b"))

# Step 6
S6_X = S5_X - 280
elements += step_card("s6", S6_X, ROW2_Y, 220, 160, 6,
                      "Score (ensemble)",
                      "α · VAE_err + (1−α) · IF\nα = 0.7\n→ anomaly score",
                      accent="#b91c1c")
elements.append(arrow("a5_6", S5_X - 4, ROW2_Y + 80, -56, 0))
elements.append(text("lbl56", S5_X - 58, ROW2_Y + 50, 56, 18,
                     "scaled", size=11, color="#64748b"))

# Step 7
S7_X = S6_X - 280
elements += step_card("s7", S7_X, ROW2_Y, 220, 160, 7,
                      "Decide & alert",
                      "score > threshold\n→ raise alert\n(ALERT_CONSECUTIVE)",
                      accent="#92400e")
elements.append(arrow("a6_7", S6_X - 4, ROW2_Y + 80, -56, 0))
elements.append(text("lbl67", S6_X - 58, ROW2_Y + 50, 56, 18,
                     "W × 1", size=11, color="#64748b"))

# Final alert callout to the left of Step 7
ALERT_X = S7_X - 230
elements.append(rect("alert", ALERT_X, ROW2_Y + 30, 200, 100, "#b91c1c", "#fee2e2", sw=2))
elements.append(text("alert_t", ALERT_X + 8, ROW2_Y + 40, 184, 80,
                     "ALERT\nProcess flagged as\nanomalous syscall\nbehaviour",
                     size=14, color="#b91c1c", align="center"))
elements.append(arrow("a7_alert", S7_X - 4, ROW2_Y + 80, -26, 0,
                      color="#b91c1c", sw=3))

# ── Bottom panel: paper components NOT implemented ───────────────────────
NI_Y = 670
elements.append(rect("ni_bg", 40, NI_Y, 1720, 180, "#92400e", "#fef3c7",
                     dashed=True, sw=2))
elements.append(text("ni_title", 60, NI_Y + 12, 1680, 28,
                     "Paper §IV.B.1 components NOT implemented in this replication",
                     size=20, color="#7c2d12", align="left"))
elements.append(text("ni_l1", 60, NI_Y + 52, 1680, 26,
                     "✗  Temporal Features (Δt mean / std / max)  —  DongTing stores raw syscall IDs without timestamps; not recoverable from the dataset.",
                     size=14, color="#0f172a", align="left"))
elements.append(text("ni_l2", 60, NI_Y + 88, 1680, 26,
                     "✗  Access List Pattern Matching (PrefixSpan)  —  deferred to Phase 2; high engineering cost, orthogonal to the main detector.",
                     size=14, color="#0f172a", align="left"))
elements.append(text("ni_l3", 60, NI_Y + 124, 1680, 26,
                     "✗  Data Augmentation of attack sequences (clone / setuid insertion)  —  incompatible with the normal-only training paradigm used by IF + VAE.",
                     size=14, color="#0f172a", align="left"))

# ── Side note: feature heritage legend ───────────────────────────────────
LG_X, LG_Y = 40, 380
elements.append(rect("lg_bg", LG_X, LG_Y, 480, 90, "#475569", "#f8fafc", sw=1))
elements.append(text("lg_t", LG_X + 8, LG_Y + 6, 464, 22,
                     "Feature heritage", size=13, color="#1e293b", align="left"))
elements.append(rect("lg_s1", LG_X + 12, LG_Y + 34, 16, 16, "#0f172a", "#e2e8f0"))
elements.append(text("lg_s1t", LG_X + 36, LG_Y + 32, 440, 18,
                     "carried over from the released train.py (pre-paper design)",
                     size=11, color="#1e293b", align="left"))
elements.append(rect("lg_s2", LG_X + 12, LG_Y + 58, 16, 16, "#166534", "#bbf7d0"))
elements.append(text("lg_s2t", LG_X + 36, LG_Y + 56, 440, 18,
                     "★ added in this replication to match paper §IV.B.1",
                     size=11, color="#1e293b", align="left"))

doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {"gridSize": 20, "viewBackgroundColor": "#ffffff"},
    "files": {},
}
print(json.dumps(doc, indent=2, ensure_ascii=False))
