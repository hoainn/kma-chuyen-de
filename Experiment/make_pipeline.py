#!/usr/bin/env python3
"""Draw the experiment pipeline (phases) → output/fig0_pipeline.png.
Shows the disjoint TRAIN / CALIBRATE / TEST phases so calibration is never
fitted on the windows used for evaluation streams."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
fig, ax = plt.subplots(figsize=(9.2, 7.4)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

def box(x, y, w, h, text, fc, ec="#333", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08",
                 linewidth=1.3, edgecolor=ec, facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2, style="-|>", c="#444", ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=14, color=c, lw=lw, linestyle=ls,
                 shrinkA=3, shrinkB=3))

C0,C1,C2,C3,CE,CS = "#e8eef7","#e7f0e7","#fbe9d6","#f7e6e6","#ede3f3","#eeeeee"
# top
box(5,9.4,3.2,0.8,"DongTing (npz)\nsyscall ID thực", CS, bold=True)
box(5,8.2,4.6,0.8,"PHASE 0 · DATA\nLọc sensor-alphabet (24 syscalls) + windowing (10/2)", C0, bold=True)
arrow(5,9.0,5,8.6)
# split lanes
xs=[2.1,5,7.9]
box(xs[0],6.9,2.7,0.9,"TRAIN\nnormal-train\n(50.877 cửa sổ)", CS)
box(xs[1],6.9,2.7,0.9,"CALIBRATION\n½ benign-val + ½ attack", CS)
box(xs[2],6.9,2.7,0.9,"TEST (rời rạc)\n½ benign-val + ½ attack", CS)
for x in xs: arrow(5,7.8,x,7.35)
# phase boxes
box(xs[0],5.2,2.7,1.15,"PHASE 1 · TRAIN\nFit Scaler + dense-AE\n+ IsolationForest\n(normal-only)", C1, bold=True)
box(xs[1],5.2,2.7,1.15,"PHASE 2 · CALIBRATE\n$T_0/T_{op}/T_{min}$\n+ band edges\n(noise & attack)", C2, bold=True)
box(xs[2],5.2,2.7,1.15,"PHASE 3 · TEST\nScore → AUC/AP=0.848\n+ điểm cửa sổ TEST", C3, bold=True)
for x in xs: arrow(x,6.45,x,5.8)
# models reused; calib params flow
arrow(xs[0]+1.35,5.2,xs[1]-1.35,5.2,ls="--",c="#2a7")   # models -> calibrate
arrow(xs[1]+1.35,5.2,xs[2]-1.35,5.2,ls="--",c="#2a7")   # models/score -> test
ax.text(5,5.62,"mô hình & scaler dùng lại (không huấn luyện lại)",ha="center",fontsize=7.5,color="#2a7")
# experiment
box(5,3.4,7.6,1.25,"THỬ NGHIỆM EMA-EVASION (RQ2)\n3 chính sách ngưỡng: fixed / ema_uncond / ema_cond\n× cường độ nhiễu {none, moderate, high} × loại {lén, ồn} · 40 trial/ô",
    CE, bold=True)
arrow(xs[2],4.62,6.0,4.05)               # TEST scores -> experiment
arrow(xs[1],4.62,4.2,4.05,ls="--",c="#b80")  # CAL params -> experiment
ax.text(3.0,4.35,"ngưỡng + band\n(từ CAL)",ha="center",fontsize=7,color="#b80")
ax.text(7.2,4.35,"điểm cửa sổ\n(từ TEST)",ha="center",fontsize=7,color="#444")
# outputs
box(5,1.5,7.0,0.95,"KẾT QUẢ\nrecall / evasion / $T_{EMA}(t)$ · Kruskal·Spearman·Wilcoxon · Hình 1–2", CS, bold=True)
arrow(5,2.78,5,1.98)
# leakage note
ax.text(5,0.35,"CALIBRATION ∩ TEST = ∅  →  ngưỡng/band không bao giờ được fit trên cửa sổ dùng để đánh giá (chống rò rỉ)",
        ha="center",fontsize=8.5,style="italic",color="#a00",
        bbox=dict(boxstyle="round,pad=0.3",fc="#fff3f3",ec="#a00",lw=1))
plt.title("Pipeline thực nghiệm — các pha tách biệt (train · calibrate · test)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig0_pipeline.png"), dpi=140, bbox_inches="tight"); plt.close()
print("OK -> output/fig0_pipeline.png")
