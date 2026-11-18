"""Architecture and pipeline diagrams for the hyperparameter posts.

    python blog/diagrams/diagrams.py        # writes the SVGs next to this script

Everything drawn here mirrors the shared code cell of the *-colab.ipynb notebooks
(make_model, run, sweep). Change the code there, change the drawing here.
Model colours match the experiment figures: MLP blue, CNN orange, ResNet-18 green.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).parent

BG, INK, DIM, RULE, PANEL = "#fcfcfb", "#1d232b", "#5b6470", "#c9d1da", "#f2f4f6"
MLP, CNN, RES = "#2a78d6", "#eb6834", "#1baf7a"
BN_FILL = "#e2f3eb"           # boxes that contain BatchNorm
SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, DejaVu Sans Mono, monospace"
RED = "#b3261e"
MARKERS = {"": DIM, "-ink": INK, "-res": RES, "-red": RED}   # arrowhead colours: url(#arrow-res) etc.


def svg(w, h, body, title):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{escape(title)}">
<title>{escape(title)}</title>
<defs>
{"".join(f'<marker id="arrow{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for k, c in MARKERS.items())}
</defs>
<rect width="{w}" height="{h}" fill="{BG}"/>
{body}
</svg>
"""


def text(x, y, s, size=13, color=INK, anchor="middle", weight="normal", font=SANS, style="normal"):
    keep = ' xml:space="preserve"' if font == MONO else ""
    return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}"{keep}>{escape(s)}</text>')


def box(x, y, w, h, label, sub="", fill="#ffffff", stroke=RULE, dashed=False, label_color=INK, size=13):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.3"{dash}/>']
    if sub:
        out.append(text(x + w / 2, y + h / 2 - 3, label, size, label_color))
        out.append(text(x + w / 2, y + h / 2 + 13, sub, 11, DIM, font=MONO))
    else:
        out.append(text(x + w / 2, y + h / 2 + 4.5, label, size, label_color))
    return "\n".join(out)


def arrow(x1, y1, x2, y2, color=DIM, dashed=False):
    dash = ' stroke-dasharray="4 4"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4"{dash} '
            f'marker-end="url(#arrow)"/>')


def path_arrow(d, color=DIM):
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.4" marker-end="url(#arrow)"/>'


# ---------------------------------------------------------------------------------------
# 1. The three models, side by side (make_model)
# ---------------------------------------------------------------------------------------
def models():
    W, H, colw = 960, 800, 270
    cols = [
        ("MLP", MLP, 30, [
            ("Input", "c × s × s", "io"),
            ("Flatten", "c·s²", ""),
            ("Linear + ReLU", "c·s² → 512", ""),
            ("Dropout(p)", "", "drop"),
            ("Linear + ReLU", "512 → 256", ""),
            ("Dropout(p)", "", "drop"),
            ("Linear", "256 → n", ""),
            ("Class scores", "n", "io"),
        ], ["KMNIST   535,818", "EuroSAT 6,425,866", "USPS     265,482"]),
        ("CNN", CNN, 345, [
            ("Input", "c × s × s", "io"),
            ("Conv 3×3 + ReLU", "c → 32, padding 1", ""),
            ("MaxPool 2×2", "32 × s/2 × s/2", ""),
            ("Conv 3×3 + ReLU", "32 → 64, padding 1", ""),
            ("MaxPool 2×2", "64 × s/4 × s/4", ""),
            ("Flatten", "64·(s/4)²", ""),
            ("Dropout(p)", "", "drop"),
            ("Linear + ReLU", "64·(s/4)² → 128", ""),
            ("Dropout(p)", "", "drop"),
            ("Linear", "128 → n", ""),
            ("Class scores", "n", "io"),
        ], ["KMNIST     421,642", "EuroSAT 2,117,962", "USPS       151,306"]),
        ("ResNet-18", RES, 660, [
            ("Input", "c × s × s", "io"),
            ("Conv 7×7 /2 + BN + ReLU", "c → 64", "bn"),
            ("MaxPool 3×3 /2", "64 channels", ""),
            ("layer1: 2 × BasicBlock", "64 channels", "bn"),
            ("layer2: 2 × BasicBlock /2", "128 channels", "bn"),
            ("layer3: 2 × BasicBlock /2", "256 channels", "bn"),
            ("layer4: 2 × BasicBlock /2", "512 channels", "bn"),
            ("Global average pool", "512", ""),
            ("Dropout(p)", "", "drop"),
            ("Linear", "512 → n", ""),
            ("Class scores", "n", "io"),
        ], ["KMNIST  11,175,370", "EuroSAT 11,181,642", "USPS    11,175,370"]),
    ]
    body, top, step, bh = [], 78, 52, 38
    for name, color, x, layers, params in cols:
        body.append(f'<rect x="{x}" y="26" width="{colw}" height="5" rx="2" fill="{color}"/>')
        body.append(text(x, 56, name, 17, INK, "start", "bold"))
        for i, (label, sub, kind) in enumerate(layers):
            y = top + i * step
            if kind == "drop":
                body.append(box(x + 40, y + 4, colw - 80, bh - 8, label, dashed=True, stroke=INK, size=12))
            elif kind == "bn":
                body.append(box(x, y, colw, bh, label, sub, fill=BN_FILL, stroke=RES))
            elif kind == "io":
                body.append(box(x + 40, y, colw - 80, bh, label, sub, fill=PANEL))
            else:
                body.append(box(x, y, colw, bh, label, sub))
            if i:   # previous box's bottom edge to this box's top edge (dropout boxes are inset 4px)
                prev_bottom = y - step + bh - (4 if layers[i - 1][2] == "drop" else 0)
                top_edge = y + (4 if kind == "drop" else 0)
                body.append(arrow(x + colw / 2, prev_bottom + 1, x + colw / 2, top_edge - 1))
        py = top + 11 * step + 4
        body.append(text(x, py, "parameters (n = 10)", 11, DIM, "start"))
        for j, p in enumerate(params):
            body.append(text(x, py + 18 + j * 16, p, 11, INK, "start", font=MONO))
    ly = H - 34
    body.append(f'<rect x="30" y="{ly - 13}" width="34" height="18" rx="4" fill="#fff" stroke="{INK}" stroke-dasharray="5 4"/>')
    body.append(text(72, ly, "Dropout(p): identity when p = 0, which is every run except the dropout sweep", 12, DIM, "start"))
    body.append(f'<rect x="590" y="{ly - 13}" width="34" height="18" rx="4" fill="{BN_FILL}" stroke="{RES}"/>')
    body.append(text(632, ly, "contains BatchNorm", 12, DIM, "start"))
    body.append(text(W - 30, ly + 22, "c channels, s × s pixels, n classes", 11, DIM, "end"))
    return svg(W, H, "\n".join(body), "The three models: MLP, small CNN and ResNet-18, with dropout and BatchNorm positions")


# ---------------------------------------------------------------------------------------
# 2. ResNet-18 in detail: stages, spatial sizes per dataset, the BasicBlock
# ---------------------------------------------------------------------------------------
def resnet():
    W, H = 1000, 560
    stages = [("Input", "c ch"), ("Stem", "conv 7×7 /2"), ("MaxPool", "3×3 /2"), ("layer1", "64 ch"),
              ("layer2", "128 ch, /2"), ("layer3", "256 ch, /2"), ("layer4", "512 ch, /2"),
              ("AvgPool", "global"), ("Dropout", "p"), ("Linear", "512 → n")]
    bn = {"Stem", "layer1", "layer2", "layer3", "layer4"}
    x0, bw, gap, y = 20, 86, 11.5, 60
    body = [text(20, 34, "ResNet-18 as built by make_model(): torchvision resnet18, stem widened to c input channels, dropout before the classifier",
                 13, INK, "start", "bold")]
    for i, (name, sub) in enumerate(stages):
        x = x0 + i * (bw + gap)
        if name == "Dropout":
            body.append(box(x, y, bw, 46, name, sub, dashed=True, stroke=INK))
        elif name in bn:
            body.append(box(x, y, bw, 46, name, sub, fill=BN_FILL, stroke=RES))
        else:
            body.append(box(x, y, bw, 46, name, sub, fill=PANEL if name == "Input" else "#fff"))
        if i:
            body.append(arrow(x - gap + 1, y + 23, x - 1, y + 23))
    # spatial sizes, computed as s -> floor((s - 1) / 2) + 1 for every stride-2 step
    rows = [("KMNIST", [28, 14, 7, 7, 4, 2, 1, 1]), ("USPS", [16, 8, 4, 4, 2, 1, 1, 1]),
            ("EuroSAT", [64, 32, 16, 16, 8, 4, 2, 1])]
    body.append(text(20, 138, "spatial size after each stage (height = width)", 11, DIM, "start"))
    for r, (ds, sizes) in enumerate(rows):
        yy = 162 + r * 22
        for i, s in enumerate(sizes):
            x = x0 + i * (bw + gap) + bw / 2
            label = f"{ds} {s}" if i == 0 else str(s)
            color = RED if 3 <= i <= 6 and sizes[i - 1] == 1 else INK   # a conv stage fed a 1x1 map
            body.append(text(x, yy, label, 12, color, font=MONO))
    body.append(text(20, 234, "red: the stage is fed a 1 × 1 map, so its 3 × 3 kernels (padding 1) only ever use the centre tap", 11, RED, "start"))

    # BasicBlock
    by = 300
    body.append(f'<rect x="20" y="{by - 40}" width="{W - 40}" height="{H - by + 20}" rx="10" fill="{PANEL}" stroke="{RULE}"/>')
    body.append(text(40, by - 14, "BasicBlock (8 of them: 2 per stage)", 14, INK, "start", "bold"))
    chain = [("Conv 3×3", "stride 1 or 2", "#fff", RULE), ("BatchNorm", "", BN_FILL, RES), ("ReLU", "", "#fff", RULE),
             ("Conv 3×3", "stride 1", "#fff", RULE), ("BatchNorm", "", BN_FILL, RES)]
    cx, cy, cw, ch, cg = 110, by + 70, 106, 46, 22
    body.append(text(60, cy + 28, "x", 16, INK, style="italic"))
    body.append(arrow(70, cy + 23, cx - 2, cy + 23))
    for i, (lab, sub, fill, stroke) in enumerate(chain):
        x = cx + i * (cw + cg)
        body.append(box(x, cy, cw, ch, lab, sub, fill=fill, stroke=stroke))
        if i:
            body.append(arrow(x - cg + 1, cy + 23, x - 1, cy + 23))
    px = cx + 5 * (cw + cg) + 4
    body.append(arrow(px - cg + 4, cy + 23, px - 1, cy + 23))
    body.append(f'<circle cx="{px + 14}" cy="{cy + 23}" r="14" fill="#fff" stroke="{INK}" stroke-width="1.4"/>')
    body.append(text(px + 14, cy + 29, "+", 18, INK))
    body.append(arrow(px + 29, cy + 23, px + 58, cy + 23))
    body.append(box(px + 60, cy, 80, ch, "ReLU"))
    body.append(arrow(px + 141, cy + 23, px + 168, cy + 23))
    body.append(text(px + 180, cy + 28, "out", 13, INK))
    # skip connection over the top
    body.append(path_arrow(f"M 90 {cy + 23} L 90 {cy - 30} L {px + 14} {cy - 30} L {px + 14} {cy + 7}"))
    body.append(text((90 + px) / 2, cy - 38, "shortcut: identity, or Conv 1×1 /2 + BatchNorm in the first block of layer2–4 (where the shape changes)", 12, DIM))
    # BN note
    ny = cy + 90
    body.append(text(40, ny, "BatchNorm, per channel:", 13, INK, "start", "bold"))
    body.append(text(205, ny, "y = γ · (x − μ) / √(σ² + ε) + β", 13, INK, "start", font=MONO))
    body.append(text(40, ny + 24, "training (model.train()): μ, σ² are the mean and variance of the current mini-batch over N × H × W positions,", 12, DIM, "start"))
    body.append(text(40, ny + 42, "so the batch size B sets their sample size. Evaluation (model.eval()): running averages collected during training.", 12, DIM, "start"))
    body.append(text(40, ny + 66, "20 BatchNorm layers in all. The MLP and the CNN have none.", 12, DIM, "start"))
    return svg(W, H, "\n".join(body), "ResNet-18 stages, spatial sizes per dataset, and the BasicBlock")


# ---------------------------------------------------------------------------------------
# 3. The experiment harness (load / holdout / to_device, sweep, run, figures + tables)
# ---------------------------------------------------------------------------------------
def harness():
    W, H = 1000, 520
    body = []
    # data column
    dx, dw = 20, 210
    body.append(text(dx, 30, "1 · Data, prepared once", 14, INK, "start", "bold"))
    data = [("torchvision dataset", "KMNIST · EuroSAT · USPS"), ("Decode to tensors", "load()"),
            ("Fixed split, seed 42", "holdout()"), ("Normalise per channel", "train mean / std"),
            ("Resident on the GPU", "train · val · test")]
    for i, (a, b) in enumerate(data):
        y = 48 + i * 74
        body.append(box(dx, y, dw, 46, a, b, fill=PANEL if i == 4 else "#fff"))
        if i:
            body.append(arrow(dx + dw / 2, y - 27, dx + dw / 2, y - 1))
    body.append(text(dx, 440, "No DataLoader in the training loop:", 11, DIM, "start"))
    body.append(text(dx, 456, "batches are index slices of GPU tensors,", 11, DIM, "start"))
    body.append(text(dx, 472, "so the timer measures compute only.", 11, DIM, "start"))

    # sweep panel
    sx, sw = 270, 440
    body.append(f'<rect x="{sx}" y="44" width="{sw}" height="456" rx="10" fill="{PANEL}" stroke="{RULE}"/>')
    body.append(text(sx, 30, "2 · sweep(param, values)", 14, INK, "start", "bold"))
    body.append(text(sx + 18, 70, "for value in grid  ×  model in (MLP, CNN, ResNet-18)  ×  seed in (0, 1, 2):", 12, INK, "start", font=MONO))
    body.append(text(sx + 18, 96, "run(model, seed, epochs, param=value)", 12, DIM, "start", font=MONO))
    rx, rw = sx + 40, sw - 80
    steps = [("torch.manual_seed(seed); make_model()", "same initial weights for every value", "#fff", RULE, False),
             ("Shuffle with a seeded generator", "same batch order for every value", "#fff", RULE, False),
             ("SGD, momentum 0.9, constant η", "last partial batch dropped", "#fff", RULE, False),
             ("model.eval(): score two sets", "train probe (first 5,000) · validation", "#fff", RULE, False),
             ("Append one row per epoch", "loss, accuracy, ‖θ‖, training seconds", "#fff", RULE, False)]
    for i, (a, b, fill, stroke, dashed) in enumerate(steps):
        y = 116 + i * 66
        body.append(box(rx, y, rw, 44, a, b, fill=fill, stroke=stroke, dashed=dashed))
        if i:
            body.append(arrow(rx + rw / 2, y - 21, rx + rw / 2, y - 1))
    # epoch loop arrow from the last step back to the shuffle step
    ly1, ly2 = 116 + 4 * 66 + 22, 116 + 66 + 22
    body.append(path_arrow(f"M {rx + rw} {ly1} L {rx + rw + 22} {ly1} L {rx + rw + 22} {ly2} L {rx + rw + 2} {ly2}"))
    body.append(text(rx + rw + 30, (ly1 + ly2) / 2 + 4, "× epochs", 11, DIM, "start"))
    # timer bracket on the SGD step
    ty = 116 + 2 * 66
    body.append(f'<path d="M {rx - 8} {ty} L {rx - 14} {ty} L {rx - 14} {ty + 44} L {rx - 8} {ty + 44}" fill="none" stroke="{INK}" stroke-width="1.2"/>')
    body.append(text(rx - 20, ty + 26, "timed", 11, INK, "end"))
    body.append(box(rx, 116 + 5 * 66 - 4, rw, 40, "After the last epoch: test accuracy, once", fill="#fff", stroke=INK))
    body.append(arrow(rx + rw / 2, 116 + 4 * 66 + 45, rx + rw / 2, 116 + 5 * 66 - 5))
    body.append(arrow(dx + dw + 2, 48 + 4 * 74 + 23, sx - 2, 48 + 4 * 74 + 23))

    # outputs column
    ox, ow = 760, 220
    body.append(text(ox, 30, "3 · Outputs", 14, INK, "start", "bold"))
    body.append(box(ox, 160, ow, 46, "runs.csv", "one row per run × epoch", fill=PANEL))
    body.append(arrow(sx + sw + 2, 183, ox - 2, 183))
    body.append(box(ox, 250, ow, 46, "plot_final · plot_curves", "figures/<slug>/*.png"))
    body.append(box(ox, 330, ow, 46, "table · best", "figures/<slug>/results.md"))
    body.append(arrow(ox + ow / 2, 207, ox + ow / 2, 249))
    body.append(path_arrow(f"M {ox + ow - 14} 207 L {ox + ow - 14} 329"))
    body.append(text(ox, 410, "best(): each model's best value is", 11, DIM, "start"))
    body.append(text(ox, 426, "picked on mean validation accuracy;", 11, DIM, "start"))
    body.append(text(ox, 442, "test accuracy is reported for it,", 11, DIM, "start"))
    body.append(text(ox, 458, "never used to choose.", 11, DIM, "start"))
    return svg(W, H, "\n".join(body), "Experiment harness: data preparation, the seeded sweep, and the outputs")


# ---------------------------------------------------------------------------------------
# 4. Why BatchNorm'd weights tolerate a large learning rate
# ---------------------------------------------------------------------------------------
def scale_invariance():
    W, H = 860, 380
    ox, oy = 70, 320
    wt = (ox + 240, oy - 180)                 # w_t
    step = (72, 96)                           # perpendicular to w_t, length 120
    wn = (wt[0] + step[0], wt[1] + step[1])   # w_{t+1}
    body = [text(20, 30, "One SGD step on weights that feed a BatchNorm layer", 14, INK, "start", "bold")]
    body.append(f'<circle cx="{ox}" cy="{oy}" r="3.5" fill="{INK}"/>')
    body.append(text(ox - 12, oy + 18, "0", 12, DIM))
    body.append(f'<path d="M {ox} {oy} L {wt[0]} {wt[1]} L {wn[0]} {wn[1]} Z" fill="{BN_FILL}" opacity="0.6"/>')
    body.append(f'<line x1="{ox}" y1="{oy}" x2="{wt[0]}" y2="{wt[1]}" stroke="{INK}" stroke-width="2.2" marker-end="url(#arrow-ink)"/>')
    body.append(f'<line x1="{ox}" y1="{oy}" x2="{wn[0]}" y2="{wn[1]}" stroke="{RES}" stroke-width="2.2" marker-end="url(#arrow-res)"/>')
    body.append(f'<line x1="{wt[0]}" y1="{wt[1]}" x2="{wn[0]}" y2="{wn[1]}" stroke="{RED}" stroke-width="2" marker-end="url(#arrow-red)"/>')
    # right-angle mark at w_t
    u = (-240 / 300 * 12, 180 / 300 * 12)
    v = (72 / 120 * 12, 96 / 120 * 12)
    p1 = (wt[0] + u[0], wt[1] + u[1]); p2 = (p1[0] + v[0], p1[1] + v[1]); p3 = (wt[0] + v[0], wt[1] + v[1])
    body.append(f'<path d="M {p1[0]:.1f} {p1[1]:.1f} L {p2[0]:.1f} {p2[1]:.1f} L {p3[0]:.1f} {p3[1]:.1f}" fill="none" stroke="{DIM}" stroke-width="1"/>')
    body.append(text(wt[0] - 70, wt[1] + 4, "w", 17, INK, style="italic"))
    body.append(text(wn[0] + 16, wn[1] + 20, "w′", 17, RES, "start", style="italic"))
    body.append(text(wt[0] + 46, wt[1] + 40, "−η ∇L(w)", 14, RED, "start"))
    body.append(text(wt[0] + 46, wt[1] + 58, "(orthogonal to w)", 11, DIM, "start"))
    body.append(text(ox + 70, oy - 18, "θ", 14, DIM, style="italic"))
    body.append(f'<path d="M {ox + 52} {oy - 39} A 65 65 0 0 1 {ox + 62} {oy - 17}" fill="none" stroke="{DIM}"/>')

    tx = 470
    lines = [
        ("Scale invariance", 14, INK, "bold", SANS),
        ("L(c·w) = L(w) for every c > 0", 13, INK, "normal", MONO),
        ("⇒  w · ∇L(w) = 0         (gradient ⟂ w)", 13, INK, "normal", MONO),
        ("⇒  ∇L(c·w) = ∇L(w) / c", 13, INK, "normal", MONO),
        ("", 6, INK, "normal", SANS),
        ("So every step lengthens w", 14, INK, "bold", SANS),
        ("‖w′‖² = ‖w‖² + η²‖∇L(w)‖²", 13, INK, "normal", MONO),
        ("and turns its direction by", 12, DIM, "normal", SANS),
        ("θ ≈ η‖∇L(ŵ)‖ / ‖w‖²,  ŵ = w / ‖w‖", 13, INK, "normal", MONO),
        ("", 6, INK, "normal", SANS),
        ("A large η inflates ‖w‖, which shrinks the", 12, DIM, "normal", SANS),
        ("next turn: the effective step corrects itself.", 12, DIM, "normal", SANS),
        ("Plain SGD shown; momentum blurs, not breaks, it.", 11, DIM, "normal", SANS),
    ]
    y = 70
    for s, size, color, weight, font in lines:
        if s:
            body.append(text(tx, y, s, size, color, "start", weight, font))
        y += size + 11
    return svg(W, H, "\n".join(body), "Geometry of an SGD step on scale-invariant weights")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in [("models", models), ("resnet18", resnet), ("harness", harness),
                       ("scale-invariance", scale_invariance)]:
        (OUT / f"{name}.svg").write_text(make(), encoding="utf-8")
        print("wrote", OUT / f"{name}.svg")
