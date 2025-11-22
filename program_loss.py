"""
Program-Distilled Energy (PDE) loss for token-level supervision.
Trains the energy model against token-level program distance instead of per-pixel CE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_lago_tokenizer import ArcLagoTokenizer


# --- token distance utilities ---

class ProgramDistanceConfig:
    def __init__(self,
                 w_type=1.0, w_color=0.5, w_params=1.0, w_iou=1.0, w_layer=0.25,
                 w_missed=1.5, normalize_hw=None):
        self.w_type   = w_type
        self.w_color  = w_color
        self.w_params = w_params
        self.w_iou    = w_iou
        self.w_layer  = w_layer
        self.w_missed = w_missed
        self.normalize_hw = normalize_hw  # (H,W) if you need explicit norm


_tok = ArcLagoTokenizer()


def _bbox_from_token(t):
    ty = t.get("type")
    if ty in ("RECT","HOLLOW_RECT","BORDER","CHECKER","C_SHAPE"):
        x, y = int(t.get("x", t.get("cx", 0))), int(t.get("y", t.get("cy", 0)))
        w, h = int(t.get("w", t.get("spacing", 1))), int(t.get("h", t.get("num_layers", 1)))
        return (y, x, y + h - 1, x + w - 1)
    if ty == "CONCENTRIC_RECTS":
        cx, cy = int(t.get("center_x", t.get("cx", 0))), int(t.get("center_y", t.get("cy", 0)))
        spacing, L = int(t.get("spacing", 1)), int(t.get("num_layers", 1))
        half = spacing * L
        return (cy - half, cx - half, cy + half, cx + half)
    if ty == "LINE":
        ori = t.get("orientation", "H")
        x, y = int(t.get("x", 0)), int(t.get("y", 0))
        w, h = int(t.get("w", 1)), int(t.get("h", 1))
        return (y, x, y, x + w - 1) if ori == "H" else (y, x, y + h - 1, x)
    if ty == "DIAG_LINE" or ty == "DIAG_CROSS_X":
        x, y = int(t.get("x", 0)), int(t.get("y", 0))
        w, h = int(t.get("w", 1)), int(t.get("h", 1))
        length = int(t.get("length", max(w, h)))
        return (y, x, y + length - 1, x + length - 1)
    return None


def _iou(bb1, bb2):
    y1,x1,y2,x2 = bb1; Y1,X1,Y2,X2 = bb2
    iy1, ix1 = max(y1, Y1), max(x1, X1)
    iy2, ix2 = min(y2, Y2), min(x2, X2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    if inter <= 0: return 0.0
    a1 = (y2 - y1 + 1) * (x2 - x1 + 1)
    a2 = (Y2 - Y1 + 1) * (X2 - X1 + 1)
    return inter / float(a1 + a2 - inter + 1e-6)


def _kind_dist(tp, tg):  return 0.0 if tp.get("type")  == tg.get("type")  else 1.0
def _color_dist(tp, tg): return 0.0 if int(tp.get("color", -1)) == int(tg.get("color", -1)) else 1.0
def _layer_dist(tp, tg): return min(1.0, abs(int(tp.get("layer", 0)) - int(tg.get("layer", 0))) / 3.0)


def _param_dist(tp, tg, H, W):
    keys = (set(tp.keys()) & set(tg.keys())) - {"type","color","layer","id"}
    if not keys: return 0.0
    s, n = 0.0, 0
    for k in keys:
        vp, vg = tp[k], tg[k]
        if isinstance(vp, str) or isinstance(vg, str):
            s += 0.0 if vp == vg else 1.0; n += 1; continue
        if k in ("x","cx","w"): scale = max(1, W)
        elif k in ("y","cy","h"): scale = max(1, H)
        elif k in ("length","radius","spacing","thickness","border_thickness",
                   "num_layers","arm_length","cell_size","spoke_length",
                   "amplitude","wavelength","num_cycles","coil_count","order"):
            scale = max(H, W)
        elif "angle" in k: scale = 180.0
        else: scale = max(H, W)
        s += abs(float(vp) - float(vg)) / float(scale); n += 1
    return s / max(1, n)


def _tokens_distance(tokens_p, tokens_g, H, W, cfg: ProgramDistanceConfig, kind_weights=None):
    # Greedy matching (sufficient for small ARC token sets)
    Np, Ng = len(tokens_p), len(tokens_g)
    used = [False] * Ng
    total, matches = 0.0, 0

    def _area(bb): return (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) if bb is not None else 0
    order = list(range(Np))
    bbs_p = [_bbox_from_token(t) for t in tokens_p]
    try: order.sort(key=lambda i: _area(bbs_p[i]), reverse=True)
    except: pass

    for pi in order:
        tp = tokens_p[pi]
        best_j, best_cost = -1, 1e9
        for j, tg in enumerate(tokens_g):
            if used[j]: continue
            wt = kind_weights.get(tg.get("type",""), 1.0) if kind_weights is not None else 1.0
            cost  = cfg.w_type   * _kind_dist(tp, tg)
            cost += cfg.w_color  * _color_dist(tp, tg)
            cost += cfg.w_params * _param_dist(tp, tg, H, W)
            bbp, bbg = _bbox_from_token(tp), _bbox_from_token(tg)
            if bbp is not None and bbg is not None:
                cost += cfg.w_iou * (1.0 - _iou(bbp, bbg))
            cost += cfg.w_layer * _layer_dist(tp, tg)
            cost *= wt  # group-balanced
            if cost < best_cost: best_cost, best_j = cost, j
        if best_j != -1 and best_cost < cfg.w_missed:
            total += best_cost; used[best_j] = True; matches += 1
        else:
            total += cfg.w_missed

    # penalize unmatched GT tokens
    total += cfg.w_missed * (Ng - matches)
    return total / max(1, Np + Ng), dict(matches=matches, Np=Np, Ng=Ng)


def _tokenize_grid_tensor(gridHW):
    # gridHW: (H, W) int
    shapes = _tok.tokenize(gridHW.detach().cpu().tolist())["shapes"]
    return shapes


def program_distance_batch(y_logits, y_star, cfg: ProgramDistanceConfig, kind_weights=None):
    # y_logits: (B,H,W,C), y_star: (B,H,W) int
    B, H, W, C = y_logits.shape
    preds = y_logits.argmax(dim=-1)  # (B,H,W)
    dists, infos = [], []
    for b in range(B):
        t_pred = _tokenize_grid_tensor(preds[b])
        t_gt   = _tokenize_grid_tensor(y_star[b])
        d, info = _tokens_distance(t_pred, t_gt, H, W, cfg, kind_weights)
        dists.append(d); infos.append(info)
    return torch.tensor(dists, device=y_logits.device, dtype=torch.float32), infos


# --- group-balanced reweighting over token kinds (EMA on counts) ---
class KindCounterEMA:
    def __init__(self, beta=0.999):
        self.counts = {}  # kind -> ema count
        self.beta = beta

    def update_with_shapes(self, shapes_list):
        # shapes_list: list of token dicts
        batch_counts = {}
        for t in shapes_list:
            k = t.get("type","UNK")
            batch_counts[k] = batch_counts.get(k, 0) + 1
        for k, v in batch_counts.items():
            c = self.counts.get(k, 0.0)
            self.counts[k] = self.beta * c + (1 - self.beta) * float(v)

    def make_weights(self):
        if not self.counts: return {}
        vals = torch.tensor(list(self.counts.values()))
        nmin, nmax = vals.min().item(), vals.max().item()
        w = {}
        for k, v in self.counts.items():
            ki = 0.0 if nmax == nmin else (v - nmin) / (nmax - nmin)
            tau = 1.0 - ki
            w[k] = tau
        # normalize weights to mean 1
        mean_w = sum(w.values()) / max(1, len(w))
        return {k: (val / (mean_w + 1e-8)) for k, val in w.items()}


# --- silhouette regularizer (multi-scale soft Dice + boundary) ---
def soft_silhouette_loss(y_logits, y_star, scales=(1,2,4)):
    # No per-cell CE; compare coarse silhouettes only
    # Build color occupancy (prob) and one-hot GT
    B,H,W,C = y_logits.shape
    P = F.softmax(y_logits.permute(0,3,1,2), dim=1)  # (B,C,H,W)
    G = F.one_hot(y_star, num_classes=C).permute(0,3,1,2).float()  # (B,C,H,W)

    dice = 0.0
    valid_scales = 0
    for s in scales:
        # Skip scales that are too large for the grid
        if s > min(H, W):
            continue
        
        if s > 1:
            k = s
            P_s = F.avg_pool2d(P, kernel_size=k, stride=k)
            G_s = F.avg_pool2d(G, kernel_size=k, stride=k)
        else:
            P_s, G_s = P, G
        inter = (P_s * G_s).sum(dim=(1,2,3))
        den   = (P_s.pow(2) + G_s.pow(2)).sum(dim=(1,2,3)) + 1e-6
        dice += (1.0 - 2.0 * inter / den).mean()
        valid_scales += 1
    
    if valid_scales > 0:
        dice /= valid_scales
    else:
        dice = torch.tensor(0.0, device=y_logits.device)

    # boundary (Sobel) at full scale
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=y_logits.device, dtype=torch.float32).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], device=y_logits.device, dtype=torch.float32).view(1,1,3,3)
    Px = F.conv2d(P.sum(1, keepdim=True), sobel_x, padding=1)
    Py = F.conv2d(P.sum(1, keepdim=True), sobel_y, padding=1)
    Gx = F.conv2d(G.sum(1, keepdim=True), sobel_x, padding=1)
    Gy = F.conv2d(G.sum(1, keepdim=True), sobel_y, padding=1)
    boundary = F.l1_loss(torch.sqrt(Px**2 + Py**2 + 1e-6), torch.sqrt(Gx**2 + Gy**2 + 1e-6))
    return dice + 0.25 * boundary  # tune 0.25 if needed


# --- the new Program‑Distilled Energy loss ---
def program_distilled_energy_loss(model, x_test, rbar, y_logits, y_star,
                                  pd_cfg: ProgramDistanceConfig,
                                  kind_ema: KindCounterEMA = None,
                                  lambda_reg=1.0, lambda_rank=1.0, lambda_sil=0.0,
                                  base_margin=0.1, margin_scale=1.0):
    """
    Returns:
        loss (scalar), metrics (dict)
    """
    with torch.no_grad():
        # compute token sets for GT to update EMA
        all_gt_shapes = []
        for b in range(y_star.shape[0]):
            all_gt_shapes += _tokenize_grid_tensor(y_star[b])
        if kind_ema is not None:
            kind_ema.update_with_shapes(all_gt_shapes)
        kind_weights = (kind_ema.make_weights() if kind_ema is not None else None)

    # energy positives/negatives
    E_pos = model.energy(x_test, rbar, y_star, canonical=False)  # (B,)
    E_neg = model.energy(x_test, rbar, y_logits, canonical=False)  # (B,)

    # program distances
    d_prog, _ = program_distance_batch(y_logits, y_star, pd_cfg, kind_weights)  # (B,)

    # (1) margin ranking: E_pos + m < E_neg, m depends on distance
    margin = base_margin + margin_scale * d_prog  # (B,)
    rank_loss = F.relu(E_pos + margin - E_neg).mean()

    # (2) regression: teach the scale of E_neg to approximate d_prog
    reg_loss = F.smooth_l1_loss(E_neg, d_prog)

    # (3) optional silhouette reg (no per-cell CE)
    sil_loss = soft_silhouette_loss(y_logits, y_star) if lambda_sil > 0.0 else torch.tensor(0.0, device=y_logits.device)

    loss = lambda_rank * rank_loss + lambda_reg * reg_loss + lambda_sil * sil_loss

    with torch.no_grad():
        pred = y_logits.argmax(dim=-1)
        acc = (pred == y_star).float().mean()  # for monitoring only

    metrics = {
        'loss_rank': rank_loss.item(),
        'loss_reg': reg_loss.item(),
        'loss_sil': sil_loss.item() if lambda_sil > 0.0 else 0.0,
        'prog_dist': d_prog.mean().item(),
        'pixel_acc': acc.item(),
        'energy_pos': E_pos.mean().item(),
        'energy_neg': E_neg.mean().item(),
        'margin_mean': margin.mean().item(),
    }
    return loss, metrics

