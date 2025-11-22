# arc_lago_tokenizer.py
# A shape-aware ARC/LAGO tokenizer with layers, relations, and scale-aware primitives.
# Pure Python (no third-party deps). Drop into your project and import `ArcLagoTokenizer`.

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Set
import math
import itertools
import collections

# Types
Point = Tuple[int, int]  # (row, col)
Box   = Tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
Grid  = List[List[int]]  # integer colors, 0..K-1 (or -1 for blank background)

# -------------------------------
# Utility: neighborhood, geometry
# -------------------------------

# 4-connected for components; 8-connected for boundary walks
N4 = [(-1,0),(1,0),(0,-1),(0,1)]
N8 = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]

def in_bounds(g: Grid, r: int, c: int) -> bool:
    return 0 <= r < len(g) and 0 <= c < len(g[0])

def bbox_of(points: Set[Point]) -> Box:
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    return (min(rs), min(cs), max(rs), max(cs))

def area_of(box: Box) -> int:
    r0,c0,r1,c1 = box
    return (r1-r0+1)*(c1-c0+1)

def box_intersection(a: Box, b: Box) -> Optional[Box]:
    r0 = max(a[0], b[0]); c0 = max(a[1], b[1])
    r1 = min(a[2], b[2]); c1 = min(a[3], b[3])
    if r0 <= r1 and c0 <= c1:
        return (r0,c0,r1,c1)
    return None

def box_center(box: Box) -> Tuple[float,float]:
    r0,c0,r1,c1 = box
    return ((r0+r1)/2.0, (c0+c1)/2.0)

def pixel_count_in_box(points: Set[Point], box: Box) -> int:
    r0,c0,r1,c1 = box
    return sum(1 for (r,c) in points if r0 <= r <= r1 and c0 <= c <= c1)

def iou_boxes(a: Box, b: Box) -> float:
    inter = box_intersection(a,b)
    if inter is None: return 0.0
    ai = area_of(inter)
    return ai / float(area_of(a) + area_of(b) - ai)

def rotate_points(points: Set[Point], about: Point, k: int) -> Set[Point]:
    # rotate by k*90 degrees around 'about'
    (ar, ac) = about
    rs = []
    for (r,c) in points:
        dr, dc = (r - ar, c - ac)
        # rotations in screen coords (row increases downward)
        if   k % 4 == 1: (nr,nc) = (-dc, dr)
        elif k % 4 == 2: (nr,nc) = (-dr, -dc)
        elif k % 4 == 3: (nr,nc) = (dc, -dr)
        else:            (nr,nc) = (dr, dc)
        rs.append((ar+nr, ac+nc))
    return set(rs)

def reflect_points(points: Set[Point], about: Point, axis: str) -> Set[Point]:
    ar, ac = about
    result = []
    for (r,c) in points:
        dr, dc = (r - ar, c - ac)
        if axis == 'h': nr, nc = (-dr,  dc)    # reflect across horizontal line through about
        elif axis == 'v': nr, nc = ( dr, -dc)  # reflect across vertical line through about
        elif axis == 'd': nr, nc = ( dc,  dr)  # main diag
        else:             nr, nc = (-dc, -dr)  # anti-diag
        result.append((ar+nr, ac+nc))
    return set(result)

def translate_to_origin(points: Set[Point]) -> Set[Point]:
    r0 = min(p[0] for p in points)
    c0 = min(p[1] for p in points)
    return set((r-r0, c-c0) for (r,c) in points)

def perimeter_estimate(points: Set[Point]) -> int:
    # count exposed edges in 4-neigh
    s = 0
    P = set(points)
    for (r,c) in points:
        for dr,dc in N4:
            if (r+dr, c+dc) not in P:
                s += 1
    return s

def is_thin_path(points: Set[Point]) -> bool:
    # each interior point has 2 neighbors (in 8-neigh), endpoints have 1 or 2 if closed
    P = set(points)
    degs = []
    for (r,c) in points:
        deg = sum((r+dr, c+dc) in P for (dr,dc) in N8)
        degs.append(deg)
    # tolerate some endpoints/branch artifacts
    end_like = sum(d <= 2 for d in degs)
    branchy  = sum(d >= 4 for d in degs)
    return branchy == 0 and end_like >= max(2, int(0.05*len(points)))

# -------------------------------
# Connected components by color
# -------------------------------

@dataclass
class Region:
    id: int
    color: int
    pixels: Set[Point]
    bbox: Box

def connected_components_by_color(g: Grid) -> List[Region]:
    H, W = len(g), len(g[0])
    visited = [[False]*W for _ in range(H)]
    regions: List[Region] = []
    rid = 0
    for r in range(H):
        for c in range(W):
            col = g[r][c]
            if col < 0 or visited[r][c]: # skip background if you use -1
                continue
            # BFS 4-connected per color
            if not visited[r][c]:
                q = collections.deque([(r,c)])
                visited[r][c] = True
                pixels = {(r,c)}
                while q:
                    rr,cc = q.popleft()
                    for dr,dc in N4:
                        nr, nc = rr+dr, cc+dc
                        if in_bounds(g, nr, nc) and not visited[nr][nc] and g[nr][nc] == col:
                            visited[nr][nc] = True
                            q.append((nr,nc))
                            pixels.add((nr,nc))
                regions.append(Region(id=rid, color=col, pixels=pixels, bbox=bbox_of(pixels)))
                rid += 1
    return regions

# -------------------------------
# Primitive catalogs
# -------------------------------

# Tetromino (7) in canonical 1x1 cells (origin at min row/col).
# Each as a set of (r,c). We’ll match under rotations/reflections and scale s>0.
TETROMINOES = {
    'I': {(0,0),(1,0),(2,0),(3,0)},
    'O': {(0,0),(0,1),(1,0),(1,1)},
    'T': {(0,0),(0,1),(0,2),(1,1)},
    'S': {(0,1),(0,2),(1,0),(1,1)},
    'Z': {(0,0),(0,1),(1,1),(1,2)},
    'J': {(0,0),(1,0),(2,0),(2,1)},
    'L': {(0,1),(1,1),(2,1),(2,0)},
}

# Pentomino (12)
PENTROMINOES = {
    'F': {(0,1),(1,0),(1,1),(1,2),(2,2)},
    'I': {(0,0),(1,0),(2,0),(3,0),(4,0)},
    'L': {(0,0),(1,0),(2,0),(3,0),(3,1)},
    'P': {(0,0),(0,1),(1,0),(1,1),(2,0)},
    'N': {(0,1),(1,1),(2,1),(2,0),(3,0)},
    'T': {(0,0),(0,1),(0,2),(1,1),(2,1)},
    'U': {(0,0),(0,2),(1,0),(1,1),(1,2)},
    'V': {(0,0),(1,0),(2,0),(2,1),(2,2)},
    'W': {(0,0),(1,0),(1,1),(2,1),(2,2)},
    'X': {(0,1),(1,0),(1,1),(1,2),(2,1)},
    'Y': {(0,0),(1,0),(2,0),(3,0),(2,1)},
    'Z': {(0,0),(0,1),(1,1),(2,1),(2,2)},
}

def all_orientations(cells: Set[Point]) -> List[Set[Point]]:
    # all 8 symmetries (4 rotations x {id, reflect v})
    base = translate_to_origin(cells)
    centers = (0,0)
    variants = []
    for k in range(4):
        rot = translate_to_origin(rotate_points(base, centers, k))
        variants.append(translate_to_origin(rot))
        ref = translate_to_origin(reflect_points(rot, centers, 'v'))
        variants.append(translate_to_origin(ref))
    # uniq sets
    uniq = []
    seen = set()
    for s in variants:
        tup = tuple(sorted(s))
        if tup not in seen:
            seen.add(tup)
            uniq.append(s)
    return uniq

def scale_cells(cells: Set[Point], s: int) -> Set[Point]:
    # scale unit cells to s-by-s blocks
    out = set()
    for (r,c) in cells:
        for dr in range(s):
            for dc in range(s):
                out.add((r*s + dr, c*s + dc))
    return out

# -------------------------------
# Primitive detectors
# -------------------------------

@dataclass
class ShapeToken:
    kind: str
    color: int
    layer: int
    params: Dict[str, int|float|str]
    id: int = -1  # region id that originated the token (for relations), optional

    def as_dict(self) -> Dict:
        d = asdict(self)
        # keep params first for readability
        return {
            "type": self.kind,
            "color": self.color,
            "layer": self.layer,
            "id": self.id,
            **self.params
        }

def is_filled_rectangle(points: Set[Point]) -> Optional[Tuple[int,int,int,int]]:
    box = bbox_of(points)
    # fully filled?
    if pixel_count_in_box(points, box) == area_of(box):
        # thickness check to see if "border" is thicker later; filled rectangle for now
        return box
    return None

def is_hollow_rectangle(points: Set[Point]) -> Optional[Tuple[Box,int]]:
    # hollow if boundary filled but inner box mostly empty
    box = bbox_of(points)
    r0,c0,r1,c1 = box
    H = r1-r0+1; W = c1-c0+1
    if H < 3 or W < 3:
        return None
    # estimate border thickness by scanning inward while all 4 edges exist
    max_t = min(H,W)//2
    P = points
    t = 0
    def edge_count(bx: Box) -> int:
        br0,bc0,br1,bc1 = bx
        ed = 0
        for c in range(bc0, bc1+1):
            ed += ((br0,c) in P) + ((br1,c) in P)
        for r in range(br0+1, br1):
            ed += ((r,bc0) in P) + ((r,bc1) in P)
        return ed
    # Try to find largest t s.t. outer t rings are solid and inner mostly empty
    for tt in range(1, max_t+1):
        outer = (r0,c0,r1,c1)
        inner = (r0+tt, c0+tt, r1-tt, c1-tt)
        if inner[0] > inner[2] or inner[1] > inner[3]:
            break
        # edges present?
        need = 2*( (outer[3]-outer[1]+1) + (outer[2]-outer[0]+1) ) - 4
        if edge_count(outer) >= 0.9*need:  # tolerate chips
            # inner mostly empty
            inner_area = area_of(inner)
            inner_count = pixel_count_in_box(points, inner)
            if inner_area > 0 and inner_count < 0.1*inner_area:
                t = tt
            else:
                break
        else:
            break
    if t >= 1:
        return (box, t)
    return None

def is_border_frame(points: Set[Point], grid_box: Box) -> bool:
    # an explicit frame around the entire grid or around outer bbox?
    # Check if region equals the grid border lines (or a subset rectangle border).
    # Simple: if its bbox equals grid_box and it's hollow rectangle with thickness >=1 and large coverage.
    hollow = is_hollow_rectangle(points)
    if not hollow: return False
    box,th = hollow
    return box == grid_box and th >= 1

def is_line(points: Set[Point]) -> Optional[Tuple[str,int,int]]:
    # H/V line with thickness=1 (or nearly 1)
    box = bbox_of(points)
    r0,c0,r1,c1 = box
    H = r1-r0+1; W = c1-c0+1
    count = len(points)
    if H == 1 and count >= 2: # horizontal
        return ('H', r0, c0)
    if W == 1 and count >= 2: # vertical
        return ('V', r0, c0)
    return None

def is_diag_45(points: Set[Point]) -> Optional[Tuple[str,Point,int]]:
    # simple 45° (NE,NW,SE,SW) line: c - r constant or c + r constant with step 1
    # check thin path, and bounding box aspect roughly diagonal
    if not is_thin_path(points):
        return None
    rs = sorted(points)
    box = bbox_of(points)
    r0,c0,r1,c1 = box
    dr = r1 - r0; dc = c1 - c0
    if dr == 0 or dc == 0:
        return None
    # test whether all points satisfy c-r == const or c+r == const within tolerance
    vals1 = set(c - r for (r,c) in points)
    vals2 = set(c + r for (r,c) in points)
    if len(vals1) == 1:
        # slope +1: SE or NW
        orientation = 'SE' if list(vals1)[0] > 0 else 'NW'
        return (orientation, (r0,c0), max(dr,dc)+1)
    if len(vals2) == 1:
        # slope -1: NE or SW
        # we infer orientation by increasing row -> col decreasing/increasing
        orientation = 'NE'  # nominal
        return (orientation, (r0,c0), max(dr,dc)+1)
    return None

def is_x_cross(points: Set[Point]) -> bool:
    # two diagonals crossing in bbox, near full coverage of those two diagonals
    box = bbox_of(points)
    r0,c0,r1,c1 = box
    H = r1-r0+1; W = c1-c0+1
    if H != W or H < 3:
        return False
    needed = 2*H - 1
    diag1 = {(r0+i, c0+i) for i in range(H)}
    diag2 = {(r0+i, c1-i) for i in range(H)}
    cov = len((diag1|diag2) & points)
    return cov >= int(0.9*len(diag1|diag2))

def match_polyomino(points: Set[Point]) -> Optional[Tuple[str,str,int,int,int]]:
    # Try tetromino then pentomino, with scale s inferred from GCD of cell extents.
    # Normalize to a binary footprint (no holes). We approximate by taking the set of pixels and seeing if it's s-scaled from a unit-cell set.
    S = translate_to_origin(points)
    box = bbox_of(S)
    H = box[2]-box[0]+1; W = box[3]-box[1]+1

    # infer scale s by looking at run-lengths along rows and cols
    # crude: estimate s as gcd of all horizontal and vertical runs
    def row_runs(ps: Set[Point]) -> List[int]:
        runs = []
        mp = collections.defaultdict(list)
        for (r,c) in ps: mp[r].append(c)
        for r, cols in mp.items():
            cols.sort()
            run = 1
            for i in range(1,len(cols)):
                if cols[i] == cols[i-1]+1:
                    run += 1
                else:
                    runs.append(run); run = 1
            runs.append(run)
        return runs

    def col_runs(ps: Set[Point]) -> List[int]:
        runs = []
        mp = collections.defaultdict(list)
        for (r,c) in ps: mp[c].append(r)
        for c, rows in mp.items():
            rows.sort()
            run = 1
            for i in range(1,len(rows)):
                if rows[i] == rows[i-1]+1:
                    run += 1
                else:
                    runs.append(run); run = 1
            runs.append(run)
        return runs

    runs = row_runs(S) + col_runs(S)
    s = 1
    if runs:
        from math import gcd
        g = 0
        for v in runs:
            g = gcd(g, v)
        s = max(1, g)

    # compress by scale s to a unit grid
    unit = set((r//s, c//s) for (r,c) in S)
    unit = translate_to_origin(unit)

    # helper: does unit match any orientation of shape at unit scale?
    def match_family(family: Dict[str, Set[Point]]) -> Optional[Tuple[str,int,int]]:
        for name, cells in family.items():
            for orient in all_orientations(cells):
                if unit == orient:
                    h = max(r for (r,_) in orient) + 1
                    w = max(c for (_,c) in orient) + 1
                    return (name, h, w)
        return None

    m = match_family(TETROMINOES)
    if m:  # (shape_id, h, w)
        sid, uh, uw = m
        return ('TETROMINO', sid, s, uh, uw)
    m = match_family(PENTROMINOES)
    if m:
        sid, uh, uw = m
        return ('PENTOMINO', sid, s, uh, uw)
    return None

def is_checkerboard_region(points: Set[Point], color: int, grid: Grid) -> Optional[Dict]:
    # detect a regular parity pattern (two colors alternating). We return parameters if
    # (i+j)%2 -> color or the other color dominates inside bbox and coverage is high.
    box = bbox_of(points)
    r0,c0,r1,c1 = box
    H = r1-r0+1; W = c1-c0+1
    total = H*W
    same_parity = 0
    other_color = None
    other_parity = 0
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
            if grid[r][c] == color and ((r+c)&1) == ((r0+c0)&1):
                same_parity += 1
            elif grid[r][c] != color:
                if other_color is None: other_color = grid[r][c]
                if grid[r][c] == other_color and ((r+c)&1) != ((r0+c0)&1):
                    other_parity += 1
    if same_parity + other_parity >= 0.85 * total and other_color is not None:
        return {"x": c0, "y": r0, "w": W, "h": H, "color2": other_color, "cell_size": 1}
    return None

def has_symmetry(points: Set[Point]) -> List[Tuple[str, int|float]]:
    # find plausible symmetry axes: horizontal, vertical, diag/anti-diag; rotational of order 2 or 4
    feats = []
    S = translate_to_origin(points)
    box = bbox_of(S)
    H = box[2]-box[0]+1; W = box[3]-box[1]+1
    # vertical axis around mid-col?
    midc = (W-1)/2.0
    mirrored_v = set((r, int(2*midc - c)) for (r,c) in S)
    if mirrored_v == S:
        feats.append(("symmetry_axis", "vertical"))
    # horizontal
    midr = (H-1)/2.0
    mirrored_h = set((int(2*midr - r), c) for (r,c) in S)
    if mirrored_h == S:
        feats.append(("symmetry_axis", "horizontal"))
    # diagonal
    diag = set((c,r) for (r,c) in S)
    if diag == S:
        feats.append(("symmetry_axis", "diagonal"))
    # anti-diagonal
    anti = set((W-1-c, H-1-r) for (r,c) in S)
    if anti == S:
        feats.append(("symmetry_axis", "anti_diagonal"))
    # rotation 180
    rot180 = set((H-1-r, W-1-c) for (r,c) in S)
    if rot180 == S:
        feats.append(("rotational", 2))
    # rotation 90/270 are rare for grid glyphs but we can test
    rot90  = translate_to_origin(rotate_points(S, (0,0), 1))
    rot270 = translate_to_origin(rotate_points(S, (0,0), 3))
    if rot90 == S or rot270 == S:
        feats.append(("rotational", 4))
    return feats

def is_concentric_rectangles(points: Set[Point]) -> Optional[Dict]:
    # try to find multiple hollow rectangles with common center
    box = bbox_of(points)
    r0,c0,r1,c1 = box
    rings = []
    P = points
    # peel onion
    t = 0
    while True:
        outer = (r0+t, c0+t, r1-t, c1-t)
        if outer[0] > outer[2] or outer[1] > outer[3]:
            break
        # check if this ring exists
        need = 2*((outer[3]-outer[1]+1)+(outer[2]-outer[0]+1)) - 4
        got  = 0
        br0,bc0,br1,bc1 = outer
        for c in range(bc0,bc1+1):
            got += ((br0,c) in P) + ((br1,c) in P)
        for r in range(br0+1, br1):
            got += ((r,bc0) in P) + ((r,bc1) in P)
        if got >= 0.8*need:
            rings.append(outer)
            t += 2  # step by 2 to leave space between rings
        else:
            break
    if len(rings) >= 2:
        return {"cx": (r0+r1)/2.0, "cy": (c0+c1)/2.0, "num_layers": len(rings), "spacing": 1}
    return None

def is_c_or_u_shape(points: Set[Point]) -> Optional[Dict]:
    # detect a rectangle with one open side (U/C)
    hollow = is_hollow_rectangle(points)
    if not hollow:
        return None
    box, th = hollow
    # find which side is most missing along border -> opening direction
    r0,c0,r1,c1 = box
    P = points
    sides = {
        "up":    sum((r0,c) in P for c in range(c0,c1+1)),
        "down":  sum((r1,c) in P for c in range(c0,c1+1)),
        "left":  sum((r,c0) in P for r in range(r0,r1+1)),
        "right": sum((r,c1) in P for r in range(r0,r1+1)),
    }
    # opening is the side with smallest coverage if that side is significantly missing
    side, val = min(sides.items(), key=lambda kv: kv[1])
    if val < 0.5*max(sides.values()):
        return {"x": c0, "y": r0, "w": c1-c0+1, "h": r1-r0+1, "opening_direction": side, "border_thickness": th}
    return None

def is_sparse_dots(points: Set[Point]) -> Optional[List[Point]]:
    # many isolated pixels with little adjacency (scatter)
    degs = []
    P = set(points)
    for (r,c) in points:
        deg = sum((r+dr, c+dc) in P for (dr,dc) in N8)
        degs.append(deg)
    if len(points) >= 4 and sum(d==0 for d in degs) >= 0.7*len(points):
        return sorted(points)
    return None

def is_crosshatch(points: Set[Point]) -> Optional[Dict]:
    # crude: check if region is union of near-uniformly spaced H and V thin lines inside bbox
    box = bbox_of(points)
    r0,c0,r1,c1 = box
    H = r1-r0+1; W = c1-c0+1
    # detect at least 2 parallel H lines and 2 parallel V lines
    rows = collections.Counter([r for (r,_) in points])
    cols = collections.Counter([c for (_,c) in points])
    h_lines = [r for r,cnt in rows.items() if cnt >= 0.8*W]
    v_lines = [c for c,cnt in cols.items() if cnt >= 0.8*H]
    if len(h_lines) >= 2 and len(v_lines) >= 2:
        spacing_h = min(abs(h_lines[i]-h_lines[i-1]) for i in range(1,len(h_lines)))
        spacing_v = min(abs(v_lines[i]-v_lines[i-1]) for i in range(1,len(v_lines)))
        return {"x": c0, "y": r0, "w": W, "h": H, "spacing_h": spacing_h, "spacing_v": spacing_v}
    return None

def detect_radial(points: Set[Point]) -> Optional[Dict]:
    # simple "spoke" detection from center of bbox
    box = bbox_of(points)
    cr, cc = box_center(box)
    # count rays in 8 directions
    dirs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    P = set(points)
    spokes = 0
    longest = 0
    for dr,dc in dirs:
        length = 0
        r, c = int(round(cr)), int(round(cc))
        while (r, c) in P:
            r += dr; c += dc; length += 1
        if length >= 2:
            spokes += 1
            longest = max(longest, length)
    if spokes >= 3:
        return {"cx": cc, "cy": cr, "num_spokes": spokes, "spoke_length": longest}
    return None

def detect_zigzag_or_wave(points: Set[Point]) -> Optional[Dict]:
    # thin path with alternating turn directions -> zigzag
    if not is_thin_path(points): return None
    # approximate with bounding box amplitude/wavelength
    box = bbox_of(points)
    r0,c0,r1,c1 = box
    amp = (r1-r0+1)
    wl  = max(2, (c1-c0+1)//2)
    orientation = "horizontal" if (c1-c0) >= (r1-r0) else "vertical"
    num_cycles = max(1, (c1-c0+1)//wl if orientation=="horizontal" else (r1-r0+1)//wl)
    return {"x": c0,"y": r0,"amplitude": amp,"wavelength": wl,"num_cycles": num_cycles,"orientation": orientation}

def detect_spiral(points: Set[Point]) -> Optional[Dict]:
    # heuristic: a single thin path whose bounding box is roughly square and path length >> box perimeter
    if not is_thin_path(points): return None
    box = bbox_of(points)
    per = 2*((box[2]-box[0]+1)+(box[3]-box[1]+1))
    if len(points) >= 2.5*per and abs((box[2]-box[0])-(box[3]-box[1])) <= 2:
        # we emit a coarse chain-code length and coil_count ~ turns
        coil = max(1, len(points)//per)
        return {"x": box[1], "y": box[0], "coil_count": coil}
    return None

# -------------------------------
# Tiling (repeating pattern)
# -------------------------------

def detect_repeating_tiles(grid: Grid) -> Optional[Dict]:
    H,W = len(grid), len(grid[0])
    # brute-force small periods up to min(8, W//2, H//2)
    maxP = max(2, min(8, H//2, W//2))
    best = None
    for ph in range(1, maxP+1):
        for pw in range(1, maxP+1):
            ok = True
            for r in range(H):
                for c in range(W):
                    if grid[r][c] != grid[r % ph][c % pw]:
                        ok = False; break
                if not ok: break
            if ok:
                best = {"tile_h": ph, "tile_w": pw}
                break
        if best: break
    return best

# -------------------------------
# Occlusion & layering (up to 4)
# -------------------------------

def build_occlusion_layers(regions: List[Region], grid: Grid, max_layers: int = 4) -> Dict[int,int]:
    """
    Very simple occlusion DAG: region B is likely above A if B's pixels fall inside A's bbox and colors differ.
    Then topologically sort; map to 0..max_layers-1 by rank compression.
    """
    # Build adjacency: A -> B if B overlaps A's bbox & colors differ
    overlaps = collections.defaultdict(set)  # B above A: A->B
    by_id = {r.id:r for r in regions}
    for a,b in itertools.permutations(regions, 2):
        inter = box_intersection(a.bbox, b.bbox)
        if not inter: continue
        # if there are pixels in intersection that belong to b (always true) and colors differ, call B above A
        if a.color != b.color:
            overlaps[a.id].add(b.id)

    # topo sort by Kahn
    indeg = collections.Counter()
    for a, outs in overlaps.items():
        for b in outs: indeg[b]+=1
    Q = collections.deque([r.id for r in regions if indeg[r.id]==0])
    order = []
    while Q:
        u = Q.popleft()
        order.append(u)
        for v in overlaps.get(u, []):
            indeg[v]-=1
            if indeg[v]==0: Q.append(v)
    # assign ranks by longest-path distance
    rank = {rid:0 for rid in order}
    for u in order:
        for v in overlaps.get(u, []):
            rank[v] = max(rank[v], rank[u]+1)
    # compress to [0..max_layers-1], keeping relative order
    ranks = sorted(set(rank.values()))
    mapping = {rv:i for i,rv in enumerate(ranks[:max_layers])}
    # clamp overflow into top layer
    layer_of = {}
    for rid, rv in rank.items():
        idx = mapping.get(rv, max_layers-1)
        layer_of[rid] = idx
    return layer_of

# -------------------------------
# Relations
# -------------------------------

def relations_between(tokens: List[ShapeToken]) -> List[Dict]:
    out = []
    # Map token id -> bbox from params where applicable
    bbs: Dict[int, Box] = {}
    for t in tokens:
        if t.kind in ("RECT","HOLLOW_RECT","BORDER","C_SHAPE","CONCENTRIC_RECTS","CHECKER"):
            x = int(t.params.get("x", t.params.get("cx",0)))
            y = int(t.params.get("y", t.params.get("cy",0)))
            w = int(t.params.get("w", t.params.get("spacing",1)))
            h = int(t.params.get("h", t.params.get("num_layers",1)))
            # careful with concentric: we approximate by outer bbox if w/h not present
            if "w" in t.params and "h" in t.params:
                bb = (y, x, y+h-1, x+w-1)
            else:
                # fallback: 3x3 around center
                cy = int(round(t.params.get("cy",0)))
                cx = int(round(t.params.get("cx",0)))
                bb = (cy-1, cx-1, cy+1, cx+1)
            bbs[t.id] = bb
        elif t.kind in ("LINE","DIAG_LINE","X_CROSS","TETROMINO","PENTOMINO","ZIGZAG","SPIRAL","RADIAL","SPARSE_DOTS","CROSSHATCH"):
            # approximate bbox from params
            if "x" in t.params and "y" in t.params and "w" in t.params and "h" in t.params:
                bb = (int(t.params["y"]), int(t.params["x"]), int(t.params["y"])+int(t.params["h"])-1, int(t.params["x"])+int(t.params["w"])-1)
                bbs[t.id] = bb

    # Pairwise relations
    ids = [t.id for t in tokens if t.id in bbs]
    for a,b in itertools.permutations(ids,2):
        A, B = bbs[a], bbs[b]
        inter = box_intersection(A,B)
        if inter:
            # inside if IoU(A,B) == area(A)/area(B) ~ 1 wrt smaller
            if area_of(inter) == area_of(A):
                out.append({"type":"REL","rel":"inside","a":a,"b":b})
            elif area_of(inter) > 0:
                out.append({"type":"REL","rel":"overlaps","a":a,"b":b})
        # touches if Manhattan distance between boxes == 1 along one axis and overlap on the other axis
        if not inter:
            # compute horizontal adjacency
            if (A[1] == B[3]+1 or B[1] == A[3]+1) and not (A[2] < B[0] or B[2] < A[0]):
                out.append({"type":"REL","rel":"touches","a":a,"b":b})
            if (A[0] == B[2]+1 or B[0] == A[2]+1) and not (A[3] < B[1] or B[3] < A[1]):
                out.append({"type":"REL","rel":"touches","a":a,"b":b})
        # left_of / right_of / above / below
        ac = (A[1]+A[3])/2.0; bc = (B[1]+B[3])/2.0
        ar = (A[0]+A[2])/2.0; br = (B[0]+B[2])/2.0
        if ac < bc - 0.1: out.append({"type":"REL","rel":"left_of","a":a,"b":b})
        if ac > bc + 0.1: out.append({"type":"REL","rel":"right_of","a":a,"b":b})
        if ar < br - 0.1: out.append({"type":"REL","rel":"above","a":a,"b":b})
        if ar > br + 0.1: out.append({"type":"REL","rel":"below","a":a,"b":b})
        # alignment
        if abs(ar - br) <= 0.001: out.append({"type":"REL","rel":"aligned_row","a":a,"b":b})
        if abs(ac - bc) <= 0.001: out.append({"type":"REL","rel":"aligned_col","a":a,"b":b})
    # deduplicate relations
    seen = set()
    uniq = []
    for r in out:
        key = tuple(sorted(r.items()))
        if key not in seen:
            seen.add(key); uniq.append(r)
    return uniq

# -------------------------------
# Post-processing: merge diagonal lines
# -------------------------------

def merge_diagonal_single_pixels(tokens: List[ShapeToken]) -> List[ShapeToken]:
    """
    Merge single-pixel RECTs that form diagonal lines.
    Diagonal pixels are only 8-connected, so they appear as separate regions
    in 4-connected component analysis.
    """
    # Find all 1x1 RECT tokens
    single_pixels = {}  # color -> list of (token_idx, x, y)
    for i, tok in enumerate(tokens):
        if tok.kind == "RECT" and tok.params.get("w") == 1 and tok.params.get("h") == 1:
            color = tok.color
            x, y = tok.params["x"], tok.params["y"]
            if color not in single_pixels:
                single_pixels[color] = []
            single_pixels[color].append((i, x, y))
    
    # For each color, find diagonal chains
    merged_indices = set()  # Indices of tokens that have been merged
    new_tokens = []
    
    for color, pixels in single_pixels.items():
        if len(pixels) < 3:  # Need at least 3 pixels for a line
            continue
        
        # Build adjacency for 8-connected diagonal neighbors
        pixel_map = {(x, y): idx for (idx, x, y) in pixels}
        
        # Try to find diagonal chains
        visited = set()
        for start_idx, start_x, start_y in pixels:
            if (start_x, start_y) in visited:
                continue
            
            # BFS to find connected diagonal chain
            chain = []
            queue = [(start_x, start_y)]
            visited_local = set()
            
            while queue:
                x, y = queue.pop(0)
                if (x, y) in visited_local or (x, y) not in pixel_map:
                    continue
                visited_local.add((x, y))
                chain.append((x, y))
                
                # Check 8 neighbors
                for dx, dy in N8:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in pixel_map and (nx, ny) not in visited_local:
                        queue.append((nx, ny))
            
            # Segment chain into straight diagonal lines
            if len(chain) >= 2:
                visited.update(chain)
                
                # Mark all original tokens for removal
                for x, y in chain:
                    merged_indices.add(pixel_map[(x, y)])
                
                # Segment the chain into straight diagonal segments
                # Build adjacency graph
                neighbors = {}
                for x, y in chain:
                    neighbors[(x, y)] = []
                    for dx, dy in N8:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in pixel_map:
                            neighbors[(x, y)].append((nx, ny))
                
                # Find segments using greedy line growing
                remaining = set(chain)
                segments = []
                
                while remaining:
                    # Start from any remaining pixel
                    start = remaining.pop()
                    
                    # Try to grow a straight diagonal line in both directions
                    # Try 4 diagonal directions: NE, SE, SW, NW
                    best_segment = [start]
                    
                    for direction in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:  # SE, SW, NE, NW
                        dx, dy = direction
                        segment = [start]
                        
                        # Grow forward
                        x, y = start
                        while True:
                            nx, ny = x + dx, y + dy
                            if (nx, ny) in remaining:
                                segment.append((nx, ny))
                                x, y = nx, ny
                            else:
                                break
                        
                        # Grow backward
                        x, y = start
                        while True:
                            nx, ny = x - dx, y - dy
                            if (nx, ny) in remaining:
                                segment.insert(0, (nx, ny))
                                x, y = nx, ny
                            else:
                                break
                        
                        if len(segment) > len(best_segment):
                            best_segment = segment
                    
                    # Add segment if it has at least 2 pixels
                    if len(best_segment) >= 2:
                        segments.append(best_segment)
                        for p in best_segment:
                            remaining.discard(p)
                    
                    # Safety: if we didn't grow, just take the single pixel
                    if len(best_segment) == 1 and start in remaining:
                        remaining.discard(start)
                
                # Create DIAG_LINE tokens for segments with >= 2 pixels
                for segment in segments:
                    if len(segment) >= 2:
                        # Determine orientation from first two pixels
                        (x1, y1), (x2, y2) = segment[0], segment[1]
                        dx, dy = x2 - x1, y2 - y1
                        
                        # Map direction to orientation
                        if dx > 0 and dy > 0:
                            orientation = 'SE'
                        elif dx > 0 and dy < 0:
                            orientation = 'SW'
                        elif dx < 0 and dy > 0:
                            orientation = 'NE'
                        else:
                            orientation = 'NW'
                        
                        # Convert to row,col and get bounding box
                        points = set((r, c) for (c, r) in segment)
                        box = bbox_of(points)
                        r0, c0, r1, c1 = box
                        
                        # Get layer from first token
                        first_token_idx = pixel_map[segment[0]]
                        layer = tokens[first_token_idx].layer
                        
                        new_token = ShapeToken(
                            "DIAG_LINE",
                            color,
                            layer,
                            {"x": c0, "y": r0, "length": len(segment), "orientation": orientation},
                            -1  # New merged token
                        )
                        new_tokens.append(new_token)
    
    # Build final token list: keep non-merged tokens + new merged tokens
    result = []
    for i, tok in enumerate(tokens):
        if i not in merged_indices:
            result.append(tok)
    result.extend(new_tokens)
    
    return result


# -------------------------------
# Tokenizer
# -------------------------------

@dataclass
class TokenizerConfig:
    max_layers: int = 4
    emit_symmetry_meta: bool = True
    emit_relations: bool = True
    assume_rect_occlusion_prior: bool = True  # "rectangle-first" interpretation prior
    merge_diagonal_lines: bool = True  # Merge diagonally-connected single pixels into lines

class ArcLagoTokenizer:
    def __init__(self, cfg: TokenizerConfig = TokenizerConfig()):
        self.cfg = cfg

    def tokenize(self, grid: Grid) -> Dict[str, List[Dict]]:
        """
        Main entry: returns a dict with:
          - 'shapes': list of shape tokens (dicts)
          - 'relations': list of relation tokens (if enabled)
          - 'meta': list of meta tokens (symmetry, tiling)
        """
        H, W = len(grid), len(grid[0])
        regions = connected_components_by_color(grid)
        layer_of = build_occlusion_layers(regions, grid, max_layers=self.cfg.max_layers)

        tokens: List[ShapeToken] = []
        for reg in regions:
            ly = layer_of.get(reg.id, 0)
            points = reg.pixels
            box = reg.bbox
            r0,c0,r1,c1 = box
            h = r1-r0+1; w = c1-c0+1

            # 1) Strong rectangle family
            rect = is_filled_rectangle(points)
            if rect:
                x, y = c0, r0
                tokens.append(ShapeToken("RECT", reg.color, ly, {"x": x, "y": y, "w": w, "h": h, "scale": 1}, reg.id))
                continue
            hollow = is_hollow_rectangle(points)
            if hollow:
                _, t = hollow
                tokens.append(ShapeToken("HOLLOW_RECT", reg.color, ly, {"x": c0, "y": r0, "w": w, "h": h, "border_thickness": t}, reg.id))
                continue
            if is_border_frame(points, (0,0,H-1,W-1)):
                tokens.append(ShapeToken("BORDER", reg.color, ly, {"x":0, "y":0, "w":W, "h":H, "thickness":1}, reg.id))
                continue

            # 2) Lines
            ln = is_line(points)
            if ln:
                orientation, rr, cc = ln
                if orientation == 'H':
                    tokens.append(ShapeToken("LINE", reg.color, ly, {"orientation":"H","x": c0, "y": rr, "w": w, "h": 1}, reg.id))
                else:
                    tokens.append(ShapeToken("LINE", reg.color, ly, {"orientation":"V","x": cc, "y": r0, "w": 1, "h": h}, reg.id))
                continue
            dln = is_diag_45(points)
            if dln:
                orientation, (sr,sc), length = dln
                tokens.append(ShapeToken("DIAG_LINE", reg.color, ly, {"x": sc, "y": sr, "length": length, "orientation": orientation}, reg.id))
                continue
            if is_x_cross(points):
                tokens.append(ShapeToken("DIAG_CROSS_X", reg.color, ly, {"x": c0, "y": r0, "w": w, "h": h}, reg.id))
                continue

            # 3) Polyominoes (with scale)
            poly = match_polyomino(points)
            if poly:
                typ, sid, s, uh, uw = poly
                tokens.append(ShapeToken(typ, reg.color, ly, {"shape_id": sid, "x": c0, "y": r0, "scale": s, "unit_h": uh, "unit_w": uw}, reg.id))
                continue

            # 4) Patterns / meta primitives
            chk = is_checkerboard_region(points, reg.color, grid)
            if chk:
                tokens.append(ShapeToken("CHECKER", reg.color, ly, chk, reg.id))
                continue
            conc = is_concentric_rectangles(points)
            if conc:
                tokens.append(ShapeToken("CONCENTRIC_RECTS", reg.color, ly, conc, reg.id))
                continue
            cu = is_c_or_u_shape(points)
            if cu:
                tokens.append(ShapeToken("C_SHAPE", reg.color, ly, cu, reg.id))
                continue
            spdots = is_sparse_dots(points)
            if spdots:
                tokens.append(ShapeToken("SPARSE_DOTS", reg.color, ly, {"positions": sorted([(c,r) for (r,c) in spdots])}, reg.id))
                continue
            xh = is_crosshatch(points)
            if xh:
                tokens.append(ShapeToken("CROSSHATCH", reg.color, ly, xh, reg.id))
                continue
            rad = detect_radial(points)
            if rad:
                tokens.append(ShapeToken("RADIAL", reg.color, ly, rad, reg.id))
                continue
            zz = detect_zigzag_or_wave(points)
            if zz:
                tokens.append(ShapeToken("ZIGZAG", reg.color, ly, zz, reg.id))
                continue
            sp = detect_spiral(points)
            if sp:
                tokens.append(ShapeToken("SPIRAL", reg.color, ly, sp, reg.id))
                continue

            # 5) Fallback: generic region with bbox & mask size
            tokens.append(ShapeToken("REGION", reg.color, ly, {"x": c0, "y": r0, "w": w, "h": h, "size": len(points)}, reg.id))

        # Post-processing: merge diagonal single pixels into diagonal lines
        if self.cfg.merge_diagonal_lines:
            tokens = merge_diagonal_single_pixels(tokens)

        # Meta: symmetry per region
        meta: List[Dict] = []
        if self.cfg.emit_symmetry_meta:
            for reg in regions:
                feats = has_symmetry(reg.pixels)
                for (key, val) in feats:
                    meta.append({"type":"META", "meta": key, "value": val, "region": reg.id})

        # Meta: tiling at grid level
        tiling = detect_repeating_tiles(grid)
        if tiling:
            meta.append({"type":"META","meta":"tiling","value":tiling})

        # Relations
        rels: List[Dict] = []
        if self.cfg.emit_relations:
            rels = relations_between(tokens)

        return {
            "shapes": [t.as_dict() for t in tokens],
            "relations": rels,
            "meta": meta
        }

# -------------#
# Quick demo   #
# -------------#
if __name__ == "__main__":
    # Tiny example grid (3 colors). Replace with a real ARC/LAGO grid.
    grid = [
        [0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,0],
        [0,1,0,0,0,0,1,0],
        [0,1,0,2,2,0,1,0],
        [0,1,0,2,2,0,1,0],
        [0,1,0,0,0,0,1,0],
        [0,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0],
    ]
    tok = ArcLagoTokenizer()
    result = tok.tokenize(grid)
    from pprint import pprint
    pprint(result)
