"""PCA-RANSAC multi-way stroke initialization from Appendix C."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2
from scipy import ndimage

from impasto.actions import Stroke
from impasto.sim.paint import PALETTE


@dataclass(frozen=True)
class StrokeProposal:
    action: np.ndarray
    color: int
    opacity: float
    mask: np.ndarray
    score: float
    model: str


def _kmeans(points, k, iterations, rng):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= k:
        return np.arange(len(points)) % max(k, 1), points.copy()
    centers = np.empty((k, points.shape[1]), dtype=np.float64)
    centers[0] = points[int(rng.integers(len(points)))]
    distances = np.full(len(points), np.inf)
    for i in range(1, k):
        distances = np.minimum(distances, np.sum((points - centers[i-1]) ** 2, axis=1))
        probabilities = distances / max(float(distances.sum()), 1e-12)
        centers[i] = points[int(rng.choice(len(points), p=probabilities))]
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(iterations):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = points[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels, centers


def _pca_frame(points):
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False) if len(points) > 2 else np.eye(2)
    values, vectors = np.linalg.eigh(covariance)
    v0 = vectors[:, int(np.argmax(values))]
    v1 = np.array([-v0[1], v0[0]])
    return center, v0, v1


def _ransac_curve(points, iterations, threshold, rng):
    center, v0, v1 = _pca_frame(points)
    local = np.stack(((points - center) @ v0, (points - center) @ v1), axis=1)
    candidates = []
    for degree, sample_count in ((1, 2), (2, 3)):
        if len(local) < sample_count:
            continue
        best = None
        for _ in range(iterations):
            sample = local[rng.choice(len(local), sample_count, replace=False)]
            if len(np.unique(sample[:,0].round(5))) < sample_count:
                continue
            try:
                coeff = np.polyfit(sample[:, 0], sample[:, 1], degree)
            except np.linalg.LinAlgError:
                continue
            residual = np.abs(local[:, 1] - np.polyval(coeff, local[:, 0]))
            inliers = residual <= threshold
            count = int(inliers.sum())
            median = float(np.median(residual[inliers])) if count else float("inf")
            score = (count, -median)
            if best is None or score > best[0]:
                best = (score, coeff, inliers, residual)
        if best is None:
            continue
        inliers = best[2]
        if int(inliers.sum()) >= sample_count:
            try:
                coeff = np.polyfit(local[inliers, 0], local[inliers, 1], degree)
            except np.linalg.LinAlgError:
                coeff = best[1]
            residual = np.abs(local[:, 1] - np.polyval(coeff, local[:, 0]))
            candidates.append((degree, coeff, residual <= threshold, residual))
    if not candidates:
        raise ValueError("RANSAC could not fit a candidate curve")
    candidates.sort(key=lambda item: (int(item[2].sum()), -float(np.median(item[3][item[2]]))), reverse=True)
    degree, coeff, inliers, residual = candidates[0]
    return center, v0, v1, degree, coeff, inliers, residual


def _pixel_to_world(points, width, height, canvas_width, canvas_height):
    points = np.asarray(points, dtype=np.float64)
    return np.stack([
        (points[..., 0] / max(width - 1, 1) - .5) * canvas_width,
        (.5 - points[..., 1] / max(height - 1, 1)) * canvas_height,
    ], axis=-1)


def _infer_color(target, base, mask, color_subset=None, target_rgb=None, base_rgb=None,
                 palette=None):
    palette = np.asarray(PALETTE if palette is None else palette, dtype=np.float32)
    if palette.max() > 1.0:
        palette = palette / 255.0
    if target_rgb is not None and base_rgb is not None:
        color_ids = np.asarray(color_subset if color_subset is not None else range(len(palette)), dtype=int)
        # Appendix B's transparency test compares the observed change with a
        # palette-to-base contrast. This matters for the published photos:
        # an opaque pigment is recorded as a pale, partially mixed pixel.
        base_pixels = np.asarray(base_rgb, dtype=np.float32)[mask]
        target_pixels = np.asarray(target_rgb, dtype=np.float32)[mask]
        if max(float(base_pixels.max()), float(target_pixels.max())) > 1:
            base_pixels, target_pixels = base_pixels / 255.0, target_pixels / 255.0
        linear = lambda x: np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
        base_pixels, target_pixels = linear(base_pixels), linear(target_pixels)
        weights = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)[mask]
        weights = np.maximum(weights, .1)
        delta = target_pixels - base_pixels
        errors = []
        for color_id in color_ids:
            contrast = linear(palette[int(color_id)])[None] - base_pixels
            denom = np.maximum(np.sum(contrast * contrast, axis=1), 1e-10)
            opacity = np.clip(np.sum(contrast * delta, axis=1) / denom, 0.0, 1.0)
            residual = np.sum((delta - opacity[:, None] * contrast) ** 2, axis=1)
            # The centerline is more reliable than antialiased boundaries;
            # trim the worst residuals before applying distance weights.
            cutoff = np.quantile(residual, .75) if len(residual) > 12 else np.inf
            keep = residual <= cutoff
            errors.append(float(np.average(residual[keep], weights=weights[keep])))
        return int(color_ids[int(np.argmin(errors))])
    pixels = target[mask]
    base_pixels = base[mask]
    if not len(pixels):
        return 0
    # A grayscale target is insufficient to identify all pigments uniquely.
    # Nearest luminance is the explicit local fallback; RGB targets can use
    # the separate color database in a future real-data run.
    delta = float(np.median(pixels - base_pixels))
    luminance = palette @ np.array([.299, .587, .114])
    color_ids = np.asarray(color_subset if color_subset is not None else range(len(PALETTE)), dtype=int)
    expected_delta = luminance[color_ids] - float(np.median(base_pixels))
    return int(color_ids[np.argmin(np.abs(expected_delta - delta))])


def _infer_opacity(color, mask, target_rgb, base_rgb, palette):
    """Estimate the Appendix-B transparency ratio from linear-RGB changes."""
    if target_rgb is None or base_rgb is None:
        return 1.0
    target_pixels = np.asarray(target_rgb, dtype=np.float32)[mask]
    base_pixels = np.asarray(base_rgb, dtype=np.float32)[mask]
    if not len(target_pixels):
        return 1.0
    if max(float(target_pixels.max()), float(base_pixels.max())) > 1:
        target_pixels, base_pixels = target_pixels / 255.0, base_pixels / 255.0
    linear = lambda x: np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
    observed = linear(target_pixels)
    base = linear(base_pixels)
    pigment = np.asarray(palette, dtype=np.float32)[int(color)]
    if pigment.max() > 1:
        pigment = pigment / 255.0
    contrast = linear(pigment)[None] - base
    ratio = np.median(np.sum(np.abs(observed - base), axis=1)) / max(
        float(np.median(np.sum(np.abs(contrast), axis=1))), 1e-6)
    return float(np.clip(ratio, .12, 1.0))


def _pixel_color_labels(target_rgb, base_rgb, changed, color_subset, palette):
    """Assign changed pixels to the palette colour with lowest RGB residual."""
    target_pixels = np.asarray(target_rgb, dtype=np.float32)[changed]
    base_pixels = np.asarray(base_rgb, dtype=np.float32)[changed]
    if max(float(target_pixels.max()), float(base_pixels.max())) > 1:
        target_pixels, base_pixels = target_pixels / 255.0, base_pixels / 255.0
    linear = lambda x: np.where(x <= .04045, x / 12.92,
                                ((x + .055) / 1.055) ** 2.4)
    target_pixels, base_pixels = linear(target_pixels), linear(base_pixels)
    palette = np.asarray(palette, dtype=np.float32)
    if palette.max() > 1:
        palette = palette / 255.0
    delta = target_pixels - base_pixels
    errors = []
    for color_id in color_subset:
        contrast = linear(palette[int(color_id)])[None] - base_pixels
        denom = np.maximum(np.sum(contrast * contrast, axis=1), 1e-10)
        opacity = np.clip(np.sum(contrast * delta, axis=1) / denom, 0.0, 1.0)
        errors.append(np.sum((delta - opacity[:, None] * contrast) ** 2, axis=1))
    return np.asarray(color_subset, dtype=int)[np.argmin(np.asarray(errors), axis=0)]


def _centerline_mask(shape, points, width=3):
    mask = np.zeros(shape, dtype=np.uint8)
    pixels = np.rint(np.asarray(points)).astype(np.int32)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, shape[1] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, shape[0] - 1)
    cv2.polylines(mask, [pixels.reshape(-1, 1, 2)], False, 1,
                  thickness=int(width), lineType=cv2.LINE_AA)
    return mask.astype(bool)


def split_difference(base, target, components=5, threshold=.015, min_pixels=18,
                     kmeans_iterations=25, ransac_iterations=80,
                     ransac_threshold_px=2.5, canvas_width=.4572,
                     canvas_height=.5334, action_bounds=None, seed=0,
                     color_hints=None, color_subset=None, base_rgb=None, target_rgb=None,
                     palette=None, color_guided=False, opaque_colors=False,
                     radius_config=None, curvature_penalty=.01, overlap_penalty=.15,
                     robust_candidates=False):
    """Return disjoint PCA-RANSAC stroke proposals in world action units."""
    base = np.asarray(base, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if base.shape != target.shape or base.ndim != 2:
        raise ValueError("base and target must be same-size grayscale images")
    if (base_rgb is None) != (target_rgb is None):
        raise ValueError("base_rgb and target_rgb must be supplied together")
    if base_rgb is not None:
        expected = (*base.shape, 3)
        if np.asarray(base_rgb).shape != expected or np.asarray(target_rgb).shape != expected:
            raise ValueError(f"RGB color references must have shape {expected}")
    if robust_candidates:
        return _multiway_candidates(base, target, components, threshold, min_pixels,
            kmeans_iterations, ransac_iterations, ransac_threshold_px, canvas_width,
            canvas_height, action_bounds, seed, color_subset, base_rgb, target_rgb,
            palette, radius_config, curvature_penalty, overlap_penalty,
            opaque_colors=opaque_colors)
    changed = np.abs(target - base) > threshold
    # Paint cannot correct an excess dark mark by drawing another dark mark
    # over empty canvas. Use only goal foreground to seed new pigment strokes;
    # the full-canvas objective still penalizes all overpainting.
    background = float(np.quantile(target, .95))
    changed &= target < background - threshold
    changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8)).astype(bool)
    labels_cc, ncc = ndimage.label(changed)
    sizes = np.bincount(labels_cc.ravel())
    changed &= sizes[labels_cc] >= min_pixels
    yy, xx = np.nonzero(changed)
    if len(xx) < min_pixels:
        return []
    points = np.stack([xx, yy], axis=1).astype(np.float64)
    height, width = base.shape
    k = min(int(components), max(1, len(points) // min_pixels))
    rng = np.random.default_rng(seed)
    color_groups = []
    pixel_labels = None
    if color_guided and base_rgb is not None and color_subset:
        pixel_labels = _pixel_color_labels(target_rgb, base_rgb, changed,
                                           color_subset,
                                           np.asarray(PALETTE if palette is None else palette))
        changed_coords = np.stack([yy, xx], axis=1)
        for color_id in color_subset:
            color_mask = np.zeros_like(changed)
            selected = pixel_labels == int(color_id)
            color_mask[changed_coords[selected, 0], changed_coords[selected, 1]] = True
            labels_color, n_color = ndimage.label(color_mask)
            sizes_color = np.bincount(labels_color.ravel())
            for label_id in range(1, n_color + 1):
                group = labels_color == label_id
                if int(sizes_color[label_id]) < max(8, min_pixels // 2):
                    continue
                score = float(np.abs(target - base)[group].sum())
                color_groups.append((score, group, int(color_id)))
        color_groups.sort(key=lambda item: item[0], reverse=True)
        color_groups = color_groups[:k]
    # Multiple k-means++ starts avoid a single poor spatial partition. This
    # still uses the paper's [x/W, y/H] spatial feature, without target actions.
    normalized = points / np.array([width, height])
    # Allocate at least one stroke to each substantial connected component,
    # then run the paper's spatial k-means within it. This constrained region
    # growing prevents a small isolated petal from losing its seed to a larger
    # neighbour. No annotation or hand-coded geometry is used.
    cc = labels_cc[yy, xx]
    ids = [i for i in np.unique(cc) if i and np.sum(cc == i) >= min_pixels]
    ids.sort(key=lambda i:np.sum(np.abs(target-base)[yy[cc==i],xx[cc==i]]),reverse=True)
    ids=ids[:k]
    allocations=np.ones(len(ids),dtype=int)
    for _ in range(k-len(ids)):
        j=int(np.argmax([np.sum(cc==i)/allocations[q]**2 for q,i in enumerate(ids)]))
        allocations[j]+=1
    labels=np.full(len(points),-1,dtype=int)
    offset=0
    if color_groups:
        for _, group, _ in color_groups:
            rows = np.flatnonzero(group[yy, xx])
            labels[rows] = offset
            offset += 1
        k = len(color_groups)
    else:
        for cid,n in zip(ids,allocations):
            rows=np.flatnonzero(cc==cid)
            pts=normalized[rows]
            best=None
            for _ in range(12):
                lab,centers=_kmeans(pts,int(n),kmeans_iterations,rng)
                inertia=float(np.sum((pts-centers[lab])**2))
                if best is None or inertia<best[0]:best=inertia,lab.copy()
            labels[rows]=best[1]+offset
            offset+=n
    proposals = []
    for component in range(k):
        group = np.zeros_like(changed)
        members = labels == component
        group[yy[members], xx[members]] = True
        # Grow from the centerline only within this spatial segment; retain
        # one connected stroke and discard unrelated specks in the segment.
        connected, nc = ndimage.label(group)
        areas = np.bincount(connected.ravel()); areas[0] = 0
        if nc == 0 or areas.max() < min_pixels:
            continue
        group = connected == areas.argmax()
        gy, gx = np.nonzero(group)
        component_points = np.stack([gx, gy], axis=1).astype(float)
        center, v0, v1 = _pca_frame(component_points)
        local = np.stack(((component_points-center) @ v0,
                          (component_points-center) @ v1), axis=1)
        lo, hi = np.percentile(local[:,0], [4, 96])
        if hi-lo < 2:
            continue
        # A raw RANSAC fit to all thick-mask pixels often selects an EDGE of
        # a broad brushstroke. Seed with transverse medians, then robustly fit
        # the centerline and grow it back to the connected mask (Appendix C).
        edges = np.linspace(lo, hi, max(7, min(24, int((hi-lo)/2))))
        centers = []
        for left, right in zip(edges[:-1], edges[1:]):
            subset = local[(local[:,0] >= left) & (local[:,0] <= right)]
            if len(subset) >= 2:
                centers.append(np.median(subset, axis=0))
        if len(centers) < 3:
            continue
        centers = np.asarray(centers)
        curve_points = center + centers[:,0,None]*v0 + centers[:,1,None]*v1
        try:
            c, u, v, degree, coeff, inliers, residual = _ransac_curve(
                curve_points, ransac_iterations, max(1.0, ransac_threshold_px), rng)
        except ValueError:
            continue
        local_curve = (curve_points-c) @ u
        s_lo, s_hi = np.percentile(local_curve, [0,100])
        t_lo, t_hi = np.polyval(coeff, [s_lo,s_hi])
        p0_px = c + s_lo*u + t_lo*v
        p2_px = c + s_hi*u + t_hi*v
        mid_px = c + (s_lo+s_hi)/2*u + np.polyval(coeff,(s_lo+s_hi)/2)*v
        p0,p2,pm = _pixel_to_world(np.stack([p0_px,p2_px,mid_px]),width,height,canvas_width,canvas_height)
        tangent = p2-p0
        length = float(np.linalg.norm(tangent))
        if length < .002:
            continue
        angle = float(np.arctan2(tangent[1],tangent[0]))
        normal = np.array([-np.sin(angle),np.cos(angle)])
        bend = float(2*np.dot(pm-(p0+p2)/2,normal))
        # Infer an initial force from connected-mask area/centerline length.
        # The learned MPC, not this geometric estimate, selects final force.
        area_m2 = group.sum()*canvas_width*canvas_height/((width-1)*(height-1))
        radius = (np.sqrt(length**2 + np.pi*area_m2)-length)/np.pi
        radius = max(radius, .0005)
        # Area is only a seed width estimate; MPC may search the full force range.
        rc = radius_config or {"minimum": .0004, "scale": .0058, "exponent": 1.65}
        force = max((radius-rc["minimum"])/rc["scale"], 0.0)**(1.0/rc["exponent"])
        action = np.array([p0[0],p0[1],length,bend,angle,force],dtype=np.float32)
        if action_bounds:
            for idx,key in ((0,'x_m'),(1,'y_m'),(2,'length_m'),(3,'bend_m'),(5,'force_N')):
                action[idx]=np.clip(action[idx],*action_bounds[key])
        # The full connected mask can contain overlapping strokes.  Color and
        # alpha are attributes of the stroke centerline, so classify a narrow
        # centerline band first and fall back to the full group when it is too
        # small. This follows the paper's centerline weighted-patch idea more
        # closely than averaging every antialiased boundary pixel.
        centerline_px = np.stack([
            c + value * u + np.polyval(coeff, value) * v
            for value in np.linspace(s_lo, s_hi, 64)
        ])
        centerline = _centerline_mask(group.shape, centerline_px, width=5)
        color_mask = centerline & changed
        if int(color_mask.sum()) < max(12, min_pixels // 2):
            color_mask = group
        guided_color = (color_groups[component][2] if color_groups else None)
        if guided_color is None and pixel_labels is not None:
            group_labels = pixel_labels[group[changed]]
            if len(group_labels):
                values, counts = np.unique(group_labels, return_counts=True)
                guided_color = int(values[int(np.argmax(counts))])
        color = (int(color_hints[component]) if color_hints is not None and component in color_hints
                 else int(guided_color) if guided_color is not None
                 else _infer_color(target,base,color_mask,color_subset,target_rgb,base_rgb,palette))
        if opaque_colors and target_rgb is not None:
            palette_array = np.asarray(PALETTE if palette is None else palette, dtype=float)
            if palette_array.max() > 1: palette_array /= 255
            pixels = np.asarray(target_rgb, dtype=float)[color_mask] / 255
            weights = cv2.distanceTransform(group.astype(np.uint8), cv2.DIST_L2, 5)[color_mask]
            mu = np.average(pixels, axis=0, weights=np.maximum(weights, .1))
            ids = list(color_subset or range(len(palette_array)))
            color = ids[int(np.argmin(np.sum((palette_array[ids]-mu)**2, axis=1)))]
        opacity = 1.0 if opaque_colors else _infer_opacity(color, color_mask, target_rgb, base_rgb,
                                 np.asarray(PALETTE if palette is None else palette))
        # Coverage is absolute unexplained image mass, not inlier fraction;
        # the latter consistently ranks tiny tail details before the fish body.
        score = float(np.abs(target-base)[group].sum() - curvature_penalty*abs(bend))
        proposals.append(StrokeProposal(action, color, opacity, group, score,
                                        'quadratic' if degree==2 else 'line'))
    proposals.sort(key=lambda p:p.score,reverse=True)
    return proposals


def _footprint(action, shape, canvas_width, canvas_height, radius_config):
    """Cheap disk footprint for proposal coverage only; never an execution model."""
    rc = radius_config or {"minimum": .0004, "scale": .0058, "exponent": 1.65}
    stroke = Stroke(*map(float, action))
    radius = rc["minimum"] + rc["scale"] * stroke.force**rc["exponent"]
    points = stroke.points(np.linspace(0, 1, 100))
    h, w = shape
    mask = np.zeros(shape, np.uint8)
    px = (points[:, 0]/canvas_width+.5)*(w-1)
    py = (.5-points[:, 1]/canvas_height)*(h-1)
    # Physical disks are ellipses in a non-square canvas texture.
    axes = (max(1, round(radius/canvas_width*(w-1))),
            max(1, round(radius/canvas_height*(h-1))))
    for x, y in zip(px, py):
        cv2.ellipse(mask, (round(x), round(y)), axes, 0, 0, 360, 1, -1)
    return mask.astype(bool)


def _multiway_candidates(base, target, components, threshold, min_pixels,
                         kmeans_iterations, ransac_iterations, ransac_threshold_px,
                         canvas_width, canvas_height, bounds, seed, color_subset,
                         base_rgb, target_rgb, palette, radius_config,
                         curvature_penalty, overlap_penalty, opaque_colors=False):
    """Multi-start spatial PCA/RANSAC with explicit coverage and overlap scoring.

    Spatial k-means still provides geometry. Coarse color-connected regions add
    candidates so a color boundary cannot force one cross-part averaged stroke.
    This supplementary grouping and RGB seed score are local extensions.
    """
    if base_rgb is None:
        return split_difference(base, target, components, threshold, min_pixels,
            kmeans_iterations, ransac_iterations, ransac_threshold_px,
            canvas_width, canvas_height, bounds, seed, color_subset=color_subset,
            radius_config=radius_config, opaque_colors=opaque_colors)
    pal = np.asarray(palette, np.float32)
    if pal.max() > 1: pal /= 255
    rgb0 = np.asarray(base_rgb, np.float32)/255
    rgb1 = np.asarray(target_rgb, np.float32)/255
    error = np.mean(np.abs(rgb1-rgb0), axis=-1)
    bg = np.median(rgb1.reshape(-1,3), axis=0)
    foreground = np.max(np.abs(rgb1-bg), axis=-1) > .06
    changed = (error > threshold) & foreground
    # We will not use color labels as action labels. They only stop unrelated
    # hues from being averaged into an invented pigment in an initial seed.
    hsv = cv2.cvtColor(rgb1, cv2.COLOR_RGB2HSV)
    hue_family = np.floor((hsv[...,0]+20)%360/40).astype(int)
    # Keep gray/dark overpainting separate from the light blue petal core.
    lab = cv2.cvtColor(rgb1, cv2.COLOR_RGB2Lab)
    family = hue_family*3 + np.digitize(lab[...,0], [53, 73])
    groups = [changed]
    for labels in (hue_family, family):
        for label in np.unique(labels[changed]):
            # Keep the connected target colour region for PCA/RANSAC so a
            # prefix does not break one physical stroke into short fragments.
            # The current residual is still used below for force estimation
            # and candidate ranking.
            mask = foreground & (labels == label)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                    np.ones((3,3), np.uint8)).astype(bool)
            ids, nc = ndimage.label(mask)
            for i in range(1,nc+1):
                group = ids==i
                if (group & changed).sum() >= min_pixels:
                    groups.append(group)
    candidates = []
    for gi, group in enumerate(groups):
        # Isolate each residual region in the difference image without
        # changing target RGB. The recursive splitter now estimates geometry
        # from the residual itself instead of from the full target blob.
        fake_base = target.copy()
        fake_base[group] = target[group]+.2
        ks = (sorted(set([components, components+2, components*2])) if gi == 0
              else list(range(1, components + 1)))
        for k in ks:
            if group.sum() < k*min_pixels: continue
            candidates.extend((proposal, group) for proposal in split_difference(
                fake_base, target, k, threshold=.03, min_pixels=min_pixels,
                kmeans_iterations=kmeans_iterations,
                ransac_iterations=ransac_iterations,
                ransac_threshold_px=ransac_threshold_px,
                canvas_width=canvas_width, canvas_height=canvas_height,
                action_bounds=bounds, seed=seed+gi+k, color_subset=color_subset,
                base_rgb=base_rgb, target_rgb=target_rgb, palette=palette,
                opaque_colors=opaque_colors, radius_config=radius_config))
    # Rank by actual explained residual under a pigment footprint, rather than
    # by component size; subtract curved detours and already-covered area.
    pool=[]
    for proposal, source_group in candidates:
        action=proposal.action.copy()
        if bounds:
            for j,key in ((0,"x_m"),(1,"y_m"),(2,"length_m"),(3,"bend_m"),(5,"force_N")):
                action[j]=np.clip(action[j],*bounds[key])
        # The full target blob is useful for a continuous centerline, but its
        # area includes paint already present in the published prefix. Estimate
        # width from the unexplained portion whenever it is large enough.
        residual_group = source_group & changed
        width_group = (residual_group if int(residual_group.sum()) >= min_pixels
                       else source_group)
        length = max(float(action[2]), 1e-6)
        area_m2 = width_group.sum()*canvas_width*canvas_height / max(
            (base.shape[1]-1)*(base.shape[0]-1), 1)
        radius = (np.sqrt(length**2 + np.pi*area_m2)-length)/np.pi
        rc = radius_config or {"minimum": .0004, "scale": .0058, "exponent": 1.65}
        action[5] = max((max(radius, .0005)-rc["minimum"])/rc["scale"], 0.0)**(
            1.0/rc["exponent"])
        if bounds:
            action[5]=np.clip(action[5],*bounds["force_N"])
        mask=_footprint(action,base.shape,canvas_width,canvas_height,radius_config)
        core=mask & residual_group
        if core.sum()<min_pixels: continue
        weights=cv2.distanceTransform(source_group.astype(np.uint8),cv2.DIST_L2,5)[core]
        pixels=rgb1[core]
        ids=list(color_subset or range(len(pal)))
        # L1 prototype classification is robust to mixed antialiased boundaries.
        # A hue-distance term avoids replacing a green leaf by a similar gray.
        htarget=cv2.cvtColor(pixels[None].astype(np.float32),cv2.COLOR_RGB2HSV)[0]
        hpalette=cv2.cvtColor(pal[None].astype(np.float32),cv2.COLOR_RGB2HSV)[0]
        hdiff=abs(hpalette[ids,0,None]-htarget[None,:,0])
        hdiff=np.minimum(hdiff,360-hdiff)/180
        errors=np.mean(abs(pal[ids,None]-pixels[None]),axis=-1)
        errors += .10*hdiff*htarget[None,:,1]
        color=ids[int(np.argmin(np.average(errors,axis=1,weights=np.maximum(weights,.1))))]
        # A connected residual may join a petal to a stem. A long curve across
        # that junction is not a single pigment stroke, even if it covers area.
        path = Stroke(*map(float, action)).points(np.linspace(0, 1, 100))
        pixels = np.column_stack([
            (path[:, 0] / canvas_width + .5) * (base.shape[1] - 1),
            (.5 - path[:, 1] / canvas_height) * (base.shape[0] - 1),
        ])
        centerline = _centerline_mask(base.shape, pixels, width=3) & foreground
        samples = hsv[centerline]
        if len(samples):
            pigment_hsv = hpalette[color]
            hue_error = np.abs(samples[:, 0] - pigment_hsv[0])
            hue_error = np.minimum(hue_error, 360 - hue_error)
            incompatible = (hue_error > 45) & (samples[:, 1] > .06)
            if incompatible.mean() > .15:
                continue
        # Correct color/shape on the local target must be feasible before MPC.
        new_error=np.mean(np.abs(rgb1-pal[color]),axis=-1)
        missing = np.max(np.abs(rgb0-bg),axis=-1)<.04
        gain=(error-new_error)*mask*(1+3*missing)
        pool.append((action,color,mask,gain,proposal))
    selected=[];covered=np.zeros(base.shape,bool)
    for _ in range(components):
        if not pool: break
        scores=[float(gain[~covered].sum()
                      -overlap_penalty*np.maximum(gain,0)[covered].sum()
                      -curvature_penalty*abs(action[3])/canvas_width)
                for action,color,mask,gain,proposal in pool]
        j=int(np.argmax(scores));action,color,mask,gain,proposal=pool.pop(j)
        selected.append(StrokeProposal(action, color, proposal.opacity,
                                        proposal.mask, scores[j], proposal.model))
        covered |= mask
    return selected
