import os, glob, argparse, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

Net = None
try:
    from train_model.model import models as Net
except Exception:
    try:
        from model import models as Net
    except Exception as e:
        print("[ERR] 모델 임포트 실패:", e)
        sys.exit(1)

def vkitti2_palette_15():
    return np.array([
        (0,   0,   0),
        (210, 0,   200),
        (90,  200, 255),
        (0,   199, 0),
        (90,  240, 0),
        (140, 140, 140),
        (100, 60,  100),
        (250, 100, 255),
        (255, 255, 0),
        (200, 200, 0),
        (255, 130, 0),
        (80,  80,  80),
        (160, 60,  60),
        (255, 127, 80),
        (0,   139, 139),
    ], dtype=np.uint8)

_CMAPS = {
    "magma":   getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_INFERNO),
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma":  cv2.COLORMAP_PLASMA,
    "turbo":   getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "cividis": getattr(cv2, "COLORMAP_CIVIDIS", cv2.COLORMAP_JET),
    "jet":     cv2.COLORMAP_JET,
    "bone":    cv2.COLORMAP_BONE,
    "gray":    None,
}

def _percentiles(x, lo=2.0, hi=98.0):
    lo_v = np.percentile(x, lo); hi_v = np.percentile(x, hi)
    if hi_v <= lo_v + 1e-6: hi_v = lo_v + 1e-6
    return lo_v, hi_v

def normalize_for_vis(depth_m, mode="p2p98", max_depth=80.0, invert=False, gamma=1.0):
    d = depth_m.astype(np.float32)
    if mode == "max":
        lo, hi = 0.0, float(max_depth)
    elif mode == "minmax":
        lo, hi = float(np.min(d)), float(np.max(d))
    else:
        lo, hi = _percentiles(d, 2.0, 98.0)
    n = (d - lo) / max(hi - lo, 1e-6)
    n = np.clip(n, 0.0, 1.0)
    if invert: n = 1.0 - n
    if abs(float(gamma) - 1.0) > 1e-6: n = np.power(n, float(gamma))
    return n

def depth_to_color(depth_m, max_depth=80.0, cmap="magma",
                   norm_mode="p2p98", invert=False, gamma=1.0, clahe=False):
    n = normalize_for_vis(depth_m, mode=norm_mode, max_depth=max_depth, invert=invert, gamma=gamma)
    u8 = (n * 255.0).astype(np.uint8)
    if clahe:
        _clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        u8 = _clahe.apply(u8)
    if cmap == "gray" or _CMAPS.get(cmap, None) is None:
        rgb = np.stack([u8, u8, u8], axis=-1)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.applyColorMap(u8, _CMAPS[cmap])

def _unwrap_state(ckpt_obj):
    if isinstance(ckpt_obj, dict) and 'state_dict' in ckpt_obj:
        state = ckpt_obj['state_dict']
    elif isinstance(ckpt_obj, dict) and 'model' in ckpt_obj and isinstance(ckpt_obj['model'], dict):
        state = ckpt_obj['model']
    elif isinstance(ckpt_obj, dict):
        state = ckpt_obj
    else:
        state = ckpt_obj
    if isinstance(state, dict):
        keys = list(state.keys())
        if keys and all(k.startswith('module.') for k in keys):
            state = {k[len('module.'):]: v for k,v in state.items()}
    return state

def detect_heads_from_state(state_keys):
    has_seg   = any(k.startswith('seg_output')   for k in state_keys)
    has_depth = any(k.startswith('depth_output') for k in state_keys)
    return has_seg, has_depth

def load_rgb(path, W, H, device, keep_bgr=False, mean=None, std=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    if not keep_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    ten = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        std_t  = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
        ten = (ten - mean_t) / std_t
    return ten.unsqueeze(0).to(device), img

def colorize_mask(mask, palette_rgb):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    n = min(len(palette_rgb), int(mask.max())+1)
    for c in range(n):
        out[mask == c] = palette_rgb[c]
    return out

def extract_depth_tensor(out):
    if isinstance(out, torch.Tensor):
        if out.ndim == 4 and out.shape[1] == 1:
            return out
    elif isinstance(out, (list, tuple)):
        for t in out:
            if isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[1] == 1:
                return t
    elif isinstance(out, dict):
        for k in ('depth', 'pred_depth', 'pred', 'result'):
            if k in out:
                v = out[k]
                if isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[1] == 1:
                    return v
                if isinstance(v, (list, tuple)):
                    for t in v:
                        if isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[1] == 1:
                            return t
                if isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, torch.Tensor) and vv.ndim == 4 and vv.shape[1] == 1:
                            return vv
        for vv in out.values():
            if isinstance(vv, torch.Tensor) and vv.ndim == 4 and vv.shape[1] == 1:
                return vv
    return None

def extract_seg_logits(out, min_classes=2):
    if isinstance(out, torch.Tensor):
        if out.ndim == 4 and out.shape[1] >= min_classes:
            return out
    elif isinstance(out, (list, tuple)):
        for t in out:
            if isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[1] >= min_classes:
                return t
    elif isinstance(out, dict):
        for v in list(out.values()):
            if isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[1] >= min_classes:
                return v
            if isinstance(v, (list, tuple)):
                for t in v:
                    if isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[1] >= min_classes:
                        return t
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, torch.Tensor) and vv.ndim == 4 and vv.shape[1] >= min_classes:
                        return vv
    return None

def save_depth_images(depth_m, out_dir, base, max_depth, cmap, norm_mode,
                      invert, gamma, clahe, save_color=True, save_gray=True):
    if save_color:
        color = depth_to_color(depth_m, max_depth=max_depth, cmap=cmap,
                               norm_mode=norm_mode, invert=invert,
                               gamma=gamma, clahe=clahe)
        cv2.imwrite(os.path.join(out_dir, f"{base}_pred_depth_color.png"), color)

    if save_gray:
        vis8 = (normalize_for_vis(depth_m, mode=norm_mode, max_depth=max_depth,
                                  invert=False, gamma=gamma) * 255.0)
        vis8 = np.clip(vis8, 0, 255).astype(np.uint8)
        if clahe:
            _clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            vis8 = _clahe.apply(vis8)
        cv2.imwrite(os.path.join(out_dir, f"{base}_pred_depth_gray.png"), vis8)

def save_seg_images(seg_mask, rgb_for_overlay, out_dir, base,
                    overlay_alpha, save_mask=True, save_overlay=True):
    pal = vkitti2_palette_15()
    seg_rgb = colorize_mask(seg_mask, pal)

    if save_mask:
        cv2.imwrite(
            os.path.join(out_dir, f"{base}_pred_seg_mask.png"),
            cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR),
        )

    if save_overlay:
        ov = (overlay_alpha * seg_rgb.astype(np.float32) +
              (1 - overlay_alpha) * rgb_for_overlay.astype(np.float32))
        ov = np.clip(ov, 0, 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(out_dir, f"{base}_pred_seg_overlay.png"),
            cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
        )

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, help="단일 이미지 경로")
    ap.add_argument("--rgb_dir", type=str, help="폴더 내 모든 이미지 처리(jpg/jpeg/png)")
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--encoder", required=True, type=str)
    ap.add_argument("--decoder", required=True, type=str)
    ap.add_argument("--width",  required=True, type=int)
    ap.add_argument("--height", required=True, type=int)
    ap.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std",  nargs=3, type=float, default=[0.229, 0.224, 0.225])
    ap.add_argument("--keep_bgr", action="store_true")
    ap.add_argument("--task", default="auto", choices=["auto","depth","seg","both"])
    ap.add_argument("--out-dir", default="out_images", type=str)
    ap.add_argument("--overlay-alpha", type=float, default=0.55)
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--cmap", default="magma", choices=list(_CMAPS.keys()))
    ap.add_argument("--norm-mode", default="p2p98", choices=["p2p98","minmax","max"])
    ap.add_argument("--invert-depth", action="store_true")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--save-raw-depth", action="store_true",
                    help="원하면 *_pred_depth_cm16.png, *_pred_depth.npy도 저장")
    ap.add_argument("--only-gray-depth", action="store_true",
                    help="depth 시 gray 이미지만 저장 (color map 미저장)")
    ap.add_argument("--only-seg-mask", action="store_true",
                    help="seg 시 seg_mask만 저장 (overlay 미저장)")

    args = ap.parse_args()

    if not args.img and not args.rgb_dir:
        raise SystemExit("하나 선택: --img 또는 --rgb_dir")

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_ckpt = torch.load(args.weights, map_location="cpu")
    state = _unwrap_state(raw_ckpt)
    has_seg, has_depth = (False, False)
    if isinstance(state, dict):
        has_seg, has_depth = detect_heads_from_state(list(state.keys()))

    cfg = {
        "model": {"encoder": args.encoder, "decoder": args.decoder},
        "model_conf": {
            "Seg":   True,
            "Depth": True,
            "od": False,
            "num_seg_classes": 15,
            "max_depth": float(args.max_depth),
        }
    }
    net = Net(cfg).to(device)
    model_sd = net.state_dict()
    safe_state = {
        k: v for k, v in state.items()
        if (k in model_sd and getattr(v, "shape", None) == model_sd[k].shape)
    }
    net.load_state_dict(safe_state, strict=False)
    net.eval()

    run_depth = run_seg = False
    if args.task == "both":
        run_depth = True
        run_seg = True
    elif args.task == "depth":
        run_depth = True
    elif args.task == "seg":
        run_seg = True
    else:
        run_depth = has_depth
        run_seg   = has_seg
        if not run_depth and not run_seg:
            run_depth = run_seg = True

    if args.img:
        img_list = [args.img]
    else:
        exts = ("*.jpg","*.jpeg","*.png","*.bmp")
        img_list = []
        for e in exts:
            img_list.extend(glob.glob(os.path.join(args.rgb_dir, e)))
        img_list.sort()

    for ip in img_list:
        base = os.path.splitext(os.path.basename(ip))[0]
        x, rgb = load_rgb(
            ip, args.width, args.height, device,
            keep_bgr=bool(args.keep_bgr),
            mean=args.mean, std=args.std,
        )
        out = net(x)

        if run_depth:
            depth_t = extract_depth_tensor(out)
            if depth_t is not None:
                if depth_t.shape[-2:] != (args.height, args.width):
                    depth_t = F.interpolate(
                        depth_t, size=(args.height, args.width),
                        mode='bilinear', align_corners=False,
                    )
                depth = depth_t[0, 0].float().clamp(min=1e-6).cpu().numpy()
                save_depth_images(
                    depth, args.out_dir, base,
                    max_depth=float(args.max_depth),
                    cmap=args.cmap, norm_mode=args.norm_mode,
                    invert=bool(args.invert_depth),
                    gamma=float(args.gamma),
                    clahe=bool(args.clahe),
                    save_color=(not args.only_gray_depth),
                    save_gray=True,
                )
                if args.save_raw_depth:
                    np.save(
                        os.path.join(args.out_dir, f"{base}_pred_depth.npy"),
                        depth.astype(np.float32),
                    )
                    cm16 = np.clip(
                        np.round(depth * 100.0),
                        0, args.max_depth * 100.0
                    ).astype(np.uint16)
                    cv2.imwrite(
                        os.path.join(args.out_dir, f"{base}_pred_depth_cm16.png"),
                        cm16,
                    )

        if run_seg:
            seg_logits = extract_seg_logits(out, min_classes=2)
            if seg_logits is not None:
                if seg_logits.shape[-2:] != (args.height, args.width):
                    seg_logits = F.interpolate(
                        seg_logits, size=(args.height, args.width),
                        mode='bilinear', align_corners=False,
                    )
                seg_mask = seg_logits[0].argmax(0).cpu().numpy().astype(np.uint8)
                save_seg_images(
                    seg_mask, rgb, args.out_dir, base,
                    overlay_alpha=float(args.overlay_alpha),
                    save_mask=True,
                    save_overlay=(not args.only_seg_mask),
                )

    print(f"[DONE] Saved images to: {args.out_dir}")

if __name__ == "__main__":
    main()
