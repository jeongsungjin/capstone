import onnxruntime as ort
import numpy as np
import cv2

# ---- utils: letterbox (YOLO style, keep aspect ratio with padding) ----
def letterbox(img, new_shape, color=(114, 114, 114), scaleup=True, stride=32):
	h0, w0 = img.shape[:2]
	new_h, new_w = new_shape  # (H, W)
	# scale ratio (new / old) and limit scaling up if desired
	r = min(new_h / h0, new_w / w0)
	if not scaleup:
		r = min(r, 1.0)
	# compute unpadded new shape
	uw = int(round(w0 * r))
	uh = int(round(h0 * r))
	# compute padding
	dw = new_w - uw
	dh = new_h - uh
	left = int(round(dw / 2.0))
	right = dw - left
	top = int(round(dh / 2.0))
	bottom = dh - top
	# resize and pad
	if (w0, h0) != (uw, uh):
		img = cv2.resize(img, (uw, uh), interpolation=cv2.INTER_LINEAR)
	padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
	return padded, r, (left, top), (uw, uh)

# ---- 1) 세션 준비 (CUDA 있으면 GPU, 없으면 CPU) ----
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess = ort.InferenceSession("/home/ctrl/capstone_2025/src/ros-bridge/YOLOP/weights/yolop-1280-1280.onnx", sess_options=so, providers=providers)

inp_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]
print("Outputs:", out_names)

# ---- 2) 드라이버블 영역 출력 이름 추정 ----
# 흔한 후보들 예: "drive_area_seg", "da_seg", "seg", "seg_da"
cand_keys = ["drive", "drivable", "da_seg", "da", "seg"]
da_name = None
for name in out_names:
    low = name.lower()
    if any(k in low for k in cand_keys) and ("lane" not in low and "ll" not in low):
        da_name = name
        break
assert da_name is not None, f"드라이버블 영역 출력명을 찾지 못했습니다. 가능한 출력: {out_names}"

# ---- 3) 전처리 (모델 입력 크기에서 직접 추출 + letterbox) ----
img = cv2.imread("/home/ctrl/capstone_2025/src/ros-bridge/YOLOP/test.jpg")  # BGR
h0, w0 = img.shape[:2]
inp_shape = sess.get_inputs()[0].shape  # [N, C, H, W] 기대
try:
	inp_h = int(inp_shape[2]) if isinstance(inp_shape[2], int) else 1280
	inp_w = int(inp_shape[3]) if isinstance(inp_shape[3], int) else 1280
except Exception:
	inp_h, inp_w = 1280, 1280

# letterbox (keep aspect ratio)
img_lb, ratio, (pad_left, pad_top), (uw, uh) = letterbox(img, (inp_h, inp_w), color=(114, 114, 114), scaleup=True, stride=32)
rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
x = rgb.astype(np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))  # CHW
x = np.expand_dims(x, 0)        # NCHW

# ---- 4) 드라이버블 영역만 추론 ----
(da_logits,) = sess.run([da_name], {inp_name: x})

# 일반적으로 shape: (1, 2, Hs, Ws)  # 2-class [background, drivable] for drive_area_seg
# 일부 모델은 (1, 1, Hs, Ws) 로 확률(또는 로짓) 1채널일 수 있음 (드뭄)
da_logits = np.squeeze(da_logits, 0)
mask_debug = {"mode": None, "thresh": None, "chosen": None}
if da_logits.ndim == 3:
	# [C, H, W]
	C = da_logits.shape[0]
	if C == 1:
		# 1채널일 경우 시그모이드 후 임계값
		prob = 1.0 / (1.0 + np.exp(-da_logits[0].astype(np.float32)))
		thr = 0.5
		mask_debug.update({"mode": "sigmoid-1ch", "thresh": thr, "mean": float(prob.mean())})
		da_prob = prob
	else:
		# 2+채널: softmax, 채널1을 drivable로 가정 (YOLOP 기본)
		logits = da_logits.astype(np.float32)
		logits -= logits.max(axis=0, keepdims=True)
		exp = np.exp(logits)
		softmax = exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-6)
		drivable_idx = 1 if C > 1 else 0
		da_prob = softmax[drivable_idx]
		means = [float(softmax[i].mean()) for i in range(C)]  # debug only
		mask_debug.update({"mode": f"softmax-{C}ch", "means": [round(m, 4) for m in means], "chosen": int(drivable_idx)})
elif da_logits.ndim == 2:
	# [H, W] 로짓/확률일 가능성 -> 시그모이드 후 임계값
	da_prob = 1.0 / (1.0 + np.exp(-da_logits.astype(np.float32)))
	mask_debug.update({"mode": "sigmoid-2d"})
else:
	raise ValueError(f"예상치 못한 출력 텐서 차원: {da_logits.shape}")

# ------- 4.5) 임계값 생성 (우선 0.5, 비정상 시 Otsu/강화/반전) -------
thr = 0.5
da_mask_full = (da_prob >= thr).astype(np.uint8)

# unpad: crop to letterboxed unpadded region before evaluating coverage
Hf, Wf = da_mask_full.shape[:2]
top, left = pad_top, pad_left
bottom = top + uh
right = left + uw
top = max(0, min(Hf, top)); bottom = max(0, min(Hf, bottom))
left = max(0, min(Wf, left)); right = max(0, min(Wf, right))
crop = da_mask_full[top:bottom, left:right]
if crop.size == 0:
	crop = da_mask_full

coverage = float(crop.sum()) / max(1.0, float(crop.size))
if coverage <= 0.01 or coverage >= 0.98:
	# Try Otsu on probabilities in cropped region
	pcrop = (da_prob[top:bottom, left:right] if da_prob.shape == da_mask_full.shape else da_prob)
	if pcrop.size == 0:
		pcrop = da_prob
	pimg = (pcrop * 255.0).astype(np.uint8)
	_, otsu_bin = cv2.threshold(pimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	da_mask_crop = (otsu_bin > 0).astype(np.uint8)
	cov2 = float(da_mask_crop.sum()) / max(1.0, float(da_mask_crop.size))
	if 0.01 <= cov2 <= 0.98:
		# place back into full mask shape
		da_mask_full[:, :] = 0
		da_mask_full[top:bottom, left:right] = da_mask_crop
		coverage = cov2
		mask_debug["thresh"] = f"otsu"
	else:
		# Try stronger fixed threshold
		for thr_try in (0.6, 0.7):
			da_mask_crop = (pcrop >= thr_try).astype(np.uint8)
			cov3 = float(da_mask_crop.sum()) / max(1.0, float(da_mask_crop.size))
			if 0.01 <= cov3 <= 0.98:
				da_mask_full[:, :] = 0
				da_mask_full[top:bottom, left:right] = da_mask_crop
				coverage = cov3
				mask_debug["thresh"] = f"{thr_try}"
				break
		# Last resort: invert crop if still extreme
		if coverage <= 0.01 or coverage >= 0.98:
			inv = 1 - (pcrop >= 0.5).astype(np.uint8)
			cov4 = float(inv.sum()) / max(1.0, float(inv.size))
			if 0.01 <= cov4 <= 0.98:
				da_mask_full[:, :] = 0
				da_mask_full[top:bottom, left:right] = inv
				coverage = cov4
				mask_debug["thresh"] = f"invert@0.5"

# 원본 크기로 리사이즈: 먼저 crop 영역만 리사이즈 후 원본 크기 매핑
da_mask_up = np.zeros((h0, w0), dtype=np.uint8)
if crop.size > 0:
	crop_up = cv2.resize(da_mask_full[top:bottom, left:right], (w0, h0), interpolation=cv2.INTER_NEAREST)
	da_mask_up = crop_up

# ---- 5) 시각화 (반투명 오버레이) ----
overlay = img.copy()
overlay[da_mask_up == 1] = (0, 255, 0)  # 드라이버블 영역을 초록으로
alpha = 0.35
vis = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

cv2.imwrite("da_mask.png", (da_mask_up * 255))
cv2.imwrite("da_overlay.png", vis)
print("Saved: da_mask.png, da_overlay.png; mask_sum:", int(da_mask_up.sum()), "coverage:", round(coverage, 4), "dbg:", mask_debug, "letterbox:", {"r": round(ratio, 4), "pad": (pad_left, pad_top), "unpadded_wh": (uw, uh)})
