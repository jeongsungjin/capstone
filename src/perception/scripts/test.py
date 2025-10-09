#!/usr/bin/env python3
import os, sys, time
import os.path as osp
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa

from PIL import Image

from utils import LetterBox, DEFAULT_LETTERBOX_SIZE

# chg pkg path
ENGINE = '/home/guest5/2.5d-object-detection/checkpoints/model_v2_half.plan'
IMG_ABS = '/home/guest5/2.5d-object-detection/samples/k729_cam1_1730382931-498000000.jpg'
# 원하는 입력 크기 (동적 엔진일 때 필요). 고정 엔진이면 엔진에서 읽어옵니다.
H, W = 832, 1440

def fail(msg):
    print(f"[ERR] {msg}", file=sys.stderr); sys.exit(1)

if not ENGINE or not IMG_ABS:
    print(f"Usage: {sys.argv[0]} </abs/path/to/engine.plan> </abs/path/to/image.jpg>")
    sys.exit(1)

if not os.path.isabs(IMG_ABS):
    fail("image path must be ABSOLUTE (e.g., /home/user/pic.jpg)")
if not os.path.isfile(ENGINE):
    fail(f"engine not found: {ENGINE}")
if not os.path.isfile(IMG_ABS):
    fail(f"image not found: {IMG_ABS}")

logger = trt.Logger(trt.Logger.WARNING)
with open(ENGINE, "rb") as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
if engine is None: fail("engine deserialize failed")

# assume exactly 1 input, 1+ outputs OK
names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
in_ids  = [i for i,n in enumerate(names) if engine.binding_is_input(i)]
out_ids = [i for i,n in enumerate(names) if not engine.binding_is_input(i)]
if len(in_ids) != 1: fail(f"expected exactly 1 input, got {len(in_ids)}")
if len(out_ids) < 1: fail("no outputs in engine")
inp_idx = in_ids[0]
inp_name = names[inp_idx]

ctx = engine.create_execution_context()
if ctx is None: fail("create_execution_context failed")

# ---- get/set input shape ----
bshape = engine.get_binding_shape(inp_idx)  # may contain -1
if -1 in bshape:
    # dynamic → use (1,3,H,W)
    shape = (1, 3, H, W)
    if not ctx.set_binding_shape(inp_idx, shape):
        fail(f"set_binding_shape failed for {inp_name} -> {shape}")
else:
    shape = tuple(bshape)

# sanity
if not ctx.all_binding_shapes_specified:
    fail("not all binding shapes specified")

# ---- preprocess (absolute path) ----
img = cv2.imread(IMG_ABS)
if img is None: fail(f"failed to read image: {IMG_ABS}")
draw_img = np.copy(img)
original_width, original_height = (img.shape[1], img.shape[0])
img = np.stack([LetterBox(DEFAULT_LETTERBOX_SIZE, auto=False, stride=32)(img)])
img = img[..., ::-1].transpose((0, 3, 1, 2))
img = np.ascontiguousarray(img)
nchw = img.astype(np.float32) / 255.0

# match engine input dtype (fp32/fp16)
in_dtype = trt.nptype(engine.get_binding_dtype(inp_idx))
if nchw.dtype != in_dtype:
    nchw = nchw.astype(in_dtype, copy=False)

# ---- buffers ----
def nbytes(idx):
    dtype = trt.nptype(engine.get_binding_dtype(idx))
    vol = int(np.prod(ctx.get_binding_shape(idx)))
    return vol * np.dtype(dtype).itemsize

bindings = [None] * engine.num_bindings
host = [None] * engine.num_bindings
dev  = [None] * engine.num_bindings
for i in range(engine.num_bindings):
    nb = nbytes(i)
    host[i] = cuda.pagelocked_empty(nb, dtype=np.uint8)
    dev[i]  = cuda.mem_alloc(nb)
    bindings[i] = int(dev[i])

# copy input
host[inp_idx][:] = nchw.ravel().view(np.uint8)
# host to device (cpu -> gpu) 
# cuda.memcpy_htod(dest, src)
cuda.memcpy_htod(dev[inp_idx], host[inp_idx])

# ---- run ----
stream = cuda.Stream()

# warmup
for _ in range(3):
    ok = ctx.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    if not ok: fail("execute_async_v2 failed (warmup)")
stream.synchronize()

# timed
t0 = time.time()
ok = ctx.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
if not ok: fail("execute_async_v2 failed")

# copy outputs
for oid in out_ids:
    cuda.memcpy_dtoh_async(host[oid], dev[oid], stream)

stream.synchronize()
t1 = time.time()
print(f"[OK] inference {1000*(t1-t0):.3f} ms")

def _compute_bounding_boxes(triangles):
    # Extract points
    x1, y1 = triangles[:, 0], triangles[:, 1]  # P1
    x2, y2 = triangles[:, 2], triangles[:, 3]  # P2
    x3, y3 = triangles[:, 4], triangles[:, 5]  # P3

    # Compute mirrored points
    x2_mirror, y2_mirror = 2 * x1 - x2, 2 * y1 - y2  # P2'
    x3_mirror, y3_mirror = 2 * x1 - x3, 2 * y1 - y3  # P3'

    # Combine all points of the parallelogram
    x_coords = np.stack([x1, x2, x3, x2_mirror, x3_mirror], axis=1)
    y_coords = np.stack([y1, y2, y3, y2_mirror, y3_mirror], axis=1)

    # Compute bounding box
    min_x = np.min(x_coords, axis=1)
    max_x = np.max(x_coords, axis=1)
    min_y = np.min(y_coords, axis=1)
    max_y = np.max(y_coords, axis=1)

    return np.stack([min_x, min_y, max_x, max_y], axis=1)

def _nms_bboxes(triangles, confidences, iou_threshold):
    bboxes = _compute_bounding_boxes(triangles)

    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = confidences.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# TODO: refactor this to make more efficient
def _non_max_suppression(
    prediction,
    conf_thres: float = 0.1,
    iou_thres: float = 0.5,
    agnostic: bool = False,
    multi_label: bool = False,
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_nms: int = 30000,
    max_wh: int = 7680,
):
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 6)  # number of classes
    nm = prediction.shape[1] - nc - 6  # number of masks
    mi = 6 + nc  # mask start index
    xc = np.max(prediction[:, 6:mi], axis=1) > conf_thres  # candidates

    # Settings
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    print('201', prediction.shape)
    prediction = prediction.transpose(0, 2, 1)  # shape(1,84,6300) to shape(1,6300,84)
    print(prediction.shape)

    output = [np.zeros((0, 8 + nm))] * bs

    # TODO: get rid of loop, we're operating on a single image only anyway
    for xi, x in enumerate(prediction):  # image index, image inference -> will only loop once for one input image
        x = x[xc[xi]]  # confidence
        
        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = np.split(x, [6, 6 + nc], axis=1)
        
        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((
                box[i], x[i, 6 + j, None], 
                j[:, None].astype(np.float16), 
                mask[i]
            ), 1)

        else:  # best class only
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True).astype(np.float16)
            x = np.concatenate((box, conf, j, mask), 1)[np.squeeze(conf) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if n > 0:  # no boxes
            if n > max_nms:  # excess boxes
                x = x[np.argsort(x[:, 6])[::-1][:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 7:8] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 6]  # scores
            boxes = x[:, :6] + c  # boxes (offset by class)

            i = _nms_bboxes(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            output[xi] = x[i]
            
    return output

def _scale_triangles(triangles, src_shape, target_shape, padding=True):
    if len(target_shape) == 2:
        target_shape = (target_shape[0], target_shape[1], 3)

    gain = min(src_shape[0] / target_shape[0], src_shape[1] / target_shape[1])  # gain  = old / new
    pad = (
        round((src_shape[1] - target_shape[1] * gain) / 2 - 0.1),
        round((src_shape[0] - target_shape[0] * gain) / 2 - 0.1),
    )  # wh padding

    if padding:
        triangles[..., 0] -= pad[0]  # center_x padding
        triangles[..., 1] -= pad[1]  # center_y padding
        triangles[..., 2] -= pad[0]  # v1_x padding
        triangles[..., 3] -= pad[1]  # v1_y padding
        triangles[..., 4] -= pad[0]  # v2_x padding
        triangles[..., 5] -= pad[1]  # v2_y padding

    triangles[..., :6] /= gain
    return triangles

def visualize_detections(detections, image, save_path) -> Image:
    for detection in detections:
        triangle = detection[:6]

        x1, y1, x2, y2, x3, y3 = triangle

        x2_mirror = 2 * x1 - x2
        y2_mirror = 2 * y1 - y2
        x3_mirror = 2 * x1 - x3
        y3_mirror = 2 * y1 - y3

        triangle_points = np.array([
            [x2, y2],
            [x3, y3],
            [x2_mirror, y2_mirror],
            [x3_mirror, y3_mirror],
        ], dtype=np.int32).reshape(-1, 1, 2)

        cv2.polylines(image, [triangle_points], isClosed=True, color=(0, 255, 0), thickness=2)

    if save_path is not None:
        cv2.imwrite(save_path, image)
    
    return Image.fromarray(image)

# ---- show output shapes + save .npy (absolute paths derived)
for oid in out_ids:
    name  = names[oid]
    dtype = trt.nptype(engine.get_binding_dtype(oid))
    oshape= tuple(ctx.get_binding_shape(oid))
    arr = np.frombuffer(host[oid], dtype=np.uint8).view(dtype).reshape(oshape)
    print(arr.dtype, arr.shape)

    arr = _non_max_suppression(arr)[0]
    results = _scale_triangles(arr[:, :6], DEFAULT_LETTERBOX_SIZE, (original_height, original_width, 3))

    out_file_ext = IMG_ABS.split('.')[-1]
    out_file = f'{osp.basename(IMG_ABS).rstrip("." + out_file_ext)}_trt_out.{out_file_ext}'
    visualize_detections(arr, draw_img, save_path=out_file)
