import sys
import cv2
import rospy
import numpy as np
import tensorrt as trt
from os.path import join as make_path

import pycuda.driver as cuda
import pycuda.autoinit  # noqa

from utils import LetterBox, DEFAULT_LETTERBOX_SIZE

class TAFtrt:
    def __init__(self, pkg_path, engine_file_name='model_v2_half'):
        engine_path = make_path(pkg_path, 'engine', f'{engine_file_name}.plan')

        rospy.loginfo('initialize start..')
        
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())

        # this is fixed number
        self.input_idx = 0
        self.output_idx = 1

        self.host = [None] * self.engine.num_bindings
        self.dev  = [None] * self.engine.num_bindings
        self.bindings = [None] * self.engine.num_bindings
        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            vol = int(np.prod(self.ctx.get_binding_shape(i)))
            nb = vol * np.dtype(dtype).itemsize

            self.host[i] = cuda.pagelocked_empty(nb, dtype=np.uint8)
            self.dev[i]  = cuda.mem_alloc(nb)
            self.bindings[i] = int(self.dev[i])

        self.stream = cuda.Stream()

        # warmup
        for _ in range(3):
            ok = self.ctx.execute_async_v2(bindings=self.bindings, stream_handle=stream.handle)
            if not ok: self.fail("execute_async_v2 failed (warmup)")
        self.stream.synchronize()

        self.ctx = self.engine.create_execution_context()

    def processing(self, img):
        self.preprocess(img)
        self.inference()
        return self.visualization()

    def preprocess(self, img):
        self.ori_w, self.ori_h = img.shape[1], img.shape[0]
        letter = np.stack([LetterBox(DEFAULT_LETTERBOX_SIZE, auto=False, stride=32)(img)])
        letter = letter[..., ::-1].transpose((0, 3, 1, 2))
        letter = np.ascontiguousarray(letter)
        nchw = letter.astype(np.float32) / 255.0

        in_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
        if nchw.dtype != in_dtype:
            nchw = nchw.astype(in_dtype, copy=False)
        
        self.host[self.input_idx][:] = nchw.ravel().view(np.uint8)
        cuda.memcpy_htod(self.dev[self.input_idx], self.host[self.input_idx])
        
    def nbytes(self, idx):
        dtype = trt.nptype(self.engine.get_binding_dtype(idx))
        vol = int(np.prod(self.ctx.get_binding_shape(idx)))
        return vol * np.dtype(dtype).itemsize

    def fail(self, msg):
        print(f"[ERR] {msg}", file=sys.stderr)
        sys.exit(1)

    def inference(self):
        ok = self.ctx.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok: self.fail("execute_async_v2 failed")

        cuda.memcpy_dtoh_async(self.host[self.output_index], self.dev[self.output_index], self.stream)
        self.stream.synchronize()

    def visualization(self, img):
        dtype = trt.nptype(self.engine.get_binding_dtype(self.output_index))
        oshape= tuple(self.ctx.get_binding_shape(self.output_index))
        results = np.frombuffer(self.host[self.output_index], dtype=np.uint8).view(dtype).reshape(oshape)

        results = self._non_max_suppression(results)[0]
        results = self._scale_triangles(results[:, :6], DEFAULT_LETTERBOX_SIZE, (self.ori_h, self.ori_w, 3))
        return self._visualization(results, img)

    def _visualization(self, detections, image):
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

        return image

    def _compute_bounding_boxes(self, triangles):
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

    def _nms_bboxes(self, triangles, confidences, iou_threshold):
        bboxes = self._compute_bounding_boxes(triangles)

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
        self, 
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

        prediction = prediction.transpose(0, 2, 1)  # shape(1,84,6300) to shape(1,6300,84)
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

                i = self._nms_bboxes(boxes, scores, iou_thres)  # NMS
                i = i[:max_det]  # limit detections
                output[xi] = x[i]
                
        return output

    def _scale_triangles(self, triangles, src_shape, target_shape, padding=True):
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
