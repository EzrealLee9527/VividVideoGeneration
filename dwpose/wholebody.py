import os
import numpy as np
import onnxruntime as ort
from .det import inference_detector
from .pose import inference_pose


class Wholebody:
    def __init__(self):
        device = "cpu"
        providers = (
            ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        )
        onnx_det = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "yolox_l.onnx"
        )
        onnx_pose = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "dw-ll_ucoco_384.onnx"
        )

        sess_opt = ort.SessionOptions()
        sess_opt.inter_op_num_threads = 1
        sess_opt.intra_op_num_threads = 1
        sess_opt.use_deterministic_compute = True

        self.session_det = ort.InferenceSession(
            path_or_bytes=onnx_det,
            sess_options=sess_opt,
            providers=providers,
        )
        self.session_pose = ort.InferenceSession(
            path_or_bytes=onnx_pose,
            sess_options=sess_opt,
            providers=providers,
        )

    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        return keypoints, scores
