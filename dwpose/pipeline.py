import onnxruntime as ort
from .det import inference_detector
from .pose import inference_pose

from torch import Tensor


class DWPose:
    def __init__(self, det_model_path: str, pose_model_path: str) -> None:
        self.det_sess = ort.InferenceSession(det_model_path)
        self.pose_sess = ort.InferenceSession(pose_model_path)

    def __call__(self, image: Tensor):
        image = image.numpy()
        image = image.transpose(1, 2, 0)
        bboxes = inference_detector(self.det_sess, image)
        kps, scores = inference_pose(self.pose_sess, bboxes, image)
        return kps
