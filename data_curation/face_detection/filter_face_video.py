import cv2
import matplotlib.pyplot as plt
from insight_face_model import FaceAnalysis

if __name__ == "__main__":
    f = "/data/users/weisiyuan/dataset/multi_modal_test_with_seg/objects/person_1.png"
    img = cv2.imread(f)
    face_ana = FaceAnalysis(allowed_modules= [
        # 'landmark_3d_68',
        'detection',
        # 'genderage',
        # 'recognition'
        ])
    face_ana.prepare(ctx_id=0, det_size=(640, 640))
    res = face_ana.get(img, max_num = 999)
    img = face_ana.draw_on(img, res)
    cv2.imwrite(
        'debug.png',
        img
    )
