import cv2
import matplotlib.pyplot as plt
from insight_face_model import FaceAnalysis

if __name__ == "__main__":
    
    face_ana = FaceAnalysis(allowed_modules= [
        # 'landmark_3d_68',
        'detection',
        # 'genderage',
        # 'recognition'
        ])
    face_ana.prepare(ctx_id=0, det_size=(640, 640))
    import ipdb;ipdb.set_trace()
    f = "./debug.png"
    img = cv2.imread(f)
    res = face_ana.get(img, max_num = 999)
    img = face_ana.draw_on(img, res)
    cv2.imwrite(
        'debug1.png',
        img
    )
