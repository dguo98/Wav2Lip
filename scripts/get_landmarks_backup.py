import os
import jacinle.io
import dlib

predictor = dlib.shape_predictor('pretrained_models/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()


def get_landmark(filepath):
    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    shape = predictor(img, dets[0])
    t = list(shape.parts())
    lm = [[tt.x, tt.y] for tt in t]
    return lm


for i in range(32):
    mp4_path = f'./mp4s/s{i}.mp4'
    if not os.path.exists(mp4_path):
        continue
    lm_path = f'./landmarks/s{i}.pkl'
    if os.path.exists(lm_path):
        continue
    image_paths = jacinle.io.lsdir(f'./frames/s{i}/*.png')
    jacinle.io.dump(lm_path, image_paths)
    landmarks = jacinle.concurrency.pool.TQDMPool(32).map(get_landmark, image_paths)
    jacinle.io.dump(lm_path, landmarks)

