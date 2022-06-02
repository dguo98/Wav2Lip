import os
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


root_dir = "/u/nlp/data/timit_videos_v2/transfer/p4"
print("root_dir=", root_dir)

os.makedirs(f"{root_dir}/landmarks")
os.makedirs(f"{root_dir}/frames")

for i in range(33):
    mp4_path = f'{root_dir}/mp4s/s{i}.mp4'
    if not os.path.exists(mp4_path):
        continue
    
    print("extract landmarks for", mp4_path)
    # TODO(demi): refactor
    os.system(f"ffmpeg -i {mp4_path} -r 25 framesk

    lm_path = f'{root_dir}/landmarks/s{i}.pkl'
    if os.path.exists(lm_path):
        continue
    image_paths = sorted(glob(f"{root_dir}/frames/s{i}/*.jpg"))
    jacinle.io.dump(lm_path, image_paths)
    landmarks = jacinle.concurrency.pool.TQDMPool(32).map(get_landmark, image_paths)
    jacinle.io.dump(lm_path, landmarks)

