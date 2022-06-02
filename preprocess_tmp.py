import sys
from IPython import embed

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
    raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
                            before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp

import face_detection

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
                                    device='cuda:{}'.format(id)) for id in range(args.ngpu)]

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(vfile, args, gpu_id):
    print("in process video file")
    video_stream = cv2.VideoCapture(vfile)
    
    vidname = os.path.basename(vfile).split('.')[0]
    fulldir = path.join(args.preprocessed_root, vidname)
    os.makedirs(fulldir, exist_ok=True)

    frames = []
    i = -1
    os.makedirs(f"{fulldir}/wav2lip_faces", exist_ok=True)
    print("process video file=", vfile)
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        
        """
        # HACK(demi): first crop
        cx=310
        cy=750
        s=450
        frame = frame[cx:cx+s, cy:cy+s]
        """

        i += 1
        print("i=", i)
        #print("frame shape=", frame.shape)

        # HACK(demi): resize first
        width = int(frame.shape[1] * 0.25)
        height = int(frame.shape[0] * 0.25)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        frames.append(frame)



    print("finish reading images:", len(frames))
    
    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
    print("batch size=", args.batch_size, " # of batches=", len(batches))
    i = -1
    for fb in batches:
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            """
            if f is None:
                continue
            """
            if f is None:
                print("i=", i, "fa, preds, fb, f, j")
                embed()

            x1, y1, x2, y2 = f
            cv2.imwrite(path.join(f"{fulldir}/wav2lip_faces", f'{i:06d}.jpg'), fb[j][y1:y2, x1:x2])

def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    fulldir = path.join(args.preprocessed_root, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio-raw.wav')

    command = template.format(vfile, wavpath)
    subprocess.call(command, shell=True)

    
def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        process_video_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()
        
def main(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))


    if os.path.exists(f"{args.data_root}.mp4"):
        filelist = [args.data_root + ".mp4"]
    else:
        filelist = glob(path.join(args.data_root, '*.mp4'))

    jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    print('Dumping audios...')

if __name__ == '__main__':
    main(args)
