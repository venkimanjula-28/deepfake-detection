"""
Pre-process all images/videos into cached .npy files for fast training.
Run once: python src/cache_dataset.py --data-dir ./Dataset/Train --frames 6
"""
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(image_size=112, margin=20, select_largest=True, post_process=False)
    print("MTCNN loaded for face detection.")
except Exception:
    mtcnn = None
    print("MTCNN not available - using Haar cascade fallback.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def extract_and_resize(path, num_frames, size=112, use_face=False):
    """Extract num_frames evenly spaced from video/image, resize to (size, size)."""
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.mp4', '.avi', '.mov'):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or num_frames
        indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
        raw = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw = [img] * num_frames

    if not raw:
        return None

    frames_out = []
    for img in raw:
        face = None
        if use_face:
            # Try MTCNN first
            if mtcnn is not None:
                try:
                    t = mtcnn(img)
                    if t is not None:
                        arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                        face = arr
                except Exception:
                    pass
            # Fallback: Haar cascade
            if face is None:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                hits = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                if len(hits) > 0:
                    x, y, w, h = hits[0]
                    face = cv2.resize(img[y:y+h, x:x+w], (size, size))
        # Fast path: just resize the full frame (no face detection)
        if face is None:
            face = cv2.resize(img, (size, size))

        frames_out.append(face)

    # Pad/truncate
    while len(frames_out) < num_frames:
        frames_out.append(frames_out[-1])
    frames_out = frames_out[:num_frames]

    # Shape: (num_frames, H, W, 3) uint8
    return np.stack(frames_out, axis=0)


def cache_split(data_dir, out_dir, num_frames, size):
    os.makedirs(out_dir, exist_ok=True)

    for label_idx, cls in enumerate(['real', 'fake']):
        cls_in  = os.path.join(data_dir, cls)
        cls_out = os.path.join(out_dir, cls)
        os.makedirs(cls_out, exist_ok=True)

        if not os.path.exists(cls_in):
            print(f"  [SKIP] {cls_in} not found")
            continue

        files = [f for f in sorted(os.listdir(cls_in))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi'))]

        print(f"  {cls}: {len(files)} files")
        skip, done = 0, 0

        for fname in tqdm(files, desc=f'  {cls}'):
            out_path = os.path.join(cls_out, os.path.splitext(fname)[0] + '.npy')
            if os.path.exists(out_path):
                done += 1
                continue
            arr = extract_and_resize(os.path.join(cls_in, fname), num_frames, size,
                                        use_face=args.use_face)
            if arr is None:
                skip += 1
                continue
            np.save(out_path, arr)
            done += 1

        print(f"    done={done} skipped={skip}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='e.g. ./Dataset/Train')
    parser.add_argument('--out-dir',  default=None,  help='cache output dir (default: data-dir + _cache)')
    parser.add_argument('--frames',   type=int, default=6)
    parser.add_argument('--size',     type=int, default=112, help='Image size (112 is 4x faster than 224)')
    parser.add_argument('--use-face', action='store_true', default=False,
                        help='Enable face detection (slower but more precise)')
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.data_dir.rstrip('/\\') + '_cache'

    print(f"Caching {args.data_dir}  →  {args.out_dir}")
    print(f"Frames={args.frames}  Size={args.size}x{args.size}")
    cache_split(args.data_dir, args.out_dir, args.frames, args.size)
    print("\nDone! Now train with:")
    print(f"  python src/train.py --data-dir {args.out_dir} --frames {args.frames} --img-size {args.size}")
