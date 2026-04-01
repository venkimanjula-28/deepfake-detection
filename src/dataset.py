import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from facenet_pytorch import MTCNN
    # Initialize MTCNN with post_process=False to get raw output [0, 1]
    # We will apply ImageNet normalization separately for consistency
    mtcnn = MTCNN(image_size=224, margin=20, select_largest=True, post_process=False)
except Exception as e:
    print(f"MTCNN not available: {e}")
    mtcnn = None


# -----------------------------------------------------------------------
# Fast cached dataset  (loads pre-saved .npy files)
# -----------------------------------------------------------------------
class CachedDeepFakeDataset(Dataset):
    """
    Loads pre-cached .npy files produced by cache_dataset.py.
    Each .npy file has shape (num_frames, H, W, 3) uint8.
    Much faster than reading raw images/videos every iteration.
    """
    def __init__(self, root_dir, num_frames=6, transform=None):
        self.num_frames = num_frames
        self.transform  = transform
        self.samples    = []   # list of (npy_path, label)

        for label, cls in enumerate(['real', 'fake']):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith('.npy'):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            arr = np.load(path)  # (T, H, W, 3) or (H, W, 3) uint8

            # Handle single-image .npy (old augmentation files may have shape H,W,3)
            if arr.ndim == 3:  # (H, W, 3)
                arr = np.stack([arr] * self.num_frames, axis=0)

            # Pad / truncate to num_frames
            if len(arr) < self.num_frames:
                pad = np.stack([arr[-1]] * (self.num_frames - len(arr)))
                arr = np.concatenate([arr, pad], axis=0)
            arr = arr[:self.num_frames]

            # Ensure contiguous, then convert: uint8 → float32 [0, 1], shape (T, 3, H, W)
            arr = np.ascontiguousarray(arr)
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0

            if self.transform is not None:
                # Apply transform frame-by-frame
                frames = [self.transform(tensor[i]) for i in range(tensor.shape[0])]
                tensor = torch.stack(frames)

            return tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            dummy = torch.zeros(self.num_frames, 3, 112, 112)
            return dummy, torch.tensor(label, dtype=torch.long)


def extract_frames_from_video(video_path, num_frames=12):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f'Cannot open video: {video_path}')

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = num_frames

        indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f'No frames extracted from {video_path}')

        return frames
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        # Return dummy frames if video fails
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        return [dummy_frame] * num_frames


def detect_faces(frames, target_size=224):
    processed = []
    for image in frames:
        try:
            if mtcnn is not None:
                # MTCNN expects numpy array in RGB format
                face = mtcnn(image)
                if face is not None:
                    # MTCNN with post_process=False returns tensor in range [0, 1]
                    processed.append(face)
                    continue
            # Fallback to Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            hits = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            if len(hits) > 0:
                x, y, w, h = hits[0]
                crop = image[y:y + h, x:x + w]
                crop = cv2.resize(crop, (target_size, target_size))
                # Convert to tensor in range [0, 1]
                processed.append(torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0)
            else:
                # No face detected, use full image
                resized = cv2.resize(image, (target_size, target_size))
                processed.append(torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0)
        except Exception as e:
            print(f"Face detection failed for frame: {e}")
            resized = cv2.resize(image, (target_size, target_size))
            processed.append(torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0)

    return processed


def pad_or_truncate(frames, num_frames):
    if len(frames) >= num_frames:
        return frames[:num_frames]
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    return frames


class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, num_frames=12, transform=None, mode='video'):
        self.root_dir = root_dir
        self.samples = []
        self.num_frames = num_frames
        self.transform = transform
        self.mode = mode

        for label, cls in enumerate(['real', 'fake']):
            class_dir = os.path.join(root_dir, cls)
            if not os.path.exists(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, fname)
                if os.path.isfile(path) and fname.lower().endswith(('.jpg', '.png', '.jpeg', '.mp4', '.avi')):
                    if fname.lower().endswith(('.mp4', '.avi')):
                        self.samples.append((path, label, 'video'))
                    else:
                        self.samples.append((path, label, 'image'))
                elif os.path.isdir(path):
                    self.samples.append((path, label, 'frames'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            path, label, kind = self.samples[idx]

            if kind == 'video':
                frames = extract_frames_from_video(path, self.num_frames)
                frames_tensor = detect_faces(frames, target_size=224)
            elif kind == 'image':
                # Single image - repeat it to fill num_frames
                image = cv2.imread(path)
                if image is None:
                    raise ValueError(f"Could not read image: {path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames = [image] * self.num_frames
                frames_tensor = detect_faces(frames, target_size=224)
            else:  # 'frames' - directory with images
                image_names = sorted([p for p in os.listdir(path) if p.lower().endswith(('.jpg', '.png', '.jpeg'))])
                images = [cv2.imread(os.path.join(path, i)) for i in image_names]
                images = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in images if i is not None]
                frames_tensor = detect_faces(images[:self.num_frames], target_size=224)

            frames_tensor = pad_or_truncate(frames_tensor, self.num_frames)
            frames_tensor = torch.stack(frames_tensor, dim=0)

            if self.transform is not None:
                b, c, h, w = frames_tensor.shape
                frames_tensor = frames_tensor.view(b, c, h, w)
                frames_tensor = self.transform(frames_tensor)
                frames_tensor = frames_tensor.view(b, c, h, w)

            return frames_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing sample {idx} ({path}): {e}")
            # Return a dummy sample to avoid crashing
            dummy_frames = torch.zeros(self.num_frames, 3, 224, 224)
            return dummy_frames, torch.tensor(label, dtype=torch.long)