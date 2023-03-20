import cv2
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as transforms

def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def scale_aligned_long(img, long_size=640):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img


def process_image(image_path):
    filename = image_path.split('/')[-1][:-4]
    img = get_img(image_path)
    # if img.shape[0] > img.shape[1]:
    #     img = cv2.transpose(img)
    #     img = cv2.flip(img, 0)
    img_meta = dict(
        org_img_size=np.array(img.shape[:2])
    )

    img = scale_aligned_short(img)
    img_meta.update(dict(
        img_size=np.array(img.shape[:2]),
        filename=filename
    ))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    data = dict(
        imgs=img,
        img_metas=img_meta
    )

    return data
