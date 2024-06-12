import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
from PIL import Image, ImageFilter

def hflip(img, _):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def Rotate(img, v):
    return img.rotate(v)

def Horizontal_Shift(img, v):
    # v > 0, shift right
    # v < 0, shift left
    assert -1 < v and v < 1
    width, height = img.size
    
    img = img.crop((0 - v * width, 0, width - v * width, height))

    return img

def Vertical_Shift(img, v):
    assert -1 < v and v < 1
    # v > 0, shift up
    # v < 0, shift down
    width, height = img.size
    

    img = img.crop((0, v * height, width, height + v * height))

    return img

def Zoom(img, v):
    assert -1 < v and v < 1
    width, height = img.size

    img = img.crop((v * width, v * height, width - v * width, height - v * height))
    img = img.resize((width, height), Image.BILINEAR)

    return img

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Sharpness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Identity(img, _):
    return img


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Solarize(img, v):
    return PIL.ImageOps.solarize(img, v)


def Edge_enhance(img, _):
    return img.filter(ImageFilter.EDGE_ENHANCE)


def Edge_enhance_more(img, _):
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


def Smooth(img, _):
    return img.filter(ImageFilter.SMOOTH)


def Smooth_more(img, _):
    return img.filter(ImageFilter.SMOOTH_MORE)


def Detail(img, _):
    return img.filter(ImageFilter.DETAIL)


def Blur(img, _):
    return img.filter(ImageFilter.BLUR)


def Sharpen(img, _):
    return img.filter(ImageFilter.SHARPEN)


def Emboss(img, _):
    return img.filter(ImageFilter.EMBOSS)


def Contour(img, _):
    return img.filter(ImageFilter.CONTOUR)


def Find_edges(img, _):
    return img.filter(ImageFilter.FIND_EDGES)


def SolarizeAdd(img, v, threshold=128):
    v = int(v)
    if random.random() > 0.5:
        v = -v
    img_np = np.array(img).astype(np.int_)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def gauss_noise(image, v):
    # row, col, ch = image.shape
    row, col, ch = 512, 512, 1
    mean, var = 0, v
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy