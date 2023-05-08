# PRETRAIN_NAMEright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a PRETRAIN_NAME of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import imgaug.augmenters as iaa
import cv2
import math
# ImageNet code should change this value
IMAGE_SIZE = 32

MEAN = np.array([[[102.9801, 115.9465, 122.7717]]])

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def iaa_dec(pil_img, func):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    return func(images=pil_img).transpose(1,2,0)

def gaussianblur(pil_img, level):
    fun = iaa.GaussianBlur(sigma=(0, level))
    return iaa_dec(pil_img, fun)

def addvalue(pil_img, level):
    fun = iaa.Add((-40, 40), per_channel=0.5)
    return iaa_dec(pil_img, fun)

def gaussian(pil_img, level):
    fun = iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255))
    return iaa_dec(pil_img, fun)

def cutout(pil_img, level):
    fun = iaa.Cutout(nb_iterations=2)
    return iaa_dec(pil_img, fun)

def laplacian(pil_img, level):
    fun = iaa.AdditiveLaplaceNoise(scale=(0, 0.2 * 255))
    return iaa_dec(pil_img, fun)

def multiply_element(pil_img, level):
    fun = iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5)
    return iaa_dec(pil_img, fun)

def multiply_channel(pil_img, level):
    fun = iaa.Multiply((0.5, 1.5), per_channel=0.5)
    return iaa_dec(pil_img, fun)


def dropout2d(pil_img, level):
    fun = iaa.Dropout2d(p=0.5)
    return iaa_dec(pil_img, fun)

def replace_element(pil_img, level):
    fun = iaa.ReplaceElementwise(0.1, [0, 255], per_channel=0.5)
    return iaa_dec(pil_img, fun)

def impulse_noise(pil_img, level):
    fun = iaa.ImpulseNoise(0.1)
    return iaa_dec(pil_img, fun)

def salt(pil_img, level):
    fun = iaa.Salt(0.1)
    return iaa_dec(pil_img, fun)

def coarse_pepper(pil_img, level):
    fun = iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1))
    return iaa_dec(pil_img, fun)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)

def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)

def rankfiler(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return pil_img.filter(ImageFilter.RankFilter(size=3, rank=0))

# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

def boxblur(pil_img, level):
    level = float_parameter(sample_level(level), 8) + 0.1
    return pil_img.filter(ImageFilter.BoxBlur(level))


def gaussianblur(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.GaussianBlur(sigma=(0, level))
    return fun(images=pil_img).transpose(1, 2, 0)


def poisson(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.AdditivePoissonNoise(40, per_channel=True)
    return fun(images=pil_img).transpose(1, 2, 0)


def snowflake(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))
    return fun(images=pil_img).transpose(1, 2, 0)

def rain(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))
    return fun(images=pil_img).transpose(1, 2, 0)

def multi(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.MultiplyElementwise((0.5, 1.5))
    return fun(images=pil_img).transpose(1, 2, 0)

def dropout(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.CoarseDropout(0.02, size_percent=0.5, per_channel=True)
    return fun(images=pil_img).transpose(1, 2, 0)


def blendalpha(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.BlendAlpha((0.0, 1.0), iaa.Grayscale(1.0))
    return fun(images=pil_img).transpose(1, 2, 0)


def canny(pil_img, level):
    pil_img = np.array(pil_img)
    pil_img = pil_img.transpose(2, 0, 1)
    fun = iaa.Canny(alpha=(0.0, 0.5))
    img = fun(images=pil_img).transpose(1, 2, 0)
    #print(img.shape)
    return img


def invert(pil_img, level):
    fun = iaa.Invert(0.25, per_channel=0.5)
    return iaa_dec(pil_img, fun)

def jpeg(pil_img, level):
    fun = iaa.JpegCompression(compression=(70, 99))
    return iaa_dec(pil_img, fun)


def apply_op(image, op, severity):
  image = np.clip(image, 0, 255).astype(np.uint8)
  pil_img = Image.fromarray(image)  # Convert to PIL.Image
  pil_img = op(pil_img, severity)
  return np.asarray(pil_img) #/ 255.


def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  #import pdb
  #pdb.set_trace()
  mean =  MEAN[0][0][::-1]#, np.array(STD)
  image = (image - mean[:, None, None]) #/ std[:, None, None]
  return image.transpose(1, 2, 0)


def denormalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  mean = MEAN[0][0][::-1]
  mean = mean[:, None, None].transpose(1, 2, 0)
  image_new = image + mean
  #image_new = image_new.transpose(2, 0, 1)  # Switch to channel-first
  return image_new



# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble,
                                                      array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
        0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def getOptimalKernelWidth1D(radius, sigma):
    return radius * 2 + 1

def gauss_function(x, mean, sigma):
    return (np.exp(- x**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)

def getMotionBlurKernel(width, sigma):
    k = gauss_function(np.arange(width), 0, sigma)
    Z = np.sum(k)
    return k/Z

def shift(image, dx, dy):
    if(dx < 0):
        shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
        shifted[:,dx:] = shifted[:,dx-1:dx]
    elif(dx > 0):
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:,:dx] = shifted[:,dx:dx+1]
    else:
        shifted = image

    if(dy < 0):
        shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
        shifted[dy:,:] = shifted[dy-1:dy,:]
    elif(dy > 0):
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy,:] = shifted[dy:dy+1,:]
    return shifted

def _motion_blur(x, radius, sigma, angle):
    width = getOptimalKernelWidth1D(radius, sigma)
    kernel = getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):
            # simulated motion exceeded image borders
            break
        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred



def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()



def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def pixelate(x, severity=1):
    severity = min(5, severity)
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x_shape = np.array(x).shape

    x = x.resize((int(x_shape[1] * c), int(x_shape[0] * c)), Image.BOX)

    x = x.resize((x_shape[1], x_shape[0]), Image.NEAREST)

    return x

def noaug(img, severity=1):
    return img


def fog(x, severity=1):
    severity = min(5, severity)
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    shape = np.array(x).shape
    max_side = np.max(shape)
    map_size = next_power_of_2(int(max_side))

    x = np.array(x) / 255.
    max_val = x.max()

    x_shape = np.array(x).shape
    if len(x_shape) < 3 or x_shape[2] < 3:
        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[
                    :shape[0], :shape[1]]
    else:
        x += c[0] * \
             plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0],
             :shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def defocus_blur(x, severity=1):
    severity = min(5, severity)
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    if len(x.shape) < 3 or x.shape[2] < 3:
        channels = np.array(cv2.filter2D(x, -1, kernel))
    else:
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    shape = np.array(x).shape
    severity = min(5, severity)
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = np.array(x)

    angle = np.random.uniform(-45, 45)
    x = _motion_blur(x, radius=c[0], sigma=c[1], angle=angle)

    if len(x.shape) < 3 or x.shape[2] < 3:
        gray = np.clip(np.array(x).transpose((0, 1)), 0, 255)
        if len(shape) >= 3 or shape[2] >= 3:
            return np.stack([gray, gray, gray], axis=2)
        else:
            return gray
    else:
        return np.clip(x, 0, 255)


extensive_aug = [gaussianblur, snowflake, rain, multi, dropout, canny, poisson]
style_augment = [solarize, posterize, brightness]
style_augment_extensive = [solarize, posterize, brightness, pixelate, fog, defocus_blur, motion_blur]
noise_augment = extensive_aug + [autocontrast, equalize, color, sharpness, brightness]
augmentations = style_augment + noise_augment
simple_aug = [autocontrast, equalize, posterize, solarize, color, contrast, brightness, sharpness]
augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]
