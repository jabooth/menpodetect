from __future__ import division
import dlib
from menpo.shape import BoundingBox
from menpo.transform import UniformScale
import numpy as np

dlib_frontal_face_detector = None

uint_image = lambda i: (i.pixels[..., 0] * 255.0).astype(np.uint8)


def boundingbox_from_rect(rect):
    return BoundingBox(np.array(((rect.top(), rect.left()),
                                 (rect.bottom(), rect.right()))))


def dlib_detect_frontal_faces(image, add_as_landmarks=True, width=300):
    global dlib_frontal_face_detector
    if dlib_frontal_face_detector is None:
        dlib_frontal_face_detector = dlib.get_frontal_face_detector()
    dlib_image = image.as_greyscale(mode='average')
    did_rescale = False
    if width is not None and dlib_image.width > width:
        did_rescale = True
        scale_factor = width / dlib_image.width
        dlib_image = dlib_image.rescale(scale_factor)
    dlib_image_pixels = uint_image(dlib_image)
    faces = dlib_frontal_face_detector(dlib_image_pixels)
    bbs = [boundingbox_from_rect(f) for f in faces]
    if did_rescale:
        bbs = [UniformScale(1/scale_factor, n_dims=2).apply(pc) for pc in bbs]
    if add_as_landmarks:
        for i, pc in enumerate(bbs):
            image.landmarks['frontal_face_{:02d}'.format(i)] = pc
    return bbs
