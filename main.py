import dlib # face detection & face recognition
import cv2 # handle image
import numpy as np # matrix
import matplotlib.pyplot as plt # matplotlib = visualization of the results
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector() # 얼굴 탐지
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

# 1. find face
# 2. detect face landmark to find eyes, nose, mouth, etc
# 3. encode face

def face_find(img):
    dets = detector(img, 1)

# if it didn't find any faces, return empty array
    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    dots, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype = np.int)
    for k, v in enumerate(dets): # looped by number of detected face
        dot = ((v.left(), v.top()), (v.right(), v.bottom()))
        dots.append(dot)

        shape = sp(img,v) # get landmarks (image, square)

        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)

    return dots, shapes, shapes_np

# get array vectors from dots
def face_encode(img, shapes):
    descriptors = []
    for shape in shapes:
        descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(np.array(descriptor))

    return np.array(descriptors)