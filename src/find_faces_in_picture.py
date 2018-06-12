from scipy import misc
import tensorflow as tf
import numpy as np
import os
import facenet
import align.detect_face
import cv2


# Load the jpg file into a numpy array
image = "/home/florencio/dev/face_recog/known_pics/gaidai.jpg"


minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
gpu_memory_fraction = 1.0
margin = 44

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

img = misc.imread(os.path.expanduser(image), mode='RGB')
img_size = np.asarray(img.shape)[0:2]
bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
if len(bounding_boxes) < 1:
    image_paths.remove(image)
    print("can't detect face, remove ", image)
    exit

print(bounding_boxes)
det = np.squeeze(bounding_boxes[0,0:4])
bb = np.zeros(4, dtype=np.int32)
bb[0] = np.maximum(det[0]-margin/2, 0)
bb[1] = np.maximum(det[1]-margin/2, 0)
bb[2] = np.minimum(det[2]+margin/2, img_size[1])
bb[3] = np.minimum(det[3]+margin/2, img_size[0])

top = bb[1]
bottom = bb[3]
left = bb[0]
right = bb[2]

print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

face_image = cv2.imread(image)
crop_img = face_image[179:362, 669:816]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()