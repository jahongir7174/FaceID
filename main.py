import warnings
from argparse import ArgumentParser

import cv2

from nets.nn import FaceDetection
from nets.nn import FaceRecognition
from utils.util import norm_crop_image

warnings.filterwarnings("ignore")

detection = FaceDetection('./weights/detection.onnx')
recognition = FaceRecognition('./weights/recognition.onnx')


def main():
    parser = ArgumentParser()
    parser.add_argument('filepath', nargs='+', help='image file paths')
    parser.add_argument('--threshold', default=0.35, help='image file paths')

    args = parser.parse_args()

    image1 = cv2.imread(args.filepath[0])
    image2 = cv2.imread(args.filepath[1])

    _, kpt = detection(image1, score_thresh=0.5, input_size=(640, 640))
    kpt = kpt[:1][0]
    face1 = norm_crop_image(image1, kpt)

    _, kpt = detection(image2, score_thresh=0.5, input_size=(640, 640))
    kpt = kpt[:1][0]
    face2 = norm_crop_image(image2, kpt)

    vector1 = recognition(face1)[0].flatten()
    vector2 = recognition(face2)[0].flatten()

    score = vector1 @ vector2
    if score < args.threshold:
        result = 'They are NOT the same person'
    else:
        result = 'They ARE the same person'
    print(result)
    print(f'Similarity: {score:.5f}')


if __name__ == '__main__':
    main()
