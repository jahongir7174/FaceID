import cv2
import numpy

from skimage.transform import SimilarityTransform

reference = numpy.array([[[38.2946, 51.6963],
                          [73.5318, 51.5014],
                          [56.0252, 71.7366],
                          [41.5493, 92.3655],
                          [70.7299, 92.2041]]], dtype=numpy.float32)


def estimate_norm(kpt, image_size=112):
    """
    Args:
        kpt: prediction keypoint
        image_size: crop image size
    Returns: cropped image
    """
    assert kpt.shape == (5, 2)
    min_index = []
    min_error = float('inf')
    transform = SimilarityTransform()

    if image_size == 112:
        src = reference
    else:
        src = float(image_size) / 112 * reference

    min_matrix = []
    kpt_transform = numpy.insert(kpt, 2, values=numpy.ones(5), axis=1)
    for i in numpy.arange(src.shape[0]):
        transform.estimate(kpt, src[i])
        matrix = transform.params[0:2, :]
        results = numpy.dot(matrix, kpt_transform.T)
        results = results.T
        error = numpy.sum(numpy.sqrt(numpy.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_index = i
            min_error = error
            min_matrix = matrix
    return min_matrix, min_index


def norm_crop_image(image, landmark, image_size=112):
    matrix, pose_index = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(image, matrix, (image_size, image_size), borderValue=0.0)
    return warped
