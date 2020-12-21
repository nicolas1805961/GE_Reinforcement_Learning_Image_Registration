import cv2


def rotate_image(image, angle, image_center):
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[::-1])
    return result


def get_new_image(big_image_rotated, transform, center):
    big_image_before = rotate_image(big_image_rotated, transform[0], center)
    patch2_test = big_image_before[center[0] - 75 + transform[2]:center[0] + 75 + transform[2], center[1] - 75 + transform[1]:center[1] + 75 + transform[1]]
    return patch2_test
