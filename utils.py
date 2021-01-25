import cv2

import matplotlib.pyplot as plt


def rotate_image(image, angle, image_center):
    new_center = (image_center[1], image_center[0])
    rot_mat = cv2.getRotationMatrix2D(new_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[::-1])
    return result

def get_new_image(big_image_rotated, transform, center, size, big_size):
    big_image_before = rotate_image(big_image_rotated, transform[0], center)
    patch2_test = big_image_before[max(0, center[0] - (big_size//2) + transform[2]):min(big_image_before.shape[0], center[0] + (big_size//2) + transform[2]), max(0, center[1] - (big_size//2) + transform[1]):min(big_image_before.shape[1], center[1] + (big_size//2) + transform[1])]

    if patch2_test.shape != (big_size, big_size):
        return None

    elif size != big_size:
        patch2_test = cv2.resize(patch2_test, (size, size), interpolation=cv2.INTER_CUBIC)

    return patch2_test


def visualize_registration(reference_image, floating_images, step):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax[0].imshow(reference_image, cmap=plt.cm.gray)
    ax[1].imshow(floating_images[step], cmap=plt.cm.gray)
    plt.show()
