import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import defaultdict


def shift_image(img, x, y):
    """
    Pad into image to center of character is center of image
    :param img: cv2 binary image
    :param x: x_center_of_image - x_center_of_character
    :param y: y_center_of_image - y_center_of_character
    :return: Padded image that center of character lies on center of image
    """
    x_abs = int(round(abs(x), 0))
    y_abs = int(round(abs(y), 0))
    if x < 0 and y < 0:
        pad_img = cv2.copyMakeBorder(img, top=y_abs, bottom=0, left=x_abs, right=0, borderType=cv2.BORDER_CONSTANT,
                                     value=0)
    elif x < 0 and y > 0:
        pad_img = cv2.copyMakeBorder(img, top=y_abs, bottom=0, left=0, right=x_abs, borderType=cv2.BORDER_CONSTANT,
                                     value=0)
    elif x > 0 and y < 0:
        pad_img = cv2.copyMakeBorder(img, top=0, bottom=y_abs, left=x_abs, right=0, borderType=cv2.BORDER_CONSTANT,
                                     value=0)
    else:
        pad_img = cv2.copyMakeBorder(img, top=0, bottom=y_abs, left=0, right=x_abs, borderType=cv2.BORDER_CONSTANT,
                                     value=0)
    return pad_img


def padding_and_hold_ratio(cv_img):
    h, w = cv_img.shape[:2]
    if h < w:
        diff = w - h
        top_pad = diff // 2
        bot_pad = diff - top_pad
        padded_img = cv2.copyMakeBorder(cv_img, top_pad, bot_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        diff = h - w
        left_pad = diff // 2
        right_pad = diff - left_pad
        padded_img = cv2.copyMakeBorder(cv_img, 0, 0, left_pad, right_pad, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
    return padded_img


def list_contours(contours):
    list = []
    max_len = 0
    for cnt in contours:
        for p in cnt:
            list.append(p)
    return list


def filter_much_small_contour(bin_img, contours, hierarchy=None, min_area=70):
    """
    Remove hole on character, hole outside character, lines that have the zero area
    """
    mask = np.ones(bin_img.shape[:2], dtype="uint8")
    inv_mask = np.ones(bin_img.shape[:2], dtype="uint8")
    total_mask = np.ones(bin_img.shape[:2], dtype="uint8")
    new_contours = []
    for index, contour in enumerate(contours):
        if int(cv2.contourArea(contour)) == 0:
            # lines are created some points(area = 0)
            for point in contour:
                bin_img[point[0][1], point[0][0]] = 0
        elif 0 < cv2.contourArea(contour) < min_area:
            # hole in character
            if hierarchy[0][index][3] != -1:
                cv2.drawContours(inv_mask, [contour], -1, 0, -1)
            # hole out in character
            else:
                cv2.drawContours(mask, [contour], -1, 0, -1)
        else:
            new_contours.append(contour)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=cv2.resize(mask, bin_img.shape))
    bin_img = cv2.bitwise_or(bin_img, 1 - inv_mask, mask=cv2.resize(total_mask, bin_img.shape))
    return bin_img, list_contours(new_contours)


def crop_char(thresh, contours):
    """
    Crop image based on contours, find the most top, bot, left, right points
    :param thresh: binary image
    :param contours: list of contours
    :return: crop of original image that contains character
    """
    # list1 = filter_much_small_contour(contours, hierarchy)
    xmax = max(i[0][0] for i in contours)
    xmin = min([i[0][0] for i in contours])
    ymax = max(i[0][1] for i in contours)
    ymin = min([i[0][1] for i in contours])
    image = thresh[max(0, ymin - 2):ymax, max(0, xmin - 2):xmax]
    return image


def find_center(list2, mask):
    # scipy.ndimage.measurements.center_of_mass¶
    """
    Find center of the character(mask)
    :param list2: list of contours
    :param mask: mask indicates the position of character
    :return: center of character
    """
    # cnts = cv2.drawContours(mask, list2, -1, (0, 255, 0), 1)

    kpCnt = len(list2)

    x = 0
    y = 0

    for kp in list2:
        x = x + kp[0][0]
        y = y + kp[0][1]

    # cv2.circle(mask, (np.uint8(np.ceil(x/kpCnt)), np.uint8(np.ceil(y/kpCnt))), 1, (255, 255, 255), 1)
    return x / kpCnt, y / kpCnt  # x_center, y_center


def normalize_mask(ori_img):
    """
    Get character image and shift center of character to center of image(a part of image can be disappeared)
    :param thresh: binary image
    :return: image that contain image, and center of image is center of character
    """

    contours, hierarchy = cv2.findContours(ori_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    thresh, contours = filter_much_small_contour(ori_img, contours, hierarchy)
    thresh= crop_char(thresh, contours)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    x_center, y_center = find_center(list_contours(contours), thresh)

    # This part different from original normalize
    # This image just contains contour
    # Initialize crop_image for original image
    blank = np.zeros(thresh.shape)
    cv2.drawContours(blank, contours, -1, (255, 0, 255), 1)
    result_img = centerlize_contour_image(blank, x_center, y_center, ori_img.shape)

    return result_img, contours, thresh.shape, x_center, y_center

def centerlize_contour_image(image,x_center, y_center, ori_img_shape):
    shifted_img = shift_image(image, image.shape[1] / 2 - x_center, image.shape[0] / 2 - y_center)
    padding_and_hold_ratio_img = padding_and_hold_ratio(shifted_img)
    padding_to_original_size_image = padding_to_original_size(padding_and_hold_ratio_img, ori_img_shape)
    return padding_to_original_size_image


def padding_to_original_size(img, original_size):
    """
      Adding padding to image to turn it to original size
      :param img: target image
      :param original_size: tuple of size image want to transfer
    """
    print(img.shape)
    height_img, width_img = img.shape[0], img.shape[1]
    height_org, width_org = original_size
    delta_height = abs(height_img - height_org)
    delta_width = abs(width_img - width_org)
    pad_img = cv2.copyMakeBorder(img, top=int(delta_height / 2), bottom=delta_height - int(delta_height / 2), \
                                 left=int(delta_width / 2), right=delta_width - int(delta_width / 2), \
                                 borderType=cv2.BORDER_CONSTANT,
                                 value=0)
    return pad_img

def preprocess_img(cv_img, threshold=110):
    """
    Combine the above functions to crop character area
    cv_img: cv bgr img
    threshold: threshold to convert bgr to binary image
    """
    ret, bin_print_img = cv2.threshold(cv_img, threshold, 1, cv2.THRESH_BINARY_INV)
    normalized_print_img, contours, crop_shape,x_center, y_center = normalize_mask(bin_print_img)

    return normalized_print_img, contours, crop_shape, x_center, y_center

