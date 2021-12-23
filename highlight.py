import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from testing_contours import list_contours
import cv2

#This algorithm is basically from the via.html
#I just use it, do not know how it work

def is_left(x0,y0,x1,y1,x2,y2):
    return (((x1 - x0) * (y2 - y0)) - ((x2 - x0) * (y1 - y0)))

def is_inside_polygon(all_points_x, all_points_y, px, py):
    if len(all_points_x) == 0 or len(all_points_y) == 0:
        return 0
    wn = 0
    n = len(all_points_x)
    for i in range(n-1):
        is_left_value = is_left( all_points_x[i], all_points_y[i],
                                 all_points_x[i+1], all_points_y[i+1],
                                 px, py)
        if all_points_y[i] <= py:
            if all_points_y[i + 1] > py and is_left_value > 0:
                wn += 1
        else:
            if all_points_y[i+1]  <= py and is_left_value < 0:
                wn -= 1

    is_left_value  = is_left(all_points_x[n-1], all_points_y[n-1],
                               all_points_x[0], all_points_y[0],
                               px, py)
    if all_points_y[n - 1] <= py:
        if all_points_y[0] > py and is_left_value > 0:
            wn += 1
    else:
        if all_points_y[0] <= py and is_left_value < 0:
            wn -= 1

    if wn == 0:
        return 0
    else:
        return 1


def extract_coordinate_from_contours(contours):
    """
      This function conentrates on extract coordinate from contours
      and put them in two globale x and y coordinate
      *params:
      ---contours: the contour line
    """

    x_coordinate = []
    y_coordinate = []
    for cnt in contours:
        coordinate = cnt[0]
        x_coordinate.append(coordinate[1])
        y_coordinate.append(coordinate[0])

    return [x_coordinate, y_coordinate]


def show_line_with_diff_color(img, contours, color='r'):
    '''This function is to hightlight to secific region of the line
        params:
        contours: contour of the line
        image: the image that want to motify
        color: color of the secific region
    '''
    for cnt in contours:
        val = cnt[0]
        img[val[1], val[0], 1] = 255
    return img


def smoothing_line(result_img, global_line_contours, local_line_contours, ranging, visualize, smoothenByX, smoothenByY, local_rate,
                   global_rate, long_rate, img_shape, highlight):
    """ This function emphasizes on smoothing specific region of the random line
      params:
      --global_line_contours: contours points of the whole line
      --local_line_contours: contours points of the regional line
      --ranging: range of contours of local line in global line
      --rate: how much user want to smoothen
      --smoothenByX: only smoothing by the X coordinate
      --smoothenByY: only smoothing by the Y coodinate
      --highlight: just show the local line with different color
    """

    local_line_x, local_line_y = extract_coordinate_from_contours(local_line_contours)
    global_line_x, global_line_y = extract_coordinate_from_contours(global_line_contours)

    if highlight:

        cv2.drawContours(result_img, [global_line_contours], -1, (255, 0, 255), 1)
        result_img = show_line_with_diff_color(result_img, local_line_contours, 'r')
        return result_img, global_line_contours

    else:
        # Smoothing only for local line

        for i in range(1, local_rate):

            r_x = i if smoothenByX else 1
            r_y = i if smoothenByY else 1

            new_local_line_y = gaussian_filter1d(local_line_y, r_y)
            new_local_line_x = gaussian_filter1d(local_line_x, r_x)

            new_local_line = [[list(a)] for a in zip(new_local_line_y, new_local_line_x)]
            new_local_line = np.asarray(new_local_line)
            global_line_contours[ranging[0]:ranging[1]] = new_local_line

            # visuale lize the process local smoothing
            if visualize:
                replicate_global = global_line_contours.copy()

                blank = np.zeros(img_shape)
                blank = convert_color_img(blank, 'x')
                cv2.drawContours(blank, [global_line_contours], -1, (255, 0, 255), 1)
                blank = show_line_with_diff_color(blank, new_local_line, 'r')
                plt.imshow(blank)
                plt.show()

        # global gaussian
        global_line_x, global_line_y = extract_coordinate_from_contours(global_line_contours)

        # define
        # font, back meaning starting node and ending node respectively
        if global_rate != 0:
            font_start = ranging[0] - int(ranging[0] * long_rate)
            font_end = ranging[0] + int(ranging[0] * long_rate)
            back_start = ranging[1] - int(ranging[1] * long_rate)
            back_end = ranging[1] + int(ranging[1] * long_rate)

            # global smoothening

            global_line_x[font_start:font_end] = gaussian_filter1d(global_line_x[font_start:font_end], global_rate)
            global_line_y[font_start:font_end] = gaussian_filter1d(global_line_y[font_start:font_end], global_rate)
            global_line_x[back_start:back_end] = gaussian_filter1d(global_line_x[back_start:back_end], global_rate)
            global_line_y[back_start:back_end] = gaussian_filter1d(global_line_y[back_start:back_end], global_rate)
            global_line_y[font_start:back_end] = gaussian_filter1d(global_line_y[font_start:back_end], global_rate)

        new_global_line = [[list(a)] for a in zip(global_line_y, global_line_x)]
        new_global_line = np.asarray(new_global_line)

        #print(new_global_line[0][-1])
        # show final result
        #new_global_line = new_global_line.append(new_global_line, [new_global_line[0][-1]], axis=0)
        result_img = cv2.drawContours(result_img, [new_global_line], -1, (255, 0, 255), 1)

        return result_img, new_global_line

def convert_color_img(img, color):
    """
    Convert color of character of binary image [0, 255]
    :param img: cv2 binary image
    :param color: 'r'/'b'/'g': convert to red/blue/green
    :return: numpy color image
    """
    cv_rgb_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    np_rgb_color = np.array(cv_rgb_img)
    noChannel = []
    if color == 'r':
        noChannel.append(0)
    elif color == 'g':
        noChannel.append(1)
    elif color == 'b':
        noChannel.append(2)
    else:
        noChanne = [0,1,2]
    for color_index in noChannel:
        np_rgb_color[np_rgb_color[:, :, color_index] == 0, color_index] = 255
    return np_rgb_color

def is_inside_contour_and_get_local_line(all_points_x, all_points_y, contours):
    start_cnt = None
    end_cnt = None
    index = 0
    previous = 0
    mul_range = []
    print(len(list_contours(contours)))
    for i, cnt in enumerate(list_contours(contours)):
        px, py = cnt
        if is_inside_polygon(all_points_x, all_points_y, px, py) == 1:
            if index == previous:
                start_cnt = i
                index += 1
                # print("start_cnt: ",start_cnt)
            if index > previous:
                end_cnt = i
        else:
            if index > previous:
                mul_range.append([start_cnt, end_cnt])
                #print('mul_range',mul_range)
                index = previous
            start_cnt = None
            end_cnt = None

    if start_cnt or end_cnt is not None:
        mul_range.append([start_cnt, end_cnt])

    if len(mul_range) == 0:
        return False, None
    else:
        return True, mul_range

    # if start_cnt or end_cnt is not None:
    #     return True, start_cnt, end_cnt
    # else:
    #     return False, None, None





if  __name__ == "__main__":
    x = [110, 106, 186, 211]
    y = [379, 411, 427, 414]
    xy = []
    for  i in zip(x,y):
        xy.append(list(i))
    print(xy)
    px = 414
    py = 135
    normalized_pred_img = cv2.imread(r'C:\Users\Admin\PycharmProjects\mocban_tool\static\uploads\36.png', 0)
    global_contours, hierarchy = cv2.findContours(normalized_pred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    global_contours = global_contours[0]
    #print(list_contours(global_contours))
    start = 0
    end = 0
    count = 0
    print("is_inside_polygon:",is_inside_polygon(x,y, 421,172))
    for i,cnt in enumerate(list_contours(global_contours)):
            px, py = cnt
            if is_inside_polygon(x, y, px, py) == 1:
                if count == 0:
                    start = i
                else:
                    end = i
                count +=1

    ranging = [start, end]
    line = global_contours[ranging[0]:ranging[1]]
    smoothed_img, g_contours = smoothing_line(global_contours, line, ranging, False, False, True, 2, 10, 0.01, normalized_pred_img.shape)

    print(is_inside_polygon(x, y, px, py))