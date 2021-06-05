import numpy as np
import cv2
import sys


def get_image():
    """
    load the image
    :return: left image and right image
    """
    print("load img started")
    # left_img = cv2.imread('cones_im2.png', 0)
    # right_img = cv2.imread('cones_im6.png', 0)
    # left_img = cv2.imread('mask_im0.png', 0)
    # right_img = cv2.imread('mask_im1.png', 0)
    left_img = cv2.imread('art_im1.png', 0)
    right_img = cv2.imread('art_im5.png', 0)
    # left_img = cv2.imread('computer_view1.png', 0)
    # right_img = cv2.imread('computer_view5.png', 0)
    # left_img = cv2.imread('stick_view1.png', 0)
    # right_img = cv2.imread('stick_view5.png', 0)

    left_img = cv2.resize(left_img, (450, 375))  # in dataset, the shape uses to be (375, 450)
    right_img = cv2.resize(right_img, (450, 375))  # another possible choice can be (640, 480)
    left_img = cv2.GaussianBlur(left_img, (5, 5), 0)
    right_img = cv2.GaussianBlur(right_img, (5, 5), 0)
    print("load img finished, shape is ", right_img.shape)
    return left_img, right_img


def calculate_census_cost(imgl, imgr, c_s, max_d):
    """
    :param imgl: left image
    :param imgr: right image
    :param c_s: size of the square census kernel
    :param max_d: the max disparity
    :return: left census volume and right census volume
    """
    H = imgl.shape[0]
    W = imgl.shape[1]
    half_c_s = int(c_s / 2)  # border pixels that can be skipped

    imgl_census_code = np.zeros((H, W), np.uint64)
    imgr_census_code = np.zeros((H, W), np.uint64)
    print("census code started")
    for i in range(half_c_s, H - half_c_s):
        for j in range(half_c_s, W - half_c_s):
            census_code_l = 0
            center_gray = imgl[i, j]
            for m in range(- half_c_s, half_c_s + 1):
                for n in range(- half_c_s, half_c_s + 1):
                    if (m, n) != (0, 0):
                        if imgl[i + m, j + n] < center_gray:
                            census_code_l += 1
                        if (m, n) != (half_c_s, half_c_s):  # make sure the length is right
                            census_code_l = census_code_l << 1
            # save in the census code map
            imgl_census_code[i, j] = census_code_l

            census_code_r = 0
            center_gray = imgr[i, j]
            for m in range(- half_c_s, half_c_s + 1):
                for n in range(- half_c_s, half_c_s + 1):
                    if (m, n) != (0, 0):
                        if imgr[i + m, j + n] < center_gray:
                            census_code_r += 1
                        if (m, n) != (half_c_s, half_c_s):  # make sure the length is right
                            census_code_r = census_code_r << 1
            imgr_census_code[i, j] = census_code_r
            # print('census code progress: ', i - half_c_s, '/', H - 2 * half_c_s)

    imgl_cost_volume = np.zeros((H, W, max_d), np.uint32)  # the final result
    imgr_cost_volume = np.zeros((H, W, max_d), np.uint32)

    # calculate the census cost considering the disparity
    for d in range(0, max_d):
        imgl_census_value = np.zeros((H, W), np.uint64)  # save the hamming distance
        imgr_census_value = np.zeros((H, W), np.uint64)

        imgl_census_code_offset = np.zeros((H, W), np.uint64)
        imgr_census_code_offset = np.zeros((H, W), np.uint64)

        imgr_census_code_offset[:, half_c_s + d: W - half_c_s] = imgr_census_code[:, half_c_s: W - half_c_s - d]
        imgl_xor = np.bitwise_xor(imgl_census_code, imgr_census_code_offset)
        while not np.all(imgl_xor == 0):
            nonzero_mask = imgl_xor != 0
            imgl_census_value[nonzero_mask] += 1
            imgl_xor = np.bitwise_and(imgl_xor - 1, imgl_xor)

        imgl_census_code_offset[:, half_c_s: W - half_c_s - d] = imgl_census_code[:, half_c_s + d: W - half_c_s]
        imgr_xor = np.bitwise_xor(imgr_census_code, imgl_census_code_offset)
        while not np.all(imgr_xor == 0):
            nonzero_mask = imgr_xor != 0
            imgr_census_value[nonzero_mask] += 1
            imgr_xor = np.bitwise_and(imgr_xor - 1, imgr_xor)

        imgl_cost_volume[:, :, d] = imgl_census_value
        imgr_cost_volume[:, :, d] = imgr_census_value
        print("Disparity %d finished." % d)

    return imgl_cost_volume, imgr_cost_volume


def cost_aggregation(c_v, max_d):
    """
    the formula
    Lr(p,d) = C(p,d)
            + min[Lr(p-r, d),Lr(p-r, d-1)+p1,Lr(p-r,d+1)+p1,miniLr(p-r, i)+p2]
            - minkLr(p-r,k)
    :param c_v:
    :param max_d:
    :return: sum of all the Lr
    """
    (H, W, D) = c_v.shape
    p1 = 10
    p2 = 120
    Lr1 = np.zeros((H, W, D), np.uint32)  # from up
    Lr2 = np.zeros((H, W, D), np.uint32)  # from left
    Lr3 = np.zeros((H, W, D), np.uint32)  # from right
    Lr4 = np.zeros((H, W, D), np.uint32)  # from down
    # Lr5 = np.zeros((H, W, D), np.uint32)  # from up_left

    print('agg from up started.')
    Lr1[0, :, :] = c_v[0, :, :]  # border, first row
    for r in range(1, H):
        for d in range(0, max_d):
            Lr1_1 = np.squeeze(Lr1[r - 1, :, d])
            if d != 0:  # disparity is not the bottom
                Lr1_2 = np.squeeze(Lr1[r - 1, :, d - 1] + p1)
            else:
                Lr1_2 = np.squeeze(Lr1_1 + p1)
            if d != max_d - 1:
                Lr1_3 = np.squeeze(Lr1[r - 1, :, d + 1] + p1)
            else:
                Lr1_3 = np.squeeze(Lr1_1 + p1)
            Lr1_4 = np.squeeze(np.min(Lr1[r - 1, :, :], axis=1) + p2)
            Lr1_5 = np.min(Lr1[r - 1, :, :], axis=1)

            Lr1[r, :, d] = c_v[r, :, d] + np.min(np.vstack([Lr1_1, Lr1_2, Lr1_3, Lr1_4]), axis=0) - Lr1_5

    print('agg from left started.')
    Lr2[:, 0, :] = c_v[:, 0, :]  # border, first column
    for c in range(1, W):
        for d in range(0, max_d):
            Lr2_1 = np.squeeze(Lr2[:, c - 1, d])
            if d != 0:  # disparity is not the bottom
                Lr2_2 = np.squeeze(Lr2[:, c - 1, d - 1] + p1)
            else:
                Lr2_2 = np.squeeze(Lr2_1 + p1)
            if d != max_d - 1:
                Lr2_3 = np.squeeze(Lr2[:, c - 1, d + 1] + p1)
            else:
                Lr2_3 = np.squeeze(Lr2_1 + p1)
            Lr2_4 = np.squeeze(np.min(Lr2[:, c - 1, :], axis=1) + p2)
            Lr2_5 = np.min(Lr2[:, c - 1, :], axis=1)

            Lr2[:, c, d] = c_v[:, c, d] + np.min(np.vstack([Lr2_1, Lr2_2, Lr2_3, Lr2_4]), axis=0) - Lr2_5

    print('agg from right started.')
    Lr3[:, 0, :] = c_v[:, -1, :]  # border, last column
    for c in range(W - 2, -1, -1):
        for d in range(0, max_d):
            Lr3_1 = np.squeeze(Lr3[:, c + 1, d])
            if d != 0:  # disparity is not the bottom
                Lr3_2 = np.squeeze(Lr3[:, c + 1, d - 1] + p1)
            else:
                Lr3_2 = np.squeeze(Lr3_1 + p1)
            if d != max_d - 1:
                Lr3_3 = np.squeeze(Lr3[:, c + 1, d + 1] + p1)
            else:
                Lr3_3 = np.squeeze(Lr3_1 + p1)
            Lr3_4 = np.squeeze(np.min(Lr3[:, c + 1, :], axis=1) + p2)
            Lr3_5 = np.min(Lr3[:, c + 1, :], axis=1)

            Lr3[:, c, d] = c_v[:, c, d] + np.min(np.vstack([Lr3_1, Lr3_2, Lr3_3, Lr3_4]), axis=0) - Lr3_5

    print('agg from down started.')
    Lr4[0, :, :] = c_v[-1, :, :]  # border, last row
    for r in range(H - 2, -1, -1):
        for d in range(0, max_d):
            Lr4_1 = np.squeeze(Lr4[r + 1, :, d])
            if d != 0:  # disparity is not the bottom
                Lr4_2 = np.squeeze(Lr4[r + 1, :, d - 1] + p1)
            else:
                Lr4_2 = np.squeeze(Lr4_1 + p1)
            if d != max_d - 1:
                Lr4_3 = np.squeeze(Lr4[r + 1, :, d + 1] + p1)
            else:
                Lr4_3 = np.squeeze(Lr4_1 + p1)
            Lr4_4 = np.squeeze(np.min(Lr4[r + 1, :, :], axis=1) + p2)
            Lr4_5 = np.min(Lr4[r + 1, :, :], axis=1)

            Lr4[r, :, d] = c_v[r, :, d] + np.min(np.vstack([Lr4_1, Lr4_2, Lr4_3, Lr4_4]), axis=0) - Lr4_5

    # print('agg from up-left started')
    # Lr5[0, :, :] = c_v[0, :, :]
    # for c in range(1, W):
    #     for d in range(0, max_d):
    #         if c <= W - H - 1:  # The path does not need to be split.
    #             for x, y in zip(range(c, W, 1), range(0, H, 1)):
    #                 Lr5_1 = Lr5[y - 1, x - 1, d]
    #                 if d != 0:
    #                     Lr5_2 = Lr5[y - 1, x - 1, d - 1] + p1
    #                 else:
    #                     Lr5_2 = Lr5_1 + p1
    #                 if d != max_d - 1:
    #                     Lr5_3 = Lr5[y - 1, x - 1, d + 1] + p1
    #                 else:
    #                     Lr5_3 = Lr5_1 + p1
    #                 Lr5_4 = np.min(Lr5[y - 1, x - 1, :], axis=0) + p2
    #                 Lr5_5 = np.min(Lr5[y - 1, x - 1, :], axis=0)
    #
    #                 Lr5[y, x, d] = c_v[y, x, d] + min(Lr5_1, Lr5_2, Lr5_3, Lr5_4) - Lr5_5
    #         else:  # the pass needs to be split
    #             for x, y in zip(range(c, W, 1), range(0, W - c, 1)):  # first part
    #                 Lr5_1 = Lr5[y - 1, x - 1, d]
    #                 if d != 0:
    #                     Lr5_2 = Lr5[y - 1, x - 1, d - 1] + p1
    #                 else:
    #                     Lr5_2 = Lr5_1 + p1
    #                 if d != max_d - 1:
    #                     Lr5_3 = Lr5[y - 1, x - 1, d + 1] + p1
    #                 else:
    #                     Lr5_3 = Lr5_1 + p1
    #                 Lr5_4 = np.min(Lr5[y - 1, x - 1, :], axis=0) + p2
    #                 Lr5_5 = np.min(Lr5[y - 1, x - 1, :], axis=0)
    #
    #                 Lr5[y, x, d] = c_v[y, x, d] + min(Lr5_1, Lr5_2, Lr5_3, Lr5_4) - Lr5_5
    #             for x, y in zip(range(0, W, 1), range(W - c, H, 1)):  # second part
    #                 if x == 0:  # the head
    #                     Lr5_1 = Lr5[y - 1, W - 1, d]
    #                     if d != 0:
    #                         Lr5_2 = Lr5[y - 1, W - 1, d - 1] + p1
    #                     else:
    #                         Lr5_2 = Lr5_1 + p1
    #                     if d != max_d - 1:
    #                         Lr5_3 = Lr5[y - 1, W - 1, d + 1] + p1
    #                     else:
    #                         Lr5_3 = Lr5_1 + p1
    #                     Lr5_4 = np.min(Lr5[y - 1, W - 1, :], axis=0) + p2
    #                     Lr5_5 = np.min(Lr5[y - 1, W - 1, :], axis=0)
    #
    #                     Lr5[y, x, d] = c_v[y, x, d] + min(Lr5_1, Lr5_2, Lr5_3, Lr5_4) - Lr5_5
    #                 else:
    #                     Lr5_1 = Lr5[y - 1, x - 1, d]
    #                     if d != 0:
    #                         Lr5_2 = Lr5[y - 1, x - 1, d - 1] + p1
    #                     else:
    #                         Lr5_2 = Lr5_1 + p1
    #                     if d != max_d - 1:
    #                         Lr5_3 = Lr5[y - 1, x - 1, d + 1] + p1
    #                     else:
    #                         Lr5_3 = Lr5_1 + p1
    #                     Lr5_4 = np.min(Lr5[y - 1, x - 1, :], axis=0) + p2
    #                     Lr5_5 = np.min(Lr5[y - 1, x - 1, :], axis=0)
    #
    #                     Lr5[y, x, d] = c_v[y, x, d] + min(Lr5_1, Lr5_2, Lr5_3, Lr5_4) - Lr5_5

    return Lr1 + Lr2 + Lr3 + Lr4


def consistency_check(dl, dr):
    """
    to check the consistency between left disparity map and the right.
    :param dl:
    :param dr:
    :return:
    """
    (H, W) = dl.shape
    print('started consistency check.')
    for i in range(0, H):
        for j in range(0, W):
            dis = dl[i, j]
            if np.abs(dr[i, j - dis] - dl[i, j]) > 1:
                dl[i, j] = 0
            else:
                pass
    return dl


def d2gray(dis_img):
    """
    Map disparity to grayscale
    :param dis_img: the disparity image
    :return: gray image that can be displayed
    """
    max_d = np.max(dis_img)
    min_d = np.min(dis_img)
    a = 255.0 / (max_d - min_d)
    b = min_d / (max_d - min_d) * (-255.0)
    return np.uint8(a * dis_img + b)


def subpixel(c_v):
    """
    fail
    :param c_v: the cost volume
    :return: subpixel result
    """
    (H, W, D) = c_v.shape
    idx_y = np.zeros((H, W), np.uint16)
    idx_x = np.zeros((H, W), np.uint16)
    for j in range(0, H):
        idx_y[j, :] = j
    for i in range(0, W):
        idx_x[:, i] = i

    d_min = np.argmin(c_v, axis=2)
    cost0 = np.min(c_v, axis=2)
    cost1 = c_v[idx_y.flatten(), idx_x.flatten(), (d_min - 1).flatten()].reshape(H, W)
    cost2 = c_v[idx_y.flatten(), idx_x.flatten(), (d_min + 1).flatten()].reshape(H, W)
    return np.uint8(d_min + (cost1 - cost2) / (2 * (cost1 + cost2 - 2 * cost0) + 0.000001))


def WTA(c_v):
    return np.argmin(c_v, axis=2)


if __name__ == '__main__':
    max_disparity = 64
    raw_l_img, raw_r_img = get_image()
    c_l_v, c_r_v = calculate_census_cost(raw_l_img, raw_r_img, 7, max_disparity)

    d_l_without_agg = d2gray(WTA(c_l_v))
    cv2.imwrite('disparity_l_without_agg.png', d_l_without_agg)
    d_r_without_agg = d2gray(WTA(c_r_v))
    cv2.imwrite('disparity_r_without_agg.png', d_r_without_agg)

    c_l_v_agg = cost_aggregation(c_l_v, max_disparity)
    d_l = WTA(c_l_v_agg)
    # d_l = subpixel(c_l_v_agg)
    d_l_gray = d2gray(d_l)
    cv2.imwrite('disparity_l_agg.png', d_l_gray)
    c_r_v_agg = cost_aggregation(c_r_v, max_disparity)
    d_r = WTA(c_r_v_agg)
    d_r_gray = d2gray(d_r)
    cv2.imwrite('disparity_r_agg.png', d_r_gray)

    d_l_check = consistency_check(d_l, d_r)
    d_l_check_gray = d2gray(d_l_check)
    cv2.imwrite('disparity_check.png', d_l_check_gray)
