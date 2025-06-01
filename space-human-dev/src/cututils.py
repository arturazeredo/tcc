
def cut_from_mask(img, mask, pmargin=False, percentile=5, top_margin=0.02, bottom_margin=0.02, left_margin=0.01,
          right_margin=0.01):
    max_area = 0
    coords = None
    for region in measure.regionprops(mask):#, coordinates='rc'):
        if region.area > max_area:
            max_area = region.area
            coords = region.coords
    right = np.percentile(coords[:, 1], 100 - percentile)
    left = np.percentile(coords[:, 1], percentile)
    top = np.percentile(coords[:, 0], percentile)
    bottom = np.percentile(coords[:, 0], 100 - percentile)
    if pmargin:
        width = right - left
        height = bottom - top
        right_margin = width * right_margin
        left_margin = width * left_margin
        top_margin = height * top_margin
        bottom_margin = height * bottom_margin

    right += right_margin
    left -= left_margin
    top -= top_margin
    bottom += bottom_margin
    left = np.round(left).astype(np.int64)
    right = np.round(right).astype(np.int64)
    top = np.round(top).astype(np.int64)
    bottom = np.round(bottom).astype(np.int64)
    top = np.clip(top, 0, img.shape[0])
    bottom = np.clip(bottom, 0, img.shape[0])
    left = np.clip(left, 0, img.shape[1])
    right = np.clip(right, 0, img.shape[1])
    foto = img[top:bottom, left:right]

    posicao_x = (right + left - 1) / 2.0
    posicao_x /= img.shape[1]

    posicao_y = (top + bottom - 1) / 2.0
    posicao_y /= img.shape[0]
    return foto, (posicao_y, posicao_x)


def cut_from_box(img, box, top_margin=0.02, bottom_margin=0.02, left_margin=0.01, right_margin=0.01, rotate=True):
    copyimg = img.copy()
    top = box[0]
    bottom = box[2]
    left = box[1]
    right = box[3]
    width = right - left
    height = bottom - top

    right_margin = width * right_margin
    left_margin = width * left_margin
    top_margin = height * top_margin
    bottom_margin = height * bottom_margin

    right += right_margin
    left -= left_margin
    top -= top_margin
    bottom += bottom_margin
    left = np.round(left).astype(np.int64)
    right = np.round(right).astype(np.int64)
    top = np.round(top).astype(np.int64)
    bottom = np.round(bottom).astype(np.int64)

    foto = copyimg[top:bottom, left:right]

    posicao_x = (right + left - 1) / 2.0
    posicao_x /= copyimg.shape[1]

    posicao_y = (top + bottom - 1) / 2.0
    posicao_y /= copyimg.shape[0]

    if rotate:
        return rotate_bond(foto, -90), (posicao_y, posicao_x)

    else:
        return foto, (posicao_y, posicao_x)
    


def wrap_increment(inputImg, inputMask, margin=0.01, percentile=5):
    max_area = 0
    coords = None

    contours = cv2.findContours(inputMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    if len(contours) != 0:
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        angle = cv2.minAreaRect(contours[0])[2]

        mask_rot = rotate_bond(inputMask, -angle, back=False)
        img_rot = rotate_bond(inputImg, -angle)

        for region in measure.regionprops(mask_rot, coordinates='rc'):
            if region.area > max_area:
                max_area = region.area
                coords = region.coords

        right = np.percentile(coords[:, 1], 100 - percentile)
        left = np.percentile(coords[:, 1], percentile)
        top = np.percentile(coords[:, 0], percentile)
        bottom = np.percentile(coords[:, 0], 100 - percentile)

        width = right - left
        height = bottom - top
        right_margin = width * margin
        left_margin = width * margin
        top_margin = height * margin
        bottom_margin = height * margin

        right += right_margin
        left -= left_margin
        top -= top_margin
        bottom += bottom_margin

        left = np.round(left).astype(np.int64)
        right = np.round(right).astype(np.int64)
        top = np.round(top).astype(np.int64)
        bottom = np.round(bottom).astype(np.int64)
        top = np.clip(top, 0, img_rot.shape[0])
        bottom = np.clip(bottom, 0, img_rot.shape[0])
        left = np.clip(left, 0, img_rot.shape[1])
        right = np.clip(right, 0, img_rot.shape[1])
        cut = img_rot[top:bottom, left:right]

        return cut
    else:
        return None


def wapp_cut_without_rotated(inputImg,inputMask,scale=0.0):
    img_trans = None
    contours = cv2.findContours(inputMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2]
    if len(contours) != 0:
        contours = sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        pt_A = [box[0][0], box[0][1]]
        pt_B = [box[1][0], box[1][1]]
        pt_C = [box[2][0], box[2][1]]
        pt_D = [box[3][0], box[3][1]]
        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))
        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_D,pt_C,pt_B,pt_A])
        hp = int(maxHeight*scale)
        wp = int(maxWidth*scale)
        output_pts = np.float32([[wp, hp],
                            [wp, maxHeight+hp],
                            [maxWidth-wp,maxHeight+hp],
                            [maxWidth-wp, hp]])

        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        try:
            img_trans = cv2.warpPerspective(inputImg,M,(maxWidth+wp, maxHeight+hp),
                                            flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
        except:
            return None

        return img_trans
    else:
        return None
