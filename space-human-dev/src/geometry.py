from skimage import measure
from src.utils import rotate_bond
import numpy as np
import cv2


def struct_points(points):
    dic = {}
    for i, p in enumerate(points):
        dic[i] = p

    return points

def distance_entre_points(points,p1,p2):
    if len(points) < p2:
        return None

    (xi,yi) = points[p1]
    (xf,yf) = points[p2]

    dist = np.sqrt(np.power(yf-yi,2)+np.power(xf-xi,2))

    return dist

def distanceX_(xf,xi):
    return  np.sqrt(np.power((xf-xi),2))

def distanceY_(yf,yi):
    return np.sqrt(np.power((yf-yi),2))


def distanceX(points,px1,px2):
    (xi,_) = points[px1]
    (xf,_) = points[px2]
    return xf-xi

def distanceY(points,py1,py2):
    (_,yi) = points[py1]
    (_,yf) = points[py2]
    return yf-yi

def distance_vectorial(points,px1,px2):
    (xi,yi) = points[px1]
    (xf,yf) = points[px2]
    return (xf-xi,yf-yi)

def proporcao_face_eixoX(points,shape,px1=0,px2=16):

    (xi, _) = points[px1]
    (xf, _) = points[px2]

    dist_x = xf-xi
    if dist_x is None:
        return None

    _, w = shape[:2]

    return dist_x/shape[1]

def proporcao_face_eixoY(points, shape, py=9):

    (_, y1) = points[19]
    (_, y2) = points[24]

    if y1 > y2:
        yi = y1
    else:
        yi = y2

    (_,yf) = points[py]

    dist_y = yf-yi

    return dist_y/shape[0]


def calcule_face_from_distance(points,direct='left'):

    if direct == 'left':
        return distance_entre_points(points,17,29) # 17: lábio inferior centro, 29: pálpebra direita

    if direct == 'right':
        return distance_entre_points(points,1,29) # 1: lábio inferior centro, 29: pálpebra direita


def get_points_region(points,direct='left',box=None):
    if len(points) > 0:
        coord = []

        if direct == 'top':
            if box is not None:
                coord = [1,18,19,20,21,23,24,25,26,17]
                shape = [points[c - 1] for c in coord]
                x,y,w,h = box
                shape.append((x+w,y))
                shape.append((x,y))
                return np.array(shape)
            else:
                print("Erro two points no definition, box face necessary,"
                      " paraments box is {}".format(type(box)))
                return []

        if direct == 'left':
            coord = [9,10,11,12,13,14,15,16,17,27,26,25,
                     24,23,28,29,30,31,34,52,63,67,58]

        if direct == 'right':
            coord = [22,21,20,19,18,1,2,3,4,5,6,7,8,
                     9,58,67,63,52,34,31,30,29,28]

        if direct == 'eyes_l':
            coord = [48,47,46,45,44,43]

        if direct == 'eyes_r':
            coord = [42,41,40,39,38,37]

        if direct == 'ocular_r':
            coord = [17,27,27,25,24,23,28,29]

        if direct == 'ocular_l':
            coord = [1,29,28,22,21,20,19,18]

        if direct == 'nose':
            coord = [32,33,33,35,36,28]

        if direct == 'lips_ex':
            coord = [49,60,59,58,57,56,
                     55,54,53,52,51]

        if direct == 'lips_in':
            coord = [68,67,66,65,64,63,62,61]

        if direct == 'down':
            coord = [5,6,7,8,9,10,11,12,13,
                     55,56,57,58,59,60,49]

        shape = [points[c - 1] for c in coord]
        return np.array(shape)

    else:
        return []


def get_points_media(points,direct='left'):
    if len(points) > 0:
        coord = []
        if direct == 'iris_r':
            coord = [477,474,475]

        if direct == 'eyes_mr': # medium right
            coord = [264, 475, 474,
                     477, 363, 387, 375]

        if direct == 'iris_analyser':
            coord = [264, 475, 474,
             477, 363, 387, 375,
             134, 470, 469,
             472, 34, 160, 146]


        if direct == 'iris_analyser2':
            coord = [264, 475, 474,
                     477, 363, 476, 478,
                     134, 470, 469,
                     472, 34, 471, 473]
            

        if direct == 'iris_analyser3': # no center, upper and bottom points
            coord = [264, 475,
                     477, 363,
                     134, 470,
                     472, 34]


        if direct == 'eyes_ml': # medium left
            coord = [134, 470, 469,
                     472, 34, 471, 473]

        if direct == 'iris_l': # left
            coord = [472,469,470]

        if direct == 'iris_lq': # iris countour (without center point)
            coord = [472, 471, 470, 473]

        if direct == 'iris_rq': # iris countour (without center point)
            coord = [477, 475, 476 , 478]
        
        if direct == 'iris_q': # iris countour both sides (without center point)
            coord = [477, 475, 476, 478, 472, 471, 470, 473]
        
        if direct == 'iris_cl': # iris center left
            coord = [469]

        if direct == 'iris_cr': # iris center right
            coord = [474]
        
        if direct == 'iris':
            coord = [477, 474, 475,
                     472, 469, 470]

        if direct == 'eyes_r':
            coord = [250,264,467,389,388,387,386,385,
                     399,363,383,382,381,375,374,391]

        if direct == 'eyes_l':
            coord = [134,174,158,159,160,161,162,247,
                     34,8,164,145,146,154,155,156]

        if direct == 'right':
            coord = [11,10,9,7,6,5,2,3,1,12,15,
                     18,200,176,153,378,401,379,
                     380,366,398,362,324,455,357,
                     390,252,285,333,298,339]

        if direct == 'left':
            coord = [11,10,9,7,6,5,2,3,1,12,15,
                     18,200,176,153,149,177,150,
                     151,137,173,59,133,94,235,
                     128,163,22,55,104,69,110]
        if direct == 'nose_countour':
            coord = [7, 21, 80, 95, 115, 116, 123, 
                     132, 142, 189, 199, 218, 219, 
                     243, 251, 310, 344, 345, 352, 
                     361, 371, 413, 421, 438, 439, 463]
        if direct == 'nose':
            coord = [2, 4, 5, 6, 7, 20, 21, 45, 46, 
                     52, 80, 95, 115, 116, 123, 126,
                     132, 135, 142, 189, 196, 199, 218,
                     219, 221, 235, 237, 238, 239, 240, 
                     241, 242, 243, 249, 251, 275, 276, 
                     282, 310, 344, 345, 352, 354, 355, 
                     361, 364, 371, 400, 413, 421, 438, 
                     439, 441, 457, 458, 459, 460, 459, 
                     460, 462, 463]
        if direct == 'eyelid_l':
            coord = [28, 29, 30, 159, 160, 161] # pairs: (0,4), (1,3), (2,5)
        if direct == 'eyelid_r':
            coord = [258, 259, 260, 386, 387, 388] # pairs: (0,4), (1,3), (2,5)
        shape = [points[c - 1] for c in coord]
        return np.array(shape)

    else:
        return []

def center_contour(cnt):
    (cX, cY) = (0,0)
    if len(cnt) > 0:
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return cX, cY

def diff_contour(old_cnt,new_cnt):

    old_cX, old_cY = center_contour(old_cnt)
    new_cX, new_cY = center_contour(new_cnt)

    dX = new_cX - old_cX
    dY = new_cY - old_cY

    return dX, dY


def diff_central_eyes(old_cp, new_cp):

    ocx, ocy = old_cp
    ncx, ncy = new_cp

    diff_cx = ncx-ocx
    diff_cy = ncy-ocy

    return diff_cx, diff_cy



def diff_points(old_points,new_points,points=[1,17,34]):

    max_p = max(points)
    diffX = []
    diffY = []

    if len(old_points) >= max_p and len(new_points) >= max_p:
        for c in points:
            (ox,oy) = old_points[c-1]
            (nx,ny) = new_points[c-1]

            diffX.append(nx-ox)
            diffY.append(ny-oy)

    return diffX, diffY

def variable_area_from_contour(old_counts,new_counts,margin=0.001):
    "Counts é uma listar Count, então é importe a passagem de multiplo counts"
    "Return listar do tamanho da quantidade de count, com valores 1,-1,0,"
    "1 = aumentou, -1 = diminuiu, 0 = dentro da margin de erro"

    lc = len(old_counts)
    rc = len(new_counts)
    results = []

    if lc > 0 and (rc-lc) == 0:
        for i in range(lc):
            old_cnt = old_counts[i]
            new_cnt = new_counts[i]

            if len(old_cnt) > 0 and len(new_cnt) > 0:
                old_area = cv2.contourArea(old_cnt)
                new_area = cv2.contourArea(new_cnt)
                diff_area = abs(new_area - old_area)

                #print("OLD_AREA: {} NEW_AREA: {} DIFF: {}".format(old_area,new_area,diff_area))

                if (diff_area < old_area*margin):
                    results.append(0)

                elif new_area > old_area:
                    results.append(1)

                else:
                    results.append(-1)

    return results


def eixo_translate(diff, eixo='x'):

    if eixo == 'x':
        #if abs(diff_sum/eixo_pixel) < margin:
        #    return "central"
        if diff < 0:
            return 'left'
        elif diff > 0:
            return 'right'
        else:
            return 'central'
    if eixo == 'y':
        if diff < 0:
            return 'top'
        elif diff > 0:
            return 'bottom'
        else:
            return 'central'


def direct_translate(diffX, diffY,shape,margin=0.05):
    check_sumX, check_sumY = 0, 0

    for dX, dY in zip(diffX,diffY):

        check_sumX += dX
        check_sumY += dY

    stringX = eixo_translate(check_sumX, eixo='x')
    stringY = eixo_translate(check_sumY, eixo='y')

    return stringX, stringY


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

