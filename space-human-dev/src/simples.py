import math

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

    dist = math.sqrt(math.power(yf-yi,2)+math.power(xf-xi,2))

    return dist

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
    (_, y2)  = points[24]

    if y1 > y2:
        yi = y1
    else:
        yi = y2

    (_,yf) = points[py]

    dist_y = yf-yi

    return dist_y/shape[0]