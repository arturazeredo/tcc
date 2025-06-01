from src.visualize import class_color
import random

def random_color():
    id = random.randint(0, 64)
    return class_color[id]


def random_colores(number):
    return [random_color() for _ in range(number)]

def random_possicao(shape, number=6):
    pxs = []
    pys = []
    for i in range(number):
        cx = random.randrange(int(shape[1]*0.1), shape[1], int(shape[1]/number))
        cy = random.randrange(int(shape[0]*0.1), shape[0], int(shape[0]/number))
        pxs.append(cx)
        pys.append(cy)

    return pxs, pys


def localizate_coord_centroid(point,quadra,number=6):
    cx, cy = point
    qx, qy = quadra

    transx = {0: 'lelf', 1: 'lelf', 2: 'central', 3:'central', 4:'rigth', 5:'rigth'}
    transy = {0: 'bottom', 1: 'bottom', 2: 'central', 3:'central', 4: 'top', 5:'top'}

    resx = []
    resy = []

    for c in cx:
        for i, x in enumerate(qx):
            if c < x:
                resx.append(transx[i-1])

    for c in cy:
        for i, y in enumerate(qx):
            if c < y:
                resy.append(transy[i-1])

    return resx, resy


def construct_task(shape, number=6):
    quadrax = [i for i in range(0, shape[1], int(shape[1]/number))]
    quadray = [i for i in range(1, shape[0], int(shape[1]/number))]

    cxs, cys = random_possicao(shape)
    colors = random_colores(number)
    txr, tyr = localizate_coord_centroid((cxs, cys),(quadrax, quadray))
    return {'cores': colors, 'cxs': cxs, 'cys': cys, 'txr': txr, 'tyr': tyr}





