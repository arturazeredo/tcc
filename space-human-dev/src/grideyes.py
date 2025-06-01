import numpy as np

from src.geometry import *
from src.antispoffing import Anti

class Grid:
    def __init__(self, shape=(480,640)):
        self.x = 0
        self.h = shape[0]

        self.map_eyes = {34: 0, 134: 1, 264: 2, 363: 3,
                         469: 4, 470: 5, 471: 6, 472: 7,
                         473: 8, 474: 9, 475: 10, 476: 11,
                         477: 12, 478: 13}
        #self.map_eyes = {264: 0, 475: 1, 474: 2, 477: 3,
        #                  363: 4, 476: 5, 478: 6, 134: 7,
        #                 470: 8, 469: 9, 472: 10, 34: 11,
        #                 471: 12, 473: 13}
        self.resp = {477: {'x': None, 'y': None},
                     475: {'x': None, 'y': None},
                     472: {'x': None, 'y': None},
                     470: {'x': None, 'y': None}}

        #self.model = Anti()


    def verifique_face(self, images):
        if len(images) == 0:
            return 0
        results = [self.model.inference(i) for i in images]
        m = np.median(results)
        return m



    def localizar(self, point, counter):

        if point == 477:
            px, py = counter[self.map_eyes[477]]
            lx, _ = counter[self.map_eyes[363]]
            _, ty = counter[self.map_eyes[475]]
            _, by = counter[self.map_eyes[478]]
            #--------------------
            dxo, _ = counter[self.map_eyes[475]]
            dxr, _ = counter[self.map_eyes[264]]
            #--------------------
            dxl = distanceX_(px, lx)
            dyt = distanceY_(py, ty)
            dyb = distanceY_(py, by)
            #---------------------
            dxor = distanceX_(dxo, dxr)

            if dxl < dxor - dxor * 0.2:
                self.resp[477]["x"] = "rigth"
            elif dxor/dxl > 0.7:
                self.resp[477]["x"] = "central"
            else:
                self.resp[477]["x"] = "left"

            if dyt < dyb - dyb*0.2:
                self.resp[477]["y"] = "bottom"
            elif dxor/dxl > 0.79:
                self.resp[477]["y"] = "central"
            else:
                self.resp[477]["y"] = "top"
        if point == 475:
            px, py = counter[self.map_eyes[475]]
            lx, _ = counter[self.map_eyes[264]]
            _, ty = counter[self.map_eyes[476]]
            _, by = counter[self.map_eyes[478]]
            dxo, _ = counter[self.map_eyes[477]]
            dxl, _ = counter[self.map_eyes[363]]
            dxr = distanceX_(px,lx)
            dyb = distanceY_(py,by)
            dyt = distanceY_(py,ty)
            dxol = distanceX_(dxo, dxl)

            if dxr < dxol-dxol*0.2:
                self.resp[475]['x'] = "left"
            elif dxol/dxr > 0.79:
                self.resp[475]['x'] = "central"
            else:
                self.resp[475]['x'] = "rigth"

            if dyt < dyb-dyb*0.2:
                self.resp[475]['y'] = "bottom"
            elif dyb/dyt > 0.79:
                self.resp[475]['y'] = "central"
            else:
                self.resp[475]['y'] = "bottom"
        if point == 472:
            px, py = counter[self.map_eyes[472]]
            dr, _ = counter[self.map_eyes[34]]
            _, ty = counter[self.map_eyes[471]]
            _, by = counter[self.map_eyes[473]]
            dxo, _ = counter[self.map_eyes[470]]
            dxl, _ = counter[self.map_eyes[134]]

            dxr = distanceX_(px, dr)
            dyt = distanceY_(py, ty)
            dyb = distanceY_(py, by)
            dxol = distanceX_(dxo,dxl)

            if dxr < dxol-dxol*0.2:
                self.resp[472]['x'] = "right"
            elif dxol/dxr > 0.79:
                self.resp[472]['x'] = "central"
            else:
                self.resp[472]['x'] = "left"

            if dyt < dyb-dyb*0.2:
                self.resp[472]['y'] = 'bottom'
            elif dyb/dyt > 0.79:
                self.resp[472]['y'] = 'central'
            else:
                self.resp[472]['y'] = "top"
        if point == 470:
            px, py = counter[self.map_eyes[470]]
            lx, _ = counter[self.map_eyes[134]]
            _, ty = counter[self.map_eyes[471]]
            _, by = counter[self.map_eyes[473]]
            dxo, _ = counter[self.map_eyes[472]]
            dxl, _ = counter[self.map_eyes[34]]

            dix = distanceX_(px,lx)
            dyt = distanceY_(py,ty)
            dyb = distanceY_(py,by)
            dxol = distanceX_(dxo, dxl)

            if dix < dxol - dxol*0.2:
                self.resp[470]['x'] = "left"
            elif dxol/dix > 0.79:
                self.resp[470]['x'] = "central"
            else:
                self.resp[470]['x'] = "right"

            if dyt < dyb-dyb*0.2:
                self.resp[470]['y'] = "bottom"

            elif dyb/dyt > 0.79:
                self.resp[470]['y'] = "central"

            else:
                self.resp[470]['y'] = "top"

        return self.resp


    def calcule_all_eyes(self,counter):
        self.localizar(477, counter)
        self.localizar(475, counter)
        self.localizar(472, counter)
        self.localizar(470, counter)
        return self.resp


    def verificar(self, m, buffer=[]):

        if self.verifique_face(buffer) > 0.8:
            return "Fake"
        if m > 4:
            return "Sucesso"
        else:
            return "Failed"















