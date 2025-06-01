from argparse import ArgumentParser
from src.mediapipe import GooMedia
from src.randomgamer import construct_task
from src.buffer import BufferR
import hashlib
from src import geometry
from src import visualize
import src.datalog as dl
from src.grideyes import Grid
import numpy as np
#import dlib
import cv2
import os

parser = ArgumentParser()
parser.add_argument("-c","--cam",type=int,default=0)
parser.add_argument("-i","--image",default="")
args = vars(parser.parse_args())


CAM_AVALIABLE = True
SERVER_RUN = False
pface = GooMedia(1080) # Extrator dos pontos
buff = BufferR()
file_name = "data/info-" + dl.get_now_string() + '.log'
os.makedirs('data', exist_ok=True)


dl.create_logging(is_server=False,dev=file_name)

user = "pedro".encode()
ticket = hashlib.sha1(user).hexdigest()
counter_delay = 0
draw_enable = False
scores = []
step_next = False
runtime= 0.0
grid_eyes = Grid()
ship_step = False

buffing = []


if len(args["image"]) > 0:
    CAM_AVALIABLE = False

if CAM_AVALIABLE:
    cam = cv2.VideoCapture(args["cam"])
    info_img = {'old': {'contour': None, 'points': None, 'box': None},
                'new': {'contour': None, 'points': None, 'box': None}}
    frame_start = True

    five_buffer = []
    c = 0
    switch = False
    old_frame = []
    while CAM_AVALIABLE:

        _, frame = cam.read()
        # print(frame.shape)

        if frame is not None:

            imgcopy = frame.copy()
            di = dl.now()
            _, landmask_list = pface.extract(frame) # Imagem capturada em determinado frame
            points = pface.points_normalize(landmask_list, imgcopy.shape, index=0) # Normalização, explicar melhor
            df = dl.now()
            dl.show_points("CAM_POINTS_", di, df, points, box=None, print_m=not (SERVER_RUN), ticket=ticket)

            iris_analyser = geometry.get_points_media(points, direct='iris_analyser')

            if frame_start:
                info_img['new']['contour'] = [iris_analyser]
                info_img['new']['points'] = points
                frame_start = False
                info_task = construct_task(frame.shape[:2])

            else:

                info_img['old']['contour'] = info_img['new']['contour']
                info_img['old']['points'] = info_img['new']['points']
                info_img['old']['box'] = info_img['new']['box']

                info_img['new']['contour'] = [iris_analyser]
                info_img['new']['points'] = points

                if len(iris_analyser) > 0:

                    direction = grid_eyes.calcule_all_eyes(iris_analyser)

                    frame = visualize.draw_text(frame, "Rigth Analyser: {}".format(direction), diff=(50, 400))

                    if draw_enable:

                        ptsx = 0
                        ptsy = 0
                        for i in [472, 470]:
                            if info_task['txr'][counter_delay] == direction[i]['x']:
                                ptsx +=2
                            if info_task['tyr'][counter_delay] == direction[i]['y']:
                                ptsy +=1

                        for i in [477, 475]:
                            if info_task['txr'][counter_delay] == direction[i]['x']:
                                ptsx += 2
                            if info_task['tyr'][counter_delay] == direction[i]['y']:
                                ptsy +=1


                        wait_eyes = (dl.now()-runtime).total_seconds()
                        if (ptsy+ptsx) > 4:
                            scores.append(1)
                            counter_delay +=1
                            if len(buffing) <= 10:
                                buffing.append(frame)

                        elif wait_eyes > 5:
                            runtime = dl.now()
                            scores.append(0)
                            counter_delay +=1
                            if len(buffing) <= 10:
                                buffing.append(frame)

            if counter_delay >= 6:
                text = grid_eyes.verificar(np.median(scores))
                draw_enable = False
                frame = visualize.draw_text(frame, text, diff=(50, 250),font_size=0.5)
                frame = visualize.draw_text(frame, "{}".format(scores), diff=(50, 200),font_size=0.5)
                if text == "Fake":
                    cv2.imshow("Frist Image", buffing[0])
                    cv2.imshow("Last Image", buffing[-1])

            if draw_enable:

                frame = visualize.draw_circule(frame, (info_task['cxs'][counter_delay],
                                                       info_task['cys'][counter_delay]),
                                               info_task['cores'][counter_delay])


            cv2.imshow("cam", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                info_task = construct_task(frame.shape[:2])
                runtime = dl.now()
                counter_delay = 0

            if key == ord("b"):
                runtime = dl.now()
                draw_enable = True

cv2.destroyAllWindows()