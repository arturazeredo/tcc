from datetime import datetime,timezone,timedelta
import json
import base64
import logging


INFO_FORMAT = '[%(name)s][%(levelname)s]%(message)s'



def create_logging(is_server=False,dev="data/info.log"):

    if not is_server:
        logging.basicConfig(filename=dev, filemode='w',level=logging.DEBUG,format=INFO_FORMAT)
        #format=INFO_FORMAT)

    return True

def get_now():
    now = datetime.now()
    diferenca = timedelta(hours=-3)
    fuso_horario = timezone(diferenca)
    now = now.astimezone(fuso_horario)
    return now
def now():
    return datetime.now()

def get_now_string():
    now = get_now()
    return now.strftime("%d-%m-%y%H-%I-%S")

def show_points(mensagem,di,df,points=None,box=None,print_m=True,ticket=None):

    points_str = ""
    if points is not None:

        for (x,y) in points:
            points_str += "{}:{};".format(x,y)


    #points_str = str.encode(points_str)
    #points_str = base64.b64encode(points_str)

    string = "[{time}][{ticket}][{runtime:8.5f}]"\
             "[{mens}][{points}][{box}]:".format(time=get_now(),
                                                ticket=ticket,
                                                mens=mensagem,
                                                points=points_str,
                                                box=box,
                                                runtime=(df-di).total_seconds())
    if print_m:
        print(string)

    logging.info(string)


def show_logging(mensagem,di,df,print_m=True,ticket=None):
    string = "[{time}][{ticket}][{runtime:8.5f}][{mens}]".format(time=get_now(),
                                                                                  ticket=ticket,
                                                                                  mens=mensagem,
                                                                                  runtime=(df-di).total_seconds())
    if print_m:
        print(string)
    logging.info(string)

def erro_logging(mensagem,print_m=True, ticket=None):
    string = "[{time}][{ticket}][{runtime:8.5f}][{mens}]".format(time=get_now(),runtime=0,
                                                                  ticket=ticket,mens=mensagem)
    if print_m:
        print(string)

    logging.error((string))
