import cv2 as cv
import mediapipe as mp
import time
import utils

#variables 
frame_counter = 0

#constants
FONTS= cv.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh
# camera object
camera = cv.VideoCapture(0)
# landmark detection funcion
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    #list[(x, y), (x,y)...]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]
    # returning the list of tuples for each landmarks
    return mesh_coord

with map_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:

    # starting time
    start_time = time.time()

    while True:

        frame_counter += 1
        ret, frame = camera.read() # getting the frame from the camera
        if not ret:
            break # no more frames break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, True)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in FACE_OVAL], utils.GRAY, opacity = 0.6)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYE], utils.GREEN, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYE], utils.GREEN, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYEBROW], utils.ORANGE, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYEBROW], utils.ORANGE, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LIPS], utils.BLACK, opacity=0.3 )
        # calculating frame per second (FPS)
        end_time = time.time() - start_time
        fps = frame_counter/end_time
        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (20, 50), bgOpacity=0.9, textThickness=2)

        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == ord('q'): # wait until 'q' key is pressed
            break
    cv.destroyAllWindows()
    camera.release()