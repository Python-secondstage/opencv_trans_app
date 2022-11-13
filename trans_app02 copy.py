import face_recognition
import cv2
import numpy as np
original_face_img = "images/man.png"


def find_face_part(face_part_name):
    face_image = face_recognition.load_image_file(original_face_img)
    face_landmarks_list = face_recognition.face_landmarks(face_image)
    return face_landmarks_list[0][face_part_name]

face_part_name = "chin"

face_part_name_landmarks = find_face_part(face_part_name) # 座標
face_img = cv2.imread(original_face_img)
# for coordinate in find_face_part(face_part_name):
#     cv2.drawMarker(
#         face_img,
#         coordinate,
#         color=(255, 0, 0),
#         markerType=cv2.MARKER_CROSS,
#         thickness=1,
#     )



show_img(face_img)
cv2.fillConvexPoly(
    face_img,
    np.array(find_face_part(face_part_name)),
    # color=(0, 238, 120)
    color=(0, 120, 238),
)

cv2.inshow("Image", face_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


