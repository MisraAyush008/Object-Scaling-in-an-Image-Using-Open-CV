from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

# Function to show array of images (intermediate results)
def show_images(images):
    for i, img in enumerate(images):
        cv2.imshow("image_" + str(i), img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#img_path = "real.jpg"
video_path = "C://Users//IDEAPAD//Desktop//minor viva 01//sample01.mp4"

# Video Processing
video = cv2.VideoCapture(video_path)

# We will calculate the reference object dimensions.
# We have considered the object at first frame as reference object
# So after calculating reference object dimensions we will break the loop
while(video.isOpened()):
    ret, frame = video.read()
    if(ret==False):
        break;
    frame = cv2.resize(frame, (640, 480))
    # Read image and preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # OBJECT SEGMNTATION
    #show_images([blur, edged])
    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)

    # Remove contours which are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]

    #cv2.drawContours(image, cnts, -1, (0,255,0), 3)

    #show_images([image, edged])
    #print(len(cnts))

    # Reference object dimensions
    # Here for reference I have used a 2cm x 2cm square
    if(len(cnts)==0):
        continue
    ref_object = cnts[0]
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    dist_in_cm = 2
    pixel_per_cm = dist_in_pixel/dist_in_cm
    break;# we break the loop at first iteration as we have calculated the dimensions of refernce objects.

# Now we have to calculate the dimensions of the moving object at each frame
while(video.isOpened()):
    ret, frame = video.read()
    if(ret==False):
        break;
    frame = cv2.resize(frame, (640, 480))
    # Read image and preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)    
    
    #show_images([blur, edged])
    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)

    # Remove contours which are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]

    if(len(cnts)==0):
        continue
        
    # Draw remaining contours
    for cnt in cnts:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        cv2.drawContours(frame, [box.astype("int")], -1, (0, 0, 255), 2)
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
        wid = euclidean(tl, tr)/pixel_per_cm
        ht = euclidean(br, bl)/pixel_per_cm
        cv2.putText(frame, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    show_images([frame])

video.release()
cv2.destroyAllWindows()
