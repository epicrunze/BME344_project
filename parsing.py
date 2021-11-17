import cv2
import numpy as np
from scipy.spatial import distance


img = None #create sample image in black and white
thresh = None #create sample image in black and white

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#find center of image and draw it (blue circle)
image_center = np.asarray(thresh.shape) / 2
image_center = tuple(image_center.astype('int32'))
cv2.circle(img, image_center, 3, (255, 100, 0), 2)

shapes = []
for contour in contours:
    # find center of each contour
    M = cv2.moments(contour)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])
    contour_center = (center_X, center_Y)

    # calculate distance to image_center
    distances_to_center = (distance.euclidean(image_center, contour_center))

    # save to a list of dictionaries
    shapes.append({'contour': contour, 'center': contour_center, 'distance_to_center': distances_to_center})

    # draw each contour (red)
    cv2.drawContours(img, [contour], 0, (0, 50, 255), 2)
    M = cv2.moments(contour)

    # draw center of contour (green)
    cv2.circle(img, contour_center, 3, (100, 255, 0), 2)

# sort the buildings
sorted_contours = sorted(shapes, key=lambda i: i['distance_to_center'])

# find contour of closest building to center and draw it (blue)
center_contour = sorted_contours[0]['contour']
cv2.drawContours(img, [center_contour], 0, (255, 0, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)

(x,y),radius = cv2.minEnclosingCircle(center_contour)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(img,center,radius,(0,255,0),2)
