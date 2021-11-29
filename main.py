import cv2
import numpy as np
from scipy.spatial import distance
import pandas as pd
import time


def main():
    x_locs = []
    y_locs = []
    radii = []
    t = []


    cap = cv2.VideoCapture('data\FB_Eyes_Closed_Trim.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"fps of video is {fps}")

    frame_counter = 0

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # convert the image to grayscale format
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # apply binary thresholding
            ret, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
            # visualize the binary image
            thresh = ((thresh * -1) + 255).astype(np.uint8)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_center = np.asarray(thresh.shape)[::-1] / 2
            image_center = tuple(image_center.astype('int32'))
            
            shapes = []
            for contour in contours:
                # find center of each contour
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    center_X = int(M["m10"] / M["m00"])
                    center_Y = int(M["m01"] / M["m00"])
                    contour_center = (center_X, center_Y)

                    # calculate distance to image_center
                    distances_to_center = (distance.euclidean(image_center, contour_center))

                    # save to a list of dictionaries
                    shapes.append({'contour': contour, 'center': contour_center, 'distance_to_center': distances_to_center})

                    # draw each contour (red)
                    # cv2.drawContours(frame, [contour], 0, (0, 50, 255), 2)
                    M = cv2.moments(contour)

                    # draw center of contour (green)
                    # cv2.circle(frame, contour_center, 3, (100, 255, 0), 2)

            # sort the contours
            sorted_contours = sorted(shapes, key=lambda i: i['distance_to_center'])

            # find closest contour to center and draw it (blue)
            
            for contour in sorted_contours:
                (x,y), radius = cv2.minEnclosingCircle(contour['contour'])
                if radius > 50:
                    center_contour = contour['contour']
                    break 

            
            (x,y), radius = cv2.minEnclosingCircle(center_contour)
            radius = int(radius)

            x_locs.append(x)
            y_locs.append(y)
            radii.append(radius)

            # image_copy1 = frame.copy()
            # cv2.circle(frame, image_center, 3, (255, 100, 0), 2)
            # cv2.drawContours(frame, [center_contour], 0, (255, 0, 0), 2)
            # cv2.drawContours(image_copy1, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
            # # see the results
            # cv2.imshow('Simple approximation', image_copy1)
            # cv2.waitKey(0)
            # cv2.imwrite('contours_simple_image1.jpg', image_copy1)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else: 
            break
        
        t.append(frame_counter * (1/fps))

        frame_counter += 1

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    df = pd.DataFrame({
        "time": t,
        "x_loc": x_locs,
        "y_loc": y_locs,
        "radius": radii,
    })

    df.to_csv("output.csv", index=False)

if __name__ == "__main__":

    start = time.time()
    main()
    end = time.time()
    print(f"time taken to process video: {end - start}")




