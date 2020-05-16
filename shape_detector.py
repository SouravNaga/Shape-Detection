import numpy as np
import cv2

img = cv2.imread('3.png')
#imgresize=cv2.resize(img,(500,500))
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
#optional output vector ,containing information topology
#threshold value which is used to classify the pixel intensities in the gray scale images
#here set minimum threshold value is 240 and generally maximum threshold value is 255
#and the threshold value decided by fourth parameter cv2.COLOR_BINARY function
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#contours can be explained simply as a curve joining all the continuous points along the boundary within same color value
#the contours are a useful tool for shape analysis ans object detection and recognition
#in cv2.findContours() function first one is source image
#second one is- contour retrieval mode
#third one is- contour approximation method
#if you pass cv2.CHAIN_APPROX_NONE here in this method all the boundary points are stored
#cv2.CHAIN_APPROX_SIMPLE here in this method it removes all redundent points and compress the contour
#cv2.RTER_TREE  that retrives all the contours and reconstruct a full hierarchy of nested contours
#cv2.imshow("img", img)
for contour in contours:
    epsilon=0.01* cv2.arcLength(contour, True)
    #parameter specifying the approximation accuracy,this is the maximum distance between the original curve and approximation
    approx = cv2.approxPolyDP(contour, epsilon, True)
    #true if the approximated curve is closed
    #approximates a polygonal curve with the specified precision
    #cv.approxPolyDP(curve,epsilon,closed)
    cv2.drawContours(img, approx, -1, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]-5
    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 4:
        x1 ,y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if aspectRatio >= 0.90 and aspectRatio <= 1.10:
          cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        else:
          cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 6:
        cv2.putText(img, "Hexagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 10:
        cv2.putText(img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))    
    else:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))


cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
