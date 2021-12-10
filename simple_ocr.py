#
# You can modify this files
#
import numpy as np
import cv2
import math
import pytesseract


# Tesseract-ocr source
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

#variable
thetaErr = math.pi / 6
debug = False
rhoErr = 20
report = False

import random

class HoadonOCR:

    def __init__(self):
        # Init parameters, load model here
        self.model = None
        self.labels = ['highlands', 'starbucks', 'phuclong', 'others']
        self.result = 'dcm'

    # TODO: implement find label
    def find_label(self, img):
        self.result = OutPut(img)
        # print(self.result)
        return self.result


def getDistance(a,b):
    return np.linalg.norm(a - b)

def getOrderPoints(points):
    '''
    Divide points into 2 part: left and right
    sort the points based on x-coordinates
    '''
    sortArgX = np.argsort(points[:,0])
    left = np.array([points[x] for x in sortArgX[0:2]])
    right = np.array([points[x] for x in sortArgX[2:4]])
    #point with bigger y is bottomLeft and vice versa
    bottomLeft = left[np.argmax(left[:,1])]
    topLeft = left[np.argmin(left[:,1])]
    #point that is farther from the topLeft is bottomRight
    if getDistance(topLeft, right[0]) > getDistance(topLeft, right[1]):
        bottomRight = right[0]
        topRight = right[1]
    else:
        bottomRight = right[1]
        topRight = right[0]
    return (topLeft, topRight, bottomRight, bottomLeft)

#detect whether string contains label or not
def contain (result, label):
    if (result.find(label) != -1): return label
    elif (label == 'Starbuck' and result.find('Store') != -1): return 'starbucks'
    elif (label == 'Starbuck' and result.find('Stare') != -1):
        return 'starbucks'
    elif (label == 'Starbuck' and result.find('store') != -1):
        return 'starbucks'
    else: return 'others'

def getParallel(line, point):
    rho, theta = line
    return getRho(theta)(point), theta

#Geting `rho` of a line that goes through `point` with angle `theta`
def getRho(theta):
    def result(point):
        #point is of the form (x, y)
        return point[0] * math.cos(theta) + point[1] * math.sin(theta)
    return result

def getIntersection(line1, line2):
    #lines are of the form (rho, theta)
    if debug:
        print(line1)
        print(line2)
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([[math.cos(theta1), math.sin(theta1)],
                [math.cos(theta2), math.sin(theta2)]])
    B = np.array([rho1, rho2])
    if debug:
        print(A)
        print(B)
    #return form: np.array([x, y]), may raise exception
    result = np.linalg.solve(A, B)
    return result

def getBoundaryIntersections(line, img):
    rho = line[0]
    theta = line[1]
    if theta >= math.pi / 2:
        newTheta = theta - math.pi / 2
    else:
        newTheta = theta + math.pi / 2
    height = img.shape[0]
    width = img.shape[1]
    leftBound = (0, 0)
    rightBound = (width, 0)
    topBound = (0, math.pi / 2)
    bottomBound = (height, math.pi / 2)
    bounds = (leftBound, rightBound, topBound, bottomBound)
    intersections = list()
    for bound in bounds:
        try:
            intersection = getIntersection(line, bound)
        except np.linalg.linalg.LinAlgError:
            continue
        else:
            intersections.append(intersection)
    rhos = [getRho(newTheta)(point) for point in intersections]
    intersections = np.array(intersections)
    numPoints = len(intersections)
    if numPoints == 4:
        return list(intersections[np.argsort(rhos)][1:3])
    elif numPoints == 2:
        return list(intersections)
    else:
        raise Exception("Error in GetBoundaryIntersections: Not enough points")

def checkSimilarRho(line, correctLine, img, rhoErr = 20):
    rho, theta = line
    intersections = getBoundaryIntersections(correctLine, img)
    rhos = [getRho(theta)(intersection) for intersection in intersections]
    if rho < max(rhos) + rhoErr and rho > min(rhos) - rhoErr:
        return True
    return False

def checkSimilarAngle(theta1, theta2):
    if theta1 <= thetaErr / 2 and theta2 >= math.pi - thetaErr / 2:
        return True  # , (theta1, theta2)
    elif theta2 <= thetaErr / 2 and theta1 >= math.pi - thetaErr / 2:
        return True  # , (theta2, theta1)
    elif abs(theta1 - theta2) < thetaErr:
        return True  # , (theta1, theta2)
    else:
        return False  # , None

#get missing edges in case only three edges are detected
#return [lonely edges, pair edges]

def getMissingEdges(correctLines):
    if checkSimilarAngle(correctLines[0][1], correctLines[1][1]):
        return [correctLines[2], correctLines[0], correctLines[1]]
    elif checkSimilarAngle(correctLines[0][1], correctLines[2][1]):
        return [correctLines[1], correctLines[0], correctLines[2]]
    else:
        return correctLines


def label_(img):
    test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    testText = pytesseract.image_to_string(test)
    testText = testText.replace(' ', '')
    # print(testText)

    labelList = ['HIGHLANDS', 'PHUCLONG', 'Starbuck']
    countRight = 0
    labelResult = 'others'

    for labelItem in labelList:
        labelResult = contain(testText, labelItem).lower()
        if (labelResult == labelItem.lower()):
            break

    return labelResult

def OutPut (img):
#Resize Imgage

    width = img.shape[1]
    height = img.shape[0]
    ratio = 500 / width
    if width > 700:
        ratio = 500 / width
        resized = cv2.resize(img, None, fx = ratio, fy = ratio, interpolation = cv2.INTER_LINEAR)
        img1 = resized
    else:
        img1 = img
    width_0 = img1.shape[1]
    height_0 = img1.shape[0]

    #Pre-processing image

    img_padded = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])

    #Gray_img

    imgray = cv2.cvtColor(img_padded,cv2.COLOR_BGR2GRAY)

    #Candy edge detection

    edges = cv2.Canny(imgray, 70, 150)

    #Gaussian filter

    gauss = cv2.GaussianBlur(edges,(3,3),0)

    #Find contour

    contours, hierarchy = cv2.findContours(gauss,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    #Find largest contour

    largest_cont = sorted(contours, key = cv2.contourArea, reverse = True)[1:2]

    #Binary image largest contour

    black = np.zeros((height_0, width_0), "uint8")

    draw = cv2.drawContours(black, largest_cont, -1, (255,255,255), 1)

    lines = cv2.HoughLines(black,1,np.pi/180, 50)
    blackHough = np.zeros((int(height * ratio), int(width * ratio)), "uint8")
    diagonal = math.sqrt(height ** 2 + width ** 2)
    lines0 = lines[:,0,:]
    for line in lines0:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + math.ceil(diagonal) * (-b))
        y1 = int(y0 + math.ceil(diagonal) * a)
        x2 = int(x0 - math.ceil(diagonal) * (-b))
        y2 = int(y0 - math.ceil(diagonal) * a)
        cv2.line(blackHough, (x1, y1), (x2, y2), (255, 255, 255), 1)

    #Find 4 different lines in hough space
    correctLines = list()
    for line in lines:
        if len(correctLines) == 4:
            break
        rho, theta = line[0]
        isNew = True
        numSimilar = 0
        for l in correctLines:
            correctTheta = l[1]
            if checkSimilarAngle(theta, correctTheta):
                numSimilar += 1
                if numSimilar == 2:
                    isNew = False
                    break
                if checkSimilarRho(line[0], l, img, rhoErr):
                    isNew = False
                    break
        if isNew:
            correctLines.append([rho, theta])
        else:
            continue

    for line in correctLines:
            line[0] = line[0] / ratio
    numLines = len(correctLines)
    if numLines < 3:
        labelResult = label_(img)
        if (labelResult == "others"):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            labelResult = label_(img)
        if (labelResult == "others"):
            img = cv2.rotate(img, cv2.ROTATE_180)
            labelResult = label_(img)
        if (labelResult == "others"):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            labelResult = label_(img)
        return labelResult

    elif numLines == 3:
        correctLines = getMissingEdges(correctLines)
        rho, theta = correctLines[0]
        intersections = getBoundaryIntersections(correctLines[1], img) + getBoundaryIntersections(correctLines[2],
                                                                                                        img)
        intersections.sort(key=getRho(theta))
        if abs(getRho(theta)(intersections[1]) - rho) > abs(getRho(theta)(intersections[2]) - rho):
            newLine = getParallel(correctLines[0], intersections[1])
        else:
            newLine = getParallel(correctLines[0], intersections[2])
        correctLines.append(newLine)

    corners = list()
    for i in range(4):
        for j in range(i + 1, 4):
            if checkSimilarAngle(correctLines[i][1], correctLines[j][1]):
                    continue
            try:
                intersection = getIntersection(correctLines[i], correctLines[j])
            except np.linalg.linalg.LinAlgError:
                continue
            else:
                corners.append(intersection)
    if len(corners) != 4:
        labelResult = label_(img)
        if (labelResult == "others"):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            labelResult = label_(img)
        if (labelResult == "others"):
            img = cv2.rotate(img, cv2.ROTATE_180)
            labelResult = label_(img)
        if (labelResult == "others"):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            labelResult = label_(img)
        return labelResult
    topLeft, topRight, bottomRight, bottomLeft = getOrderPoints(np.array(corners, dtype="float32"))
    oldCorners = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype="float32")
    # Compute new width and height
    newWidth = max(getDistance(topLeft, topRight), getDistance(bottomLeft, bottomRight))
    newHeight = max(getDistance(topLeft, bottomLeft), getDistance(topRight, bottomRight))
    # Compute 4 new corners
    newCorners = np.array([
        [0, 0],
        [newWidth - 1, 0],
        [newWidth - 1, newHeight - 1],
        [0, newHeight - 1]], dtype="float32")

    #Compute transformation matrix
    transMat = cv2.getPerspectiveTransform(oldCorners, newCorners)
    #Transform
    resultImage = cv2.warpPerspective(img, transMat, (int(newWidth), int(newHeight)))
    if (newWidth > newHeight):
        resultImage = cv2.rotate(resultImage,cv2.ROTATE_90_COUNTERCLOCKWISE)
    #resized_result = cv2.resize(resultImage, None, fx = ratio, fy = ratio, interpolation = cv2.INTER_LINEAR)
    labelResult = label_(resultImage)

    if(labelResult =="others"):
        resultImage = cv2.rotate(resultImage,cv2.ROTATE_180)
        labelResult = label_(resultImage)
    if (labelResult=="others"):
        labelResult = label_(img)
    if(labelResult=="others"):
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        labelResult = label_(img)
    if (labelResult == "others"):
        img = cv2.rotate(img, cv2.ROTATE_180)
        labelResult = label_(img)
    if (labelResult == "others"):
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        labelResult = label_(img)
    return labelResult

