import imutils
import numpy as np
import cv2
import math
import time
import random

background = None
win1 = 'Hand Tracking'
win2 = 'Handy little game'
cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
cv2.moveWindow(win1, 500, 0)
cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
cv2.moveWindow(win2, 0,0)
# region of interest (ROI) coordinates
top, right, bottom, left = 80, 400, 315, 640

def idxToPoint(contour, array):
    return np.array([contour[x[0]][0] for x in array])

def run_avg(image, imageWeight):
    global background
    if background is None:
        background = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, background, imageWeight)

def segment(image, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    thresholded = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)
    contours = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    if len(contours) != 0:
        segmented = max(contours, key=cv2.contourArea)    
        return (thresholded, segmented) 

def contourCenter(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    return cX, cY

def ptdistance(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def getSimplifiedHull(contour):
    hullwIdx = cv2.convexHull(contour, returnPoints=False)
    maxDist = 15
    clusters = [[]]
    clusternum = 0
    for i in range(len(hullwIdx)):
        if ptdistance(contour[hullwIdx[i-1, 0]][0], contour[hullwIdx[i, 0]][0]) < maxDist:
            clusters[clusternum].append(hullwIdx[i])
        else:
            clusters.append([hullwIdx[i]])
            clusternum += 1
    if ptdistance(contour[hullwIdx[-1, 0]][0], contour[hullwIdx[0,0]][0]) < maxDist:
        clusters[0] += clusters[-1]
        clusters.pop(-1)
    simplifiedHull = []
    for cluster in clusters:
        if cluster != []:
            #cluster.sort(key=lambda x: (contour[x[0]][0,0], contour[x[0]][0,1]))
            simplifiedHull.append(cluster[len(cluster)//2])
    return np.array(simplifiedHull)


def getFingerTips(contour, hull=None):
    if hull is None:
        hull = getSimplifiedHull(contour)
    try:
        defects = cv2.convexityDefects(contour, hull)
    except:
        print('error')
        return
    if defects is None:
        print('kosong')
        return

    edges = {}
    for defect in defects:
        cv2.circle(clone, tuple(contour[defect[0,2], 0]  + (right, top)), 3, (255,0,255))
        if defect[0,0] in edges:
            edges[defect[0,0]].append(defect[0,2])
        else:
            edges[defect[0,0]] = [defect[0,2]]

        if defect[0,1] in edges:
            edges[defect[0,1]].append(defect[0,2])
        else:
            edges[defect[0,1]] = [defect[0,2]]
    fingertips = []
    for key in edges:
        if len(edges[key]) == 2:
            point0 = contour[key,0]
            point1 = contour[edges[key][0], 0]
            point2 = contour[edges[key][1], 0]
            vector1 = (point0 - point1)/ ptdistance(point0, point1)
            vector2 = (point0 - point2) / ptdistance(point0, point2)
            sudut = np.degrees(np.arccos(np.dot(vector1, vector2)))
            if sudut < 70:
                fingertips.append([key])
    print(len(edges), len(fingertips))
    fingertips = idxToPoint(contour, fingertips)
    
    for point in fingertips:
        cv2.circle(clone, tuple(point + (right, top)), 7, (0,0,255))
    return fingertips

def getFurthestPoint(origin, points):
    maxdist = 0
    maxpoint = origin
    for point in points:
        dist = ptdistance(origin, point)
        if dist > maxdist:
            maxdist = dist
            maxpoint = point
    return maxpoint

def renderSprite(mask, BG, color, pos, size=None, rotation=None):
    if size is not None:
        mask = imutils.resize(mask, height=size)
    if rotation is not None:
        col, row = mask.shape
        mask = cv2.warpAffine(mask, rotation, (col, row))

    shapes = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    shape = sorted(shapes, key=lambda x: cv2.moments(x)['m00'])[-1]
    cv2.drawContours(BG, [shape], 0, color, -1, offset=pos)
    if len(shapes) > 1:
        cv2.drawContours(BG, shapes[:-1], -1, (0,0,0), -1, offset=pos)
    

#fungsi utama
if __name__ == "__main__":
    ######## Handtracking Setup ########
    aWeight = 0.5
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 80, 400, 315, 640

    ##### Game Setup #####
    gameBG = cv2.imread('background.jpeg', 1)
    playermask = cv2.imread('playermask.png', 0)
    enemymask = cv2.imread('enemy.png', 0)
    shotmask = cv2.imread('shot.png', 0)
    playermask = imutils.resize(playermask, height=40)
    gameBG = imutils.resize(gameBG, height=500)    
    gameWidth = 822
    ctrlScale = (left - right)
    enemies = [] #list of enemy coordinates and direction 
    shots = [] #list of coordinate and direction vector
    shotSpeed = 5
    enemySpeed = 3

    score = 0
    life = 5

    num_frames = 0

    while(True):
        # init frame time
        looptime = time.clock()

        #inisialisasi untuk deteksi
        origin = None
        fingers = None

        grabbed, frame = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2RGBA)
        gray = cv2.GaussianBlur(roi, (7, 7), 0)
        
        if num_frames < 30:
            run_avg(gray, aWeight)
            num_frames += 1
            continue
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                hullinIdx = getSimplifiedHull(segmented)
                hull = cv2.convexHull(segmented)
                origin = contourCenter(segmented)
                fingers = getFingerTips(segmented, hullinIdx)
                if origin is not None and fingers is not None:
                    pointer = tuple(getFurthestPoint(origin, fingers))
                    pointerPoint = pointer[0] + right, pointer[1] + top
                    originPoint = origin[0]+right, origin[1]+top
                    cv2.circle(clone, originPoint, 7, (0,0,255))
                    cv2.line(clone, originPoint, pointerPoint, (0,0,255))
                
                cv2.drawContours(clone, [segmented + (right, top), hull + (right, top)], -1, (255, 0, 0))
            num_frames += 1

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        cv2.imshow(win1, clone)

        
        ####### Game Update ######
        gameBG = cv2.imread('background.jpeg', 1)
        playermask = cv2.imread('playermask.png', 0)
        playermask = imutils.resize(playermask, height=40)
        gameBG = imutils.resize(gameBG, height=500)  
        player = cv2.findContours(playermask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1][0]
        
        rotMatrix = None
        degree = 0
        if origin is not None:
            playerXpos = origin[0]*gameWidth//ctrlScale
            if fingers is not None:
                tan = (pointer[0]-origin[0])/((pointer[1]-origin[1])+0.01)
                degree = math.degrees(math.atan(tan))
                rotMatrix = cv2.getRotationMatrix2D((20, 20), degree, 1)
        else:
            playerXpos = gameWidth//2

        # render player
        renderSprite(playermask, gameBG, (0,0,255), (playerXpos, 450), rotation=rotMatrix)
        
        #create enemy randomly
        if random.random() < 0.008:
            enemies.append([[random.random()*gameWidth,10], [random.randint(-enemySpeed, enemySpeed), 2]])

        #player shoot periodically, tergantung jumlah jari
        if fingers is not None:
            if len(fingers) > 3 and num_frames % 60 == 0:
                for i in range(-60,61,30):
                    shots.append([[playerXpos+20, 450], [-shotSpeed*math.sin(math.radians(i)), -shotSpeed*math.cos(math.radians(i))]])
            
            elif len(fingers) > 0 and len(fingers) <= 3 and num_frames % 15 == 0:
                shots.append([[playerXpos+20, 450], [-shotSpeed*math.sin(math.radians(degree)), -shotSpeed*math.cos(math.radians(degree))]])
        
            elif len(fingers) == 0 and num_frames % 40 == 0:
                shots.append([[playerXpos+20, 450], [2*shotSpeed, 0]])
                shots.append([[playerXpos+20, 450], [-2*shotSpeed, 0]])
        
        # delete shot and enemy that hit
        for enemy in enemies:
            hit = False
            for shot in shots:
                if ptdistance([enemy[0][0]+20, enemy[0][1]+20], [shot[0][0]+10,shot[0][1]+10]) <= 30:
                    shots.remove(shot)
                    hit = True
                    break
            if hit:
                score += 1
                enemies.remove(enemy)

        # move other shot and enemy
        for enemy in enemies:
            enemy[0][0] += enemy[1][0]
            enemy[0][1] += enemy[1][1]
            if enemy[0][0] < 0 or enemy[0][0] >= gameWidth:
                enemy[1][0] *= -1
            elif enemy[0][1] < 0 or enemy[0][1] >= 500:
                life -= 1
                enemies.remove(enemy)
            else:
                renderSprite(enemymask, gameBG, (100,100,100), (int(enemy[0][0]), int(enemy[0][1])), size=40)

        for shot in shots:
            shot[0][0] += shot[1][0]
            shot[0][1] += shot[1][1]
            if shot[0][0] < 0 or shot[0][0] >= gameWidth or shot[0][1] < 0 or shot[0][1] >= 500:
                shots.remove(shot)
            else:
                renderSprite(shotmask, gameBG, (0, 255, 255), (int(shot[0][0]), int(shot[0][1])), size=20)

        # display score and life point
        cv2.putText(gameBG, "Life: "+str(life), (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
        cv2.putText(gameBG, "Score: "+str(score), (10,60), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
        cv2.imshow(win2, gameBG)

        ###### frame rate and loop control ######
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"): # q untuk quit
            break
        elif keypress == ord('r'): # r untuk recalibrate
            background = None
            num_frames = 0

        if life == 0:
            cv2.putText(gameBG,"Game Over", (200,250), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255))
            cv2.putText(gameBG,"Press Enter to play again", (200,350), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
            cv2.putText(gameBG,"or any other button to quit", (200,400), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
            cv2.imshow(win2, gameBG)
            keypress = cv2.waitKey(0)
            if keypress == ord('\r'):
                enemies = []
                shots = []
                score = 0
                life = 5
            else:
                break

        while time.clock()-looptime < 0.04:
            pass

# free up memory
camera.release()
cv2.destroyAllWindows()

