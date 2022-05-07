import cv2
import mediapipe as mp
import numpy as np

class handDetector():
    def __init__(self,mode=False,max_num_hands=2,min_detection_confidence=0.5,min_track_confidence = 0.5):
        self.mode = mode
        self.max_num_hands= max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_track_confidence = min_track_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode,max_num_hands, min_detection_confidence, min_track_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,frame,draw=True):
        frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)
        return frame

    def getPosition(self,frame,handNum = 0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
        return lmList

    def fingersUp(self,frame):
        pos = self.getPosition(frame)
        fingers = [8,12,16,20]
        which_up = [0,0,0,0]
        for i in range(4):
            if (pos[fingers[i]][2] < pos[fingers[i]-2][2]):
                which_up[i] = 1
        return which_up



def main():
    #SPECIFY COLOR PALLETE 
    color_index = 0
    colors = [(0,0,255),(0,162,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(255,0,162),(255,0,255),(0,0,0)] # R O Y G C B Pu Pi Black

    #CREATE PAINT WINDOW
    paintWindow = np.zeros((720,1280,3),np.uint8)

    #GET WEBCAM
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    detector = handDetector(min_detection_confidence=0.8,min_track_confidence=0.65)

    xp,yp = 0,0 #Variables to know previous position of hand
    prev_state = 0 # 0 for selection mode 1 for drawing mode

    while True:
        _,frame = cap.read()
        frame = cv2.flip(frame,1)
        frame = detector.findHands(frame,draw=False)
        lmList = detector.getPosition(frame)

        #ADDING BLOCKS ON FRAME
        frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
        frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
        frame = cv2.rectangle(frame, (275,1), (370,65), colors[1], -1)
        frame = cv2.rectangle(frame, (390,1), (485,65), colors[2], -1)
        frame = cv2.rectangle(frame, (505,1), (600,65), colors[3], -1)
        frame = cv2.rectangle(frame, (620,1), (715,65), colors[4], -1)
        frame = cv2.rectangle(frame, (735,1), (830,65), colors[5], -1)
        frame = cv2.rectangle(frame, (850,1), (945,65), colors[6], -1)
        frame = cv2.rectangle(frame, (965,1), (1060,65), colors[7], -1)
        frame = cv2.rectangle(frame, (1080,1), (1175,65), (0,0,0), -1)

        cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ORANGE", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (405, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "CYAN", (650, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (755, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "PURPLE", (870, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "PINK", (995, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ERASER", (1100, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


        if len(lmList) != 0:
            which_up = detector.fingersUp(frame)
            pos_x,pos_y = lmList[8][1],lmList[8][2]
            radius = 5
            center = (pos_x,pos_y)
            #IF NOT IN ERASER MODE
            if color_index != 8:
                cv2.circle(frame,center,radius,(255,255,255),2)
            else:
                cv2.circle(frame,center,radius,(255,255,255),20)

            # CONDITION TO NOT DRAW WHEN TWO FINGERS ARE UP
            if(which_up[0] == 1 and which_up[1] == 1):
                prev_state = 0
                if center[1] <= 65:
                    if 40 <= center[0] <= 140: 
                        paintWindow[67:,:,:] = 0 # ALL CLEAR
                    elif 160 <= center[0] <= 255:
                        color_index = 0 # RED
                    elif 275 <= center[0] <= 370:
                        color_index = 1 # ORANGE
                    elif 390 <= center[0] <= 485:
                        color_index = 2 # YELLOW
                    elif 505 <= center[0] <= 600:
                        color_index = 3 # GREEN
                    elif 620 <= center[0] <= 715:
                        color_index = 4 # CYAN
                    elif 735 <= center[0] <= 830:
                        color_index = 5 # BLUE
                    elif 850 <= center[0] <= 945:
                        color_index = 6 # PURPLE
                    elif 965 <= center[0] <= 1060:
                        color_index = 7 # PINK
                    elif 1080 <= center[0] <= 1175:
                        color_index = 8 # ERASER

            #DRAWING MODE
            elif(which_up[0] == 1):
                #TO PREVENT LINE FROM JOINING AFTER LINE BREAK 
                if (xp == 0 and yp == 0) or prev_state != 1  :
                    (xp,yp) = center
                if color_index != 8:
                    cv2.line(frame,(xp,yp),center,colors[color_index],2) #CAN ADD THICKNESS LATER
                    cv2.line(paintWindow,(xp,yp),center,colors[color_index],2)
                else: 
                    cv2.line(frame,(xp,yp),center,colors[color_index], 5) #CAN ADD THICKNESS LATER
                    cv2.line(paintWindow,(xp,yp),center,colors[color_index],20)
                (xp,yp) = center
                prev_state = 1
                
            else:
                prev_state = 0

        #MERGING OF PAINTWINDOW AND FRAME TO MAKE BETTER UI
        paintGray = cv2.cvtColor(paintWindow,cv2.COLOR_BGR2GRAY)
        _,paintInv = cv2.threshold(paintGray,50,255,cv2.THRESH_BINARY_INV)
        paintInv = cv2.cvtColor(paintInv,cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame,paintInv)
        frame = cv2.bitwise_or(frame,paintWindow)

        cv2.imshow("Tracking", frame)
        #cv2.imshow("Paint", paintWindow)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()