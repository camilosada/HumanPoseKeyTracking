import cv2
import mediapipe as mp
import time
import math
import numpy as np 
 
class poseDetector():
 
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
 
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)
        self.jointAngles = {'right_elbow': (12,14,16),'left_elbow':(11,13,15),'right_knee': (24,26,28), 'left_knee': (23,25,27)}
 
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img
 
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
 
    def findAngle(self, img, ja, draw=True):
 
        # Get the landmarks
        x1, y1 = self.lmList[self.jointAngles[ja][0]][1:]
        x2, y2 = self.lmList[self.jointAngles[ja][1]][1:]
        x3, y3 = self.lmList[self.jointAngles[ja][2]][1:]

        v21_y, v21_x = y1 - y2, x1 - x2
        v23_y, v23_x = y3 - y2, x3 - x2
 
        # Calculate the Angle
        #angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle = math.degrees(math.acos( ( (v21_y*v23_y) + (v21_x*v23_x) ) /
                                        ( math.sqrt((v21_y**2) + (v21_x**2)) * math.sqrt((v23_y**2) + (v23_x**2)) ) ))
        # if angle < 0:
        #     angle += 180
 
        
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 48, y2 + 52),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def findAngularVelocity(self, x_time, y_angle):
        
        self.angularVel = []# don't do it with self, change with each angle
        dx_time = np.diff(x_time)
        dy_angle = np.diff(y_angle)
        self.angularVel = [i / j for i, j in zip(dy_angle, dx_time)]
        
        return self.angularVel

    def findAngularAcceleration(self, x_time, y_vel):
        
        self.angularAcc = [] # don't do it with self, change with each angle
        dx_time = np.diff(x_time[1:])
        #dx_time2 = np.diff(dx_time)
        dy_vel = np.diff(y_vel)
        
        self.angularAcc = [i / j for i, j in zip(dy_vel, dx_time)]
        
        return self.angularAcc

def main():
    cap = cv2.VideoCapture('walking1.mp4')#
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        if success == True:
            img = detector.findPose(img)
            lmList = detector.findPosition(img, draw=False)
            # if len(lmList) != 0:
            #     print(lmList[14])
            #     cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
    
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
    
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)
    
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.imshow("Image", img)

            if (cv2.waitKey(25) & 0xFF == ord('q')):
                break
        else:
            break
 
 
if __name__ == "__main__":
    main()