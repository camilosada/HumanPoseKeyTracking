import cv2
import numpy as np
import time
import jointPose as jp
import psutil
import matplotlib.pyplot as plt
 
cap = cv2.VideoCapture('arm_flexion.mp4')
fps_video= 30
plt.rcParams['animation.html'] = 'jshtml'
fig = plt.figure()
ax = fig.add_subplot(111)
fig.show()



iteration = 0
x_time, y_ang = [], []

detector = jp.poseDetector()
joints = ('right_elbow','left_elbow','right_knee','left_knee')
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()

    if success == True:
        img = cv2.resize(img, (1280, 720))
    
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        # print(lmList)
        if len(lmList) != 0:
            
            angle = detector.findAngle(img, joints[0])
            
            #angle = detector.findAngle(img, 11, 13, 15,False)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))
            # print(angle, per)
    
            # Check for the dumbbell curls
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0
            #print(count)
    
            # Draw Bar
            # cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            # cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            # cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
            #             color, 4)
    
            # Draw Curl Count
            # cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            # cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
            #             (255, 0, 0), 25)
            x_time.append(iteration*(1/fps_video))
            y_ang.append(angle)
            ax.plot(x_time, y_ang, color='b')
            fig.canvas.draw()
            ax.set_xlim( right=iteration*(1/fps_video)+(1/fps_video)*2)
            iteration += 1        
    
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5)
    
        cv2.imshow("Image", img)

        if (cv2.waitKey(25) & 0xFF == ord('q')):
            break
    else:
        break

angVel = detector.findAngularVelocity(x_time, y_ang)
angAcc = detector.findAngularAcceleration(x_time, angVel)
print(len(x_time),len(y_ang),len(angVel),len(angAcc))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig2.show()
ax2.plot(x_time[1:], angVel, color='r')
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
fig3.show()
ax3.plot(x_time[2:], angAcc, color='g')

while True:
    if (cv2.waitKey(25) & 0xFF == ord('q')):
        break
print('nn')