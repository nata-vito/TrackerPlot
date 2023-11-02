import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp

class Headpose:
    def __init__(self):
        # MediaPipe Config
        self.mp_face_mesh                           = mp.solutions.face_mesh
        self.face_mesh                              = self.mp_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
        self.mp_drawing                             = mp.solutions.drawing_utils
        self.drawing_spec                           = self.mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)
        
        # Camera config
        self.cap                                    = cv2.VideoCapture(0)
        self.width_cam, self.height_cam             = 640, 480
        self.frame_R                                = 100
        self.cap.set(3, self.width_cam)
        self.cap.set(4, self.height_cam)
        
        # Interpolation config
        self.smoothening                            = 8
        self.pTime                                  = 0
        self.plocX, self.plocY                      = 0, 0          # Previous locations of x and y
        self.clocX, self.clocY                      = 0, 0          # Current locations of x and y
        self.width_screen, self.height_screen       = 1920, 1080
        
        self.data                                   = {'timestamp': [],
                                                       'x': [],
                                                       'y': []}
        self.df                                     = None
        self.path_to_csv                            = '/home/nata-brain/camera_ws/src/EyeHeadTrack/vision/dataset'
        
        
    # Preprocessing the image for the model
    def preProcessImage(self, image):
        # Flip the image horizontally for a later selfie-view display and also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        # To improve performance
        image.flags.writeable = False
        
        # Get the result
        results = self.face_mesh.process(image)
        
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image, results
    
    
    # Get X, Y and Z coordinates
    def getCoords(self, face_2d, face_3d):
        focal_length        = 1 * self.width_cam
        dist_matrix         = np.zeros((4, 1), dtype=np.float64)
        
        # Convert it to the NumPy array
        face_2d     = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d     = np.array(face_3d, dtype=np.float64)
        cam_matrix  = np.array([[focal_length, 0, self.height_cam / 2],
                                [0, focal_length, self.width_cam / 2],
                                [0, 0, 1]])

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the rotation degrees
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        
        return x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix
        
        
    # Get the text output
    def getInfo(self, image, x, y, z, p1, p2):
        # See where the user's head tilting
        text = ''
        if y < -10:
            text = "Looking Left"
        elif y > 10:
            text = "Looking Right"
        elif x < -10:
            text = "Looking Down"
        elif x > 10:
            text = "Looking Up"
        else:
            text = "Forward"
        
        # Add the text an line on the image
        cv2.line(image, p1, p2, (255, 0, 0), 3)
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    
    def saveRealPos(self, op):
        # Effective position of the point obtained by interpolation
        if op == 0:
            self.data['timestamp'].append(time.time())              # Timestamp. To convert, please use datetime lib -> (datetime.fromtimestamp(timestamp)
            self.data['x'].append(self.width_screen - self.clocX)   # X
            self.data['y'].append(self.clocY)                       # Y
            
        # Save df in csv file    
        elif op == 1:
            self.df = pd.DataFrame(self.data)
            self.df.to_csv(f'{self.path_to_csv}/gaze_points.csv')
   
   
    # Interpolation between camera points and screen points
    def interpolation(self, image, point):
        # Convert Coordinates as our cv window is 640*480 but my screen is full HD so have to convert it accordingly
        x = np.interp(point[0], (self.frame_R, self.width_cam - self.frame_R), (0, self.width_screen))      # converting x coordinates
        y = np.interp(point[1], (self.frame_R, self.height_cam - self.frame_R), (0, self.height_screen))    # converting y coordinates
        
        # Smoothen Values avoid fluctuations
        self.clocX = self.plocX + (x - self.plocX) / self.smoothening
        self.clocY = self.plocY + (y - self.plocY) / self.smoothening
        
        # Put circle in screen
        cv2.circle(image, (point[0], point[1]), 15, (255, 0, 255), cv2.FILLED)      # circle shows that we are in moving mode
        self.plocX, self.plocY = self.clocX, self.clocY
        
        self.saveRealPos(op = 0)


    # Convert pixels to cm
    def pixelToCm(self, pixels = 0):
        self.centi = round(((pixels * 2.54)/96), 2)


    # Convert cm to pixels
    def cmToPixels(self, centi = 0):
        self.pixels = round(((centi * 96)/2.54), 2)
        
        
    # Prediction and results extraction
    def inference(self, image, results, start):
        self.height_cam, self.width_cam, img_c  = image.shape
        face_3d                                 = []
        face_2d                                 = []
        p1, p2                                  = '', ''
        
        # Process the result to 2d an 3d face
        if results.multi_face_landmarks:
            # Rectangle representing screen orientation self.df = pd.DataFrame(self.data)
        
            cv2.rectangle(image, (self.frame_R, self.frame_R), (self.width_cam - self.frame_R, self.height_cam - self.frame_R), (255, 0, 255), 2)
            
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        
                        if idx == 1:
                            nose_2d = (lm.x * self.width_cam, lm.y * self.height_cam)
                            nose_3d = (lm.x * self.width_cam, lm.y * self.height_cam, lm.z * 3000)
                            
                        # Get the 2D Coordinates
                        x, y = int(lm.x * self.width_cam), int(lm.y * self.height_cam)
                        
                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])
                                
                x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix = self.getCoords(face_2d, face_3d)
               
                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))                                 # Point 1
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))              # Point 2
                
                self.interpolation(image, p2)
                self.getInfo(image, x, y, z, p1, p2)
            
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime    
            cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            self.mp_drawing.draw_landmarks(
                        image = image,
                        landmark_list = face_landmarks,
                        connections = self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec = self.drawing_spec,# Get the 2D Coordinates
                        connection_drawing_spec = self.drawing_spec)
                
        return image, p1, p2        
                
                
    # Runs the entire algorithm    
    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            start = time.time()

            image, results = self.preProcessImage(image)

            image, p1, p2 = self.inference(image, results, start)

            cv2.imshow('Head Pose Estimation', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        self.cap.release()
        self.saveRealPos(op = 1)
        
        
if __name__ == '__main__':
    pose = Headpose()
    pose.run()
    
    