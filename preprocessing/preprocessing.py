import cv2
import os
import copy
from OpenSeeFace.tracker import Tracker

class Preprocessor:
    def __init__(self, model_type=3, detection_threshold=0.6, threshold=None, 
    max_faces=1, discard_after=5, scan_every=3, bbox_growth=0.0, max_threads=4, silent=False, 
    model_dir=None, no_gaze=False, use_retinaface=False, max_feature_updates=0, static_model=False, 
    feature_level=2, try_hard=False):

        self.tracker = Tracker(width=224, height=224, model_type=model_type, detection_threshold=detection_threshold, threshold=threshold, 
        max_faces=max_faces, discard_after=discard_after, scan_every=scan_every, bbox_growth=bbox_growth, max_threads=max_threads, silent=silent,
        model_dir=model_dir, no_gaze=no_gaze, use_retinaface=use_retinaface, max_feature_updates=max_feature_updates, static_model=static_model,
        feature_level=feature_level, try_hard=try_hard)

        self.save_mouth_image = False
        self.dir_save = ""
        self.pad = int
        self.mout_crop_threshold = float
    
    def save_mouth_crop(self, crop_mouth_image, face_id, frame_id):
        if self.save_mouth_image:
            path_dir = os.path.join(self.dir_save, str(face_id))
            # check path exists
            if not os.path.exists(path_dir):
                # create folder
                os.makedirs(path_dir)
            cv2.imwrite(os.path.join(path_dir, f'{frame_id}.jpg'), crop_mouth_image)
            print(f'Face {face_id} - Frame {frame_id} saved successfully')
    
    def crop_mouth(self, frame, frame_id):
        # predict and landmarks face
        faces = self.tracker.predict(frame)

        # crop mouth
        for face in faces:
            face = copy.copy(face)
            if face.success:
                if face.conf >= self.mouth_crop_threshold:
                    # get mouth landmarks
                    for pt_num, (x,y,c) in enumerate(face.lms[48:66]):
                        # get top left and bottom right
                        if pt_num == 0:
                            x_min = x
                            x_max = x
                            y_min = y
                            y_max = y
                        else:
                            if x < x_min:
                                x_min = x
                            if x > x_max:
                                x_max = x
                            if y < y_min:
                                y_min = y
                            if y > y_max:
                                y_max = y
                    x_min = int(x_min - self.pad)
                    x_max = int(x_max + self.pad)
                    y_min = int(y_min - self.pad)
                    y_max = int(y_max + self.pad)
                    # crop mouth
                    crop_mouth_image = frame[x_min:x_max, y_min:y_max]
                    # save crop mouth
                    self.save_mouth_crop(crop_mouth_image=crop_mouth_image, face_id = face.id, frame_id = frame_id)
    
    def draw_frame(self, frame, height, width):
        faces = self.tracker.predict(frame)
        for face in faces:
            if face.success:
                for pt_num, (x,y,c) in enumerate(face.lms):
                    x = int(x + 0.5)
                    y = int(y + 0.5)
                
                    frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,0))
                    color = (0, 255, 0)
                    if pt_num >= 66:
                        color = (255, 255, 0)
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = color
                    x += 1
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = color
                    y += 1
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = color
                    x -= 1
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = color
        return frame
                    
    def preprocess(self, video_path, pos_mes = 100, mouth_crop_threshold = 0.0, pad = 10, save = True, dir_save = "", show = True):

        self.mouth_crop_threshold = mouth_crop_threshold
        self.dir_save = dir_save
        self.pad = pad
        self.save_mouth_image = save

        # open video
        cap = cv2.VideoCapture(video_path)
        # get video info
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_rate = cap.get(5)
        frame_count = int(cap.get(7))
        # get video name
        video_name = os.path.basename(video_path)

        self.tracker.width = frame_width
        self.tracker.height = frame_height

        # init frame id
        frame_id = 0
        # print info
        print(f'Video name: {video_name}, frame width: {frame_width}, frame height: {frame_height}, frame rate: {frame_rate}, frame count: {frame_count}')
        # loop through video
        while cap.isOpened():
            # set frame / mes (frame per mes) 
            cap.set(cv2.CAP_PROP_POS_MSEC,(frame_id*pos_mes))
            # read frame
            ret, frame = cap.read()
            if ret:
                # crop mouth
                self.crop_mouth(frame=frame, frame_id=frame_id)
                # update frame id
                frame_id += 1

                if show:
                    # show frame
                    frame = self.draw_frame(frame=frame, height=frame_height, width=frame_width)
                    cv2.imshow('frame', frame)
                    # wait key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        # release video
        cap.release()

        cv2.destroyAllWindows()
                    
                