import cv2
import os
import copy
import pickle
import argparse
import numpy as np
from OpenSeeFace.tracker import Tracker

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Create Landmark Tracker')
    # -- utils
    parser.add_argument('--video-path', default=None, help='raw video directory folder or file path')
    parser.add_argument('--pos-mes', default=10, help='frame per second')
    # -- for landmark
    parser.add_argument('--save-landmark', default=False, help='boolean to save landmark picke file')
    parser.add_argument('--dir-save-landmark', default=None, help='the directory of saving mouth landmark pickle file, if save-landmark is False, this param will be ignored')
    # --for crop mouth
    parser.add_argument('--save-crop-mouth', default=False, help='boolean to save crop mouth image')
    parser.add_argument('--mouth-crop-threshold', default=0.0, help='threshold to crop mouth')
    parser.add_argument('--pad', default=10, help='pad to crop mouth, if crop-mouth is False, this param will be ignored')
    parser.add_argument('--dir-save-crop-mouth', default=None, help='the directory of saving crop mouth image, if save-crop-mouth is False, this param will be ignored')
    parser.add_argument('--show', default=True, help='boolean to show landmark image')

    args = parser.parse_args()
    return args

class LandmarkTracker:
    def __init__(self, model_type=3, detection_threshold=0.6, threshold=None, 
    max_faces=1, discard_after=5, scan_every=3, bbox_growth=0.0, max_threads=4, silent=False, 
    model_dir=None, no_gaze=False, use_retinaface=False, max_feature_updates=0, static_model=False, 
    feature_level=2, try_hard=False):

        self.tracker = Tracker(width=224, height=224, model_type=model_type, detection_threshold=detection_threshold, threshold=threshold, 
        max_faces=max_faces, discard_after=discard_after, scan_every=scan_every, bbox_growth=bbox_growth, max_threads=max_threads, silent=silent,
        model_dir=model_dir, no_gaze=no_gaze, use_retinaface=use_retinaface, max_feature_updates=max_feature_updates, static_model=static_model,
        feature_level=feature_level, try_hard=try_hard)

        self.save_mouth_image = False
        self.dir_save_mouth_crop = ""
        self.dir_save_landmark = ""
        self.pad = int
        self.mout_crop_threshold = float
    
    def save_mouth_crop(self, crop_mouth_image, face_id, frame_id):
        if self.save_mouth_image:
            path_dir = os.path.join(self.dir_save_mouth_crop, str(face_id))
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
                    
    def get_landmark(self, video_path, pos_mes = 100,
    save_landmark=False, mouth_crop_threshold = 0.0, pad = 10, 
    save_crop_mouth = True, dir_save_crop_mouth="", dir_save_landmark="", show = True):

        self.mouth_crop_threshold = mouth_crop_threshold
        self.dir_save_mouth_crop = dir_save_crop_mouth
        self.dir_save_landmark = dir_save_landmark
        self.pad = pad
        self.save_mouth_image = save_crop_mouth

        # open video
        cap = cv2.VideoCapture(video_path)
        # get video info
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_rate = cap.get(5)
        frame_count = int(cap.get(7))
        frame_interval = frame_count / frame_rate
        # get video name
        video_name = os.path.basename(video_path)

        self.tracker.width = frame_width
        self.tracker.height = frame_height

        # init frame id
        frame_id = 0

        landmarks = []
        # print info
        print(f'Video name: {video_name}, frame width: {frame_width}, frame height: {frame_height}, frame rate: {frame_rate}, frame count: {frame_count}')
        # loop through video
        while cap.isOpened():
            # set frame / mes (frame per mes) 
            #cap.set(cv2.CAP_PROP_POS_MSEC,(frame_id*pos_mes))
            
            # cap number frames = 29 =  frame_rate
            cap.set(cv2.CAP_PROP_POS_FRAMES, (frame_id) * frame_interval)
            # read frame
            ret, frame = cap.read()
            if ret:
                # crop mouth
                if self.save_mouth_image:
                    self.crop_mouth(frame=frame, frame_id=frame_id)
                # landmark
                if save_landmark:
                    faces = self.tracker.predict(frame=frame)
                    landmark = []
                    for face in faces:
                        if face.success:
                            for pt_num, (x,y,c) in enumerate(face.lms):
                                landmark.append((y,x))
                    # convert to numpy array
                    landmark = np.array(landmark)
                    # save landmark
                    if landmark.shape[0] > 0:
                        landmarks.append(landmark)
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
        
        # save landmark to pickle file with directory
        print(self.dir_save_landmark)
        # EX: an/train/t.mp4 => get an/train
        label_name = os.path.dirname(video_path).split("/")[-2] + "/" + os.path.basename(os.path.dirname(video_path))
        save_path  = os.path.join(self.dir_save_landmark, label_name)
        if save_landmark:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, f'{video_name}.pkl'), 'wb') as f:
                pickle.dump(landmarks, f)
                print(f'Landmark saved successfully')

        # release video
        cap.release()

        cv2.destroyAllWindows()
                    
# main function
if __name__ == '__main__':
    args = load_args()

    Landmark = LandmarkTracker()
    sub_folders = ["train", "test", "val"]
    #check video path is folder or file
    if os.path.isdir(args.video_path):
        for folder in os.listdir(args.video_path):
            for sub_folder in sub_folders:
                for file in os.listdir(os.path.join(args.video_path, folder, sub_folder)):
                    #check the file landmark has exist
                    if os.path.exists(os.path.join(args.dir_save_landmark, folder, sub_folder, file + ".pkl")):
                        continue
                    if file.endswith(".mp4"):
                        video_path = os.path.join(args.video_path, folder, sub_folder, file)
                        Landmark.get_landmark(video_path = video_path, pos_mes = args.pos_mes, 
                        save_crop_mouth= args.save_crop_mouth, save_landmark=args.save_landmark, 
                        mouth_crop_threshold = args.mouth_crop_threshold, 
                        pad = args.pad, dir_save_crop_mouth=args.dir_save_crop_mouth,
                        dir_save_landmark=args.dir_save_landmark,
                        show = args.show)
    else:
        Landmark.get_landmark(video_path = args.video_path, pos_mes = args.pos_mes, 
        save_crop_mouth= args.save_crop_mouth, save_landmark=args.save_landmark, 
        mouth_crop_threshold = args.mouth_crop_threshold, 
        pad = args.pad, dir_save_crop_mouth=args.dir_save_crop_mouth,
        dir_save_landmark=args.dir_save_landmark,
        show = args.show)


