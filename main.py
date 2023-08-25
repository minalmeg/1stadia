import numpy as np
from enum import Enum
import time
import cv2
from cv2.xfeatures2d import matchGMS
from os import listdir
from os.path import isfile, join


class FeatureMatching:
    def __init__(self, target_path, template_path):
        self.tmp_fldr_path = template_path
        self.target_img =  cv2.imread(target_path)
        self.feature_detector = cv2.AKAZE_create()
        self.ratio = 1  
        self.min_match_count = 5
        
    def preprocess(self,template):
        bf = cv2.BFMatcher()

        kp, des = self.feature_detector.detectAndCompute(self.target_img, None)
        kp0, des0 = self.feature_detector.detectAndCompute(template, None)
        queryimg = template

        # Apply blute-force knn matching between keypoints
        matches = bf.knnMatch(des0, des, k=2)

        # Adopt only good feature matches
        good = [[m] for m, n in matches if m.distance < self.ratio * n.distance]

        # Find Homography
        if (len(good) > self.min_match_count):
            src_pts = np.float32([kp0[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w, c = template.shape  # Assume color camera
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(self.target_img, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)

        # Visualize the matches
        draw_params = dict(flags=2)
        #draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=0)
        #img = cv2.drawMatchesKnn(template, kp0, self.target_img, kp, good, None, **draw_params)
        img = frame
        return img

    def read_templates(self):
        all_files = [f for f in listdir(self.tmp_fldr_path) if isfile(join(self.tmp_fldr_path, f))]
        counter = 0
        for pth in all_files:    
            counter += 1
            template = cv2.imread(pth)
            output = self.preprocess(template)
            cv2.imshow("Template " + str(counter), output)
        cv2.waitKey(0) 

if __name__ == '__main__':
    tmp_folder_path = "template/"
    target_path = "Pepsi-vs-Coca-Cola.jpeg"
    ft_mt = FeatureMatching(target_path,tmp_folder_path)
    ft_mt.read_templates()

