'''
The program takes in template image, target image and detector type. It then performs feature matching based on the input
and gives output of matched features. 
The program is divided into 3 parts - 
1. The detector / descriptor 
    - Returns features and their descriptors for target and template image
    - We use SIFT or ASIFT here
2. The matcher 
    - Returns matching features between target and template based on their descriptors
    - We use FLANN here
3. homography 
    - transformation that maps the points in tmp image to the corresponding points in the target image
4. Evaluation Metric
    A. F1 Score
        - Currently only works with "target/Pepsi-vs-Coca-Cola.jpeg" and "target/rotate180.png"
    B. (Inliers / Matches) * 100
        - Uses inbuilt RANSAC Algorithm to get rid of outliers from the good matches.
        - The highest number of inliers to matches ratio indicates are robust detector/threshold.
'''


import numpy as np
from enum import Enum
import time
import cv2
from os import listdir
from multiprocessing.pool import ThreadPool
import time


class FeatureMatching:
    def __init__(self, target_path, template_path):
        self.tmp_img = cv2.imread(template_path)
        self.target_img =  cv2.imread(target_path)
        self.feature_detector = cv2.SIFT_create()
        self.min_match_count = 5
        self.eval_metric = "inliers"
        # self.pos = "left"

    def affine_skew(self,tilt, phi, img, mask=None):
        '''
        affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

        Ai - is an affine transform matrix from skew_img to img
        '''
        h, w = img.shape[:2]
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
        A = np.float32([[1, 0, 0], [0, 1, 0]])
        if phi != 0.0:
            phi = np.deg2rad(phi)
            s, c = np.sin(phi), np.cos(phi)
            A = np.float32([[c,-s], [ s, c]])
            corners = [[0, 0], [w, 0], [w, h], [0, h]]
            tcorners = np.int32( np.dot(corners, A.T) )
            x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
            A = np.hstack([A, [[-x], [-y]]])
            img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if tilt != 1.0:
            s = 0.8*np.sqrt(tilt*tilt-1)
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
            img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
            A[0] /= tilt
        if phi != 0.0 or tilt != 1.0:
            h, w = img.shape[:2]
            mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
        Ai = cv2.invertAffineTransform(A)
    
        return img, mask, Ai


    def affine_detect(self,img, mask=None, pool=None):
        '''
        affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

        Apply a set of affine transformations to the image, detect keypoints and
        reproject them into initial image coordinates.
        See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

        ThreadPool object may be passed to speedup the computation.
        '''
        params = [(1.0, 0.0)]
        for t in 2**(0.5*np.arange(1,6)):
            for phi in np.arange(0, 180, 72.0 / t):
                params.append((t, phi))
        def f(p):
            t, phi = p
            timg, tmask, Ai = self.affine_skew(t, phi, img)
            keypoints, descrs = self.feature_detector.detectAndCompute(timg, tmask)
            for kp in keypoints:
                x, y = kp.pt
                kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
            if descrs is None:
                descrs = []
            return keypoints, descrs

        keypoints, descrs = [], []
        if pool is None:
            ires = it.imap(f, params)
        else:
            ires = pool.imap(f, params)

        for i, (k, d) in enumerate(ires):
            print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
            keypoints.extend(k)
            descrs.extend(d)
        return keypoints, np.array(descrs)
        
    def get_kp(self,matches,all_kp,good_kp):
        '''
        get_kp(\matches,all_kp,good_kp) ->all matches, good matches

        returns left all matches, right all matches, left good matches, right good matches
        '''
        image = self.target_img
        width,height, channels = image.shape[1], image.shape[0], image.shape[2]
        half_width = width//2
        all_kp_idx = np.float32([key_point.pt for key_point in all_kp]).reshape(-1, 1, 2)
        good_kp_idx = good_kp.reshape(good_kp.shape[0],2)
        left_kp = sum([coord[0][0] <= half_width for coord in all_kp_idx])
        righ_kp = sum([coord[0][0] > half_width for coord in all_kp_idx])
        left_match = sum([coord[0] <= half_width for coord in good_kp_idx])
        righ_match = sum([coord[0] > half_width for coord in good_kp_idx])
        return left_kp,righ_kp,left_match,righ_match
    
    def evaluate_f1(self,ld,rd,lm,rm):
        '''
        evaluate_f1(ld,rf,lm,rm) -> left all matches, right all matches, left good matches, right good matches

        returns f1 score
        '''
        if self.pos == "left":
            Precision = lm / (lm + rm)
            Recall = lm / (lm + (ld - lm)) 
        else:
            Precision = rm / (rm + lm)
            Recall = rm / (rm + (rd-rm)) 
        f1_score =  round(2 * (Precision * Recall) / (Precision + Recall),3)
        return f1_score

    def match_ft(self,ratio,det_name):
        '''
        match_ft(ratio, det_name) -> ratio for distance measure, name of the detector to be used

        returns image with bounding box + matches as well as score
        '''
        score = 0
        if det_name == "SIFT":
            kp, des = self.feature_detector.detectAndCompute(self.target_img, None)
            kp0, des0 = self.feature_detector.detectAndCompute(self.tmp_img, None)
        elif det_name == "ASIFT":
            pool=ThreadPool(processes = cv2.getNumberOfCPUs())
            kp, des = self.affine_detect(self.target_img,pool=pool)
            kp0, des0 = self.affine_detect(self.tmp_img,pool=pool)
        else:
            print("Please choose correct detector (ASIFT/SIFT)")
            return None,score

        # FLANN Matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) 
        flann = cv2.FlannBasedMatcher(index_params,search_params)

        #knnMatch returns 2 best matches for all descriptors 
        matches = flann.knnMatch(des0, des, k=2)
        
        # Keep good feature matches only
        good = [[m] for m, n in matches if m.distance < ratio * n.distance]
        
        # Find Homography
        if (len(good) > self.min_match_count):
            src_pts = np.float32([kp0[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            

            # Draw Bounding Box
            h, w, c = self.tmp_img.shape 
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(self.target_img, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            print("No good matches found")
            return self.target_img, score
        if self.eval_metric == "f1":
            ld,rd,lm,rm = self.get_kp(good,kp,dst_pts)
            score = self.evaluate(ld,rd,lm,rm)
        else:
            score = (np.sum(mask)/len(mask))*100 # (inliers / matches) * 100 
        # Visualize the matches
        draw_params = dict(flags=2)
        img = cv2.drawMatchesKnn(self.tmp_img, kp0, frame, kp, good, None, **draw_params)
        return img,score
        

if __name__ == '__main__':
    # tmp_path = "template/coca-cola.png"
    tmp_path = "template/pepsi.png"
    target_path = "target/original.jpeg"
    ft_mt = FeatureMatching(target_path,tmp_path)
    detector_name = "SIFT"
    evaluation_dict = [0.5]
    output_path = "output/" + detector_name + "/"  + target_path.split("/")[1].split(".")[0] + "_" + tmp_path.split("/")[1]
    print(output_path)
    for ratio in evaluation_dict:
        start_time = time.time()
        img,score = ft_mt.match_ft(ratio,detector_name)
        print("--- Time taken : %s seconds ---" % (time.time() - start_time))
        print("Evaluation for ratio " + str(ratio) + ": " + str(round(score,3)))
        img = cv2.resize(img, (960,480), interpolation= cv2.INTER_LINEAR)
        cv2.imshow("Output", img)
        # cv2.imwrite(output_path, img)
        cv2.waitKey(0)
        cv2.destroyWindow("output")
            
    
    
    


    

