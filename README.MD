# Feature Matching

The program takes in template image, target image and detector type. It then performs feature matching based on the input
and gives output of matched features. 

## Installation of Libraries

It's recommended that you use virtualenv so that existing packages aren't tempered with. You'll need Python 3 installed with pip. Run:

`pip3 install - requirements.txt`

## Usage

`python3 main.py`


## Results

Metric used -

`(Inliers / Matches) * 100`

![Alt text](results.png?raw=true "Results")


## References

- [OpenCV Feature Detection and Description](https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md)
- [Feature Matching](https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md)
- [ASIFT](https://github.com/makelove/OpenCV-Python-Tutorial/blob/master/%E5%AE%98%E6%96%B9samples/asift.py)
