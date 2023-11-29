# Spot People Recognition
Yolo has been implemented using a dedicated module of OpenCV [docs](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html). 
Limited by the use of only the CPU, the *darknet* framework was employed.You can find [here](https://github.com/AlexeyAB/darknet) the main repository to download, compile and train a yolo network with *darknet*. 
Versions later than Yolov4 are not suitable for OpenCV, so Yolov3-tiny has been employed, representing a good compromise between reliability and processing time. 

## Installation
Clone this repo to your workspace and rename the package name to "spot_human_detetction". 

To use this package correctly, it is necessary to check if you have the following dependencies:
- OpenCV
- cv_bridge

Inside this repository you can already find the config files for Yolov3, Yolov3-tiny and a training test for YoloV3-tiny is called yolo yolo-tiny-tabi.
The .weights files are downloadable from these links:
- Yolov3.weights: [Dowlonad](https://mega.nz/file/0GMxDRzR#GXPZvzUAYqA-9Bv66gm3BzdPiVOm-p0NUNTlwxq_8Ww)
- Yolov3-tiny.weigths: [Dowlonad](https://mega.nz/file/cfNRARwQ#Ry5i5LEmigOe8x1idFBgTXRJVA_RIU2kyxOZz-H5BY4)
- Yolo-tiny-tabi.weights: [Dowlonad](https://mega.nz/file/1T8iTCjL#6hESjGGpTlmCB4gegabCss5VGdiUWpuL1xj04n5pA68)

Download these files and save them in the 'yolo' folder of the package.

Now you are redy to run the human_detection_v2.py node. 
