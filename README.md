# sevensegment-python
sevensegment-python outputs numerical value from an image containing 7-segment.

## About
By specifying the image and the area of 7-segments in it, each segment is divided and the numerical value is output. 

Since horizontal and vertical features are used for numerical judgment, reducing distortion in the input image is effective for improving accuracy.

## Requirement
<pre>
OpenCV      4.2.0  
Numpy       1.18.4  
Matplotlib  3.1.3
</pre>

## Installation
### OpenCV
```bash
$ pip install opencv-python
```

If you use Anaconda,
```bash
$ conda install -c conda-forge opencv
```

## LICENSE
[MIT](https://choosealicense.com/licenses/mit/)
