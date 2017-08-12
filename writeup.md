###Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./image/raw_undist.png "Undistorted"
[image1]: ./image/RoadTransformed.png "Road Transformed"
[image2]: ./image/RoadTransformed.png "Road Transformed"
[image3]: ./image/HLS.png "HLS image"
[image4]: ./image/HSV.png "HSV image"
[image5]: ./image/Lab.png "Lab image"
[image6]: ./image/Sobelx.png "Sobel x gredient"
[image7]: ./image/Sobely.png "Sobel y gredient"
[image8]: ./image/Sobelmag.png "Sobel gredient magnitude"
[image9]: ./image/Sobeldir.png "Sobel gredient direction"
[image10]: ./image/combine.png "Combine"
[image11]: ./image/warpedperspectivetransform.png "Warped perpective"
[image12]: ./image/hist.png "histogram"
[image13]: ./image/detectbywindowsliding.png "Fitbywindowsliding"
[image14]: ./image/detectbymargin.png "Fitbymargin"
[image15]: ./examples/color_fit_lines.jpg "Fit Visual"
[image16]: ./image/output.png "Output"
[video17]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cell one of the IPython notebook located in "./mycode.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image0]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image1]

I used  the matrix and the coefficent of the camera to undistort the raw image.
you can find the code at the cell two and three of my code "./mycode.ipynb"
After the undistortion transforming,it looks likes:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used several transforming method to generate a binary image for detecting the lane lines.they are as follows:

* use the BGR to HLS convertion method to get the saturation channel of the image
* use the BGR to HSV convertion method to get the value channel of the image
* use the BGR to Lab convertion method to get the value channel of the image
* use the sobel detection method to get the x gradient of the image
* use the sobel detection method to get the y gradient of the image
* combine x gradient and y gredient of the image to get its gradient magnitude
* combine x gradient and y gredient of the image to get its gradient direction

All aboved outcome image are as fllows:

![alt text][image3]

        S channel of the HLS image

![alt text][image4]

        V channel of the HSV image

![alt text][image5]

        L channel of the Lab image

![alt text][image6]

         X gradient of the image

![alt text][image7]

         Y gredient of the image

![alt text][image8]

         Gradient magnitude of the image

![alt text][image9]

         Gradient direction of the image

At the last of this step ,I used them to get a combination of color and gradient thresholds to generate a binary image. My code is like this:

```python
imgcomb=np.zeros_like(imgbysobelmag)
imgcomb[(imgbyhlscolor==1)|(imgbyhsvcolor==1)|(imgbysobelmag==1)\
        |(imgbysobelx==1)|(imgbysobely==1)]=1
```
Because I found that the Gradient direction of the image have much noise,I don't use it in my combination.
Here's an example of my output for this step.  
![alt text][image10]

      combination binary of the image


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `GetPerspectiveTransformImg()`, which appears in the Cell twelve of the IPython notebook).  The `GetPerspectiveTransformImg()` function takes as inputs an image (`img`).Then it uses the source (`src`) and destination (`dst`) points to get a tranform matrix.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
                [[(img_size[0] / 2) - 80, img_size[1] / 2 + 100],
                [((img_size[0] / 6) - 10), img_size[1]],
                [(img_size[0] * 5 / 6) + 10, img_size[1]],
                [(img_size[0] / 2 + 80), img_size[1] / 2 + 100]])
dst = np.float32(
                [[(img_size[0] / 4), 0],
                [(img_size[0] / 4), img_size[1]],
                [(img_size[0] * 3 / 4), img_size[1]],
                [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 521, 410      | 300, 0        | 
| 190, 621      | 300, 621      |
| 1011, 621     | 901, 621      |
| 681, 410      | 901, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.It looks as this:

![alt text][image11]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image11]
![alt text][image12]
![alt text][image13]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I define a function named 'CalculateLaneCurveRadAndVehicleOffset" to calculate lane curve radius and the offset from the vehicle to the lane center. 
This function get three parameters:

* y_eval:  which position you want to calculate
* fit:  the 2nd order polynomial parameter you just fit
* ym_per_pix = 30/720:  the distance that a horizontal pixel equals 
* xm_per_pix = 3.7/700:  the distance that a vertical pixel equals

From these three parameters, I fit new polynomial line which have a real scale of the world.Then we can calculate its curve radius of every point on the curve.
I use the x coordinate of the start point of the curve substract the x coordinate of the center of the image which indicate the vehicle's center.
After scaling this difference(multiply by ym_per_pix),I get the distance between the vehicle center to the lane center.  
The detail code can be found in  cell eighteen of my code"./mycode.ipynb"

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the Cell nineteen of my code "./mycode.ipynb" in the function ` AddDectectedLaneToImg()`. Here is an example of my result on a test image:


![alt text][image16]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

At last, I used all aboved method and tricks to combine a pipeline function to detect the lane of image.
In this pipeline,I use some tricks to improve the performance.For example,I use a Class object to store the last three detected output and use them to calculate a best fit.For each step,I just modified the fit a bit.The formula is as follows:

       current_best_fit=0.95*last_best_fit+0.05*current_fit
       
Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have used several type of transforming method to get a good binary image for detecting the lane line.But some times they all will be failure.Because the parameter we choosed,such as the threshold, can not perform very well under all conditions.Some value can perform well in some images but badly in other images.I found two types of  condition under which the pipeline will fail:

* one condition is that if the transforming method perform not well and no point can be detected from the binary image.
* another condition is that if pipeline have a bad detected outcome in the first image of the video,it will perform badly later.

The first condition may be improved by using a method that is:
if lane line at one side is not detected at all,we can use other side lane line as a reference to generate it.

The second condition may be avoided by using a method that is:
when we use the transforming method ot get a binary image,we can use a dynamical value which can be defined by the image's average color value (or some other feature). 
