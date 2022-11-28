# 45828894 Improved UNet

Segment the ISICs data set with the Improved UNet [1] with all labels having a minimum Dice similarity
coefficient of 0.8 on the test set.

## Dependencies
 - tensorflow==2.5.0
 - jupyterlab
 - matplotlib

## Driver Script
None

## Description

This task was attempted but not finished. Majority of the structure setup was complete but not finished.

## Discussion
As many people had queries about having small holes inside their segmentation. I would like to share my insights in the problem.

This problem can be easily fixed by applying any contour extracting algorithm such as `ant walk`.
If there are holes in the existing segmentation, it will return two contours, one describing the actual segmentation, the other describing the hole.
by simply taking the largest contour (which is always the segmentation) and applying it to a blank binary array. A new segmentation without holes can be created.

## Results

None

## References
 - https://arxiv.org/abs/1802.10508v1