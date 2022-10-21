# Brain MRI Super-Resolution Network - 46464383
This algorithm leverages an efficient subpixel CNN to upsample an image when given a low resolution version. THese techniques have applications in HDTV, medical imaging, satellite imaging as well as in face recognition and surveillance [2]. In this report, I am focusing on medical imaging. My algorithm upsamples MRI brain scans to prodice a higher resolution image. By increasing the resolution, a clearer picture can be extracted, allowing medical practitioners to view and analyse the output with more clarity. Generally, a high resolution is required to facilitate accurate and early diagnosis. This can be hindered by short time frames and patients motion during the scan which may result in blurry images.


how the algorithm works
figure/visualisation

## Dependencies
- Tensorflow 2.9.2
- Keras 2.9.0
- Matplotlib 3.2.2
- PIL 7.1.2
- Numpy 1.21.6

math and os libraries are also used (Python 3.7.15)

The code run with these libraries should produce similar results to those presented in this report. However, due to the nature of a neural network, there will be variations in exact training and testing scores.
## Usage
3. provide example inputs, outputs and plots of your algorithm
### Inputs
The input into the model is a 64x64x1 image of an MRI.
### Outputs
![Comparison between original image, model outout and bicubic scaling](./images/multi_comp_5.png)

## Preprocessing
5. Describe any specific pre-processing you have used with references if any. Justify your training, validation
and testing splits of the data

## References
[1] https://keras.io/examples/vision/super_resolution_sub_pixel/#run-model-prediction-and-plot-the-results
[2] https://arxiv.org/pdf/1609.05158.pdf
[3] https://onlinelibrary.wiley.com/doi/epdf/10.1002/cmr.a.21249
