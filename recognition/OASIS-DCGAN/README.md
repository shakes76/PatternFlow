# OASIS-DCGAN
*By Reece Jocumsen (44786803)*

Deep Convolutional Generative Adversarial Network using the OASIS brain MRI scans. DCGAN structure following [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf), particularly Section 3 - Approach and Model Architecture.

## Usage
OASIS-DCGAN requires the following libraries:
* tensorflow-gpu or tensorflow for the model
* numpy for image preprocessing
* matplotlib for image plotting and saving

To run from command prompt:
`py driver.py train_dir test_dir optional_result_dir`
    
Where:
`train_dir` = the directory where training images are stored (e.g. `C:\Users\Public\Pictures\training\` or `training\`)
`test_dir` = the directory where test images are stored (e.g. `C:\Users\Public\Pictures\training\` or `training\`)
`optional_result_dir` = the directory where generated images will be stored (e.g. `C:\Users\Public\Pictures\result` or `result\`, default: `result\`)

## Examples of Generated Images
<a data-flickr-embed="true" href="https://www.flickr.com/photos/141453561@N03/50573025182/in/dateposted-public/" title="0511201447"><img src="https://live.staticflickr.com/65535/50573025182_5ebf7fabe7.jpg" width="500" height="135" alt="0511201447"></a><script async src="//embedr.flickr.com/assets/client-code.js" charset="utf-8"></script>

<a data-flickr-embed="true" href="https://www.flickr.com/photos/141453561@N03/50573025182/in/dateposted-public/" title="0511201417"><img src="https://live.staticflickr.com/65535/50572893516_99987dc6c0_k.jpg" width="500" height="135" alt="0511201417"></a><script async src="//embedr.flickr.com/assets/client-code.js" charset="utf-8"></script>


