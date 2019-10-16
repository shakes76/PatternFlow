# Denoising
Various denoise algorthms implemented for Tensorflow

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

## Contributing
* Fork the dedicated 'topic-algorithms' branch
* Create a directory for your algorithm and place your code into it.
* Your code should have functions in a separate module and a driver (main) script that runs it with parameters defined.
* The driver script should preferably either plot or save results
* Add a README.md file as described by the report assessment
* You may upload a low res image (< 2 MB) into a folder called 'resources' in your fractal directory for displaying on the README.md file
* You can see an example of this in the [SMILI repository](https://github.com/shakes76/smili).
* Then put in a pull request for the repository owner to approve and that's it!

## Environment 
* torch version : 1.2.0
* mac os version: 10.14.6

## Algorithm developed
* denoise_tv_chambolle

## Result 

* Comparisons between the original image and denoised image in torch and numpy \
![result](https://i.imgur.com/5rJt22s.png)
