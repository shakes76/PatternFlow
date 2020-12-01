# Gamma Correction

## Backgroud

> Gamma correction, or often simply gamma, is a nonlinear operation used to encode and decode luminance or tristimulus values in video or still image systems.

## Structure

```python
PatternFlow
    image
        gamma_correction
            main.py             # Driver script.
            gamma_correction.py # Core function.
            input.png           # Test input image.
            output.png          # Output image.
            README.md           # Summary.
```

## Dependency

- Opencv        # Used in driver script for read and display image.
- Tensorflow    # Used in core function to to parallels calculation.
- If test case is not necessarily, skip opencv installation with only tensorflow installed.

```
pip install opencv-python
pip install --upgrade tensorflow
```

## Usage

```
git clone https://github.com/shakes76/PatternFlow.git
cd image
cd gamma_correction
python main.py
```

## Result
![Input](input.png)
![Output](output.png)


## Author

Name: Lawrence Hu
ID: 45702024

