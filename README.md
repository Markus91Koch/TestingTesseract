# TestingTesseract
Test code for OCR using PyTesseract with different preprocessing steps

- largely based on the tutorial https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
- skew correction based on https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/

## How to use

- To load an image use the flags -i/--image in the command line and the image filename
- To select a certain preprocessing step, use -p/--preprocess and one of the options "blur" or "thresh" for blurring or thresholding of the image before OCR ensues

## Future steps
- perspective and other distortion correction
- local contrast correction
- transforming output text to suitable table format
