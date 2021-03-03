# CT Perfusion Repo
# Aim: Generating perfusion maps from CTA images using different segmentation tools.

## The developed approach uses only the specific "DEN" file format as input.

### First enhancement with normalization and contrast stretching are performed. 

### Currently three segmentation algorithm were developed for vessels extraction:

### 1. Segmentation of CTA images using Gaussian mixture model with expectation maximization.
### 2. Segmentation of CTA images using Fuzzy c-means.
### 3. Segmentation of CTA images using K-means. 

### All the segmentation tools are followed by applying connected componant labelling algorithm for selecting the local AIF. 

### Signal attenuation curve is drawn from the selected AIF location then using fitting spline, the Time to peak and maximum concentration are calculated from the curve. 


## Files Discription:

### extractionTool.py > is a semi automatic algorithm uses GMM segmentation tool, it requires only selecting a point for the Arterial Input Function on the first frame of the loaded file, thus the script will draw automatically the signal attenuation curve from the selected artery or vien. 
