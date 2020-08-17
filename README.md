# CT Perfusion Repo
# Aim: Generating perfusion maps from CT images using different segmentation tools.

## This repo has 3 segmentation tools:

### 1. Segmentation of CT images using Gaussian mixture model with expectation maximization.
### 2. Segmentation of CT images using Fuzzy c-means.
### 3. Segmentation of CT images using K-means. 

## All the segmentation tools are followed by applying connected componant labelling algorithm for selecting local AIF. 

## Signal attenuation curve is drawn from the selected AIF location then using fitting spline, the Time to peak and maximum concentration are calculated from the curve. 


## Files Discription:

### attcurve.py > is an semi automatic algorithm uses GMM segmentation tool, it requires only selecting a point for the Arterial Input Function on the first frame of the loaded file, thus the script will draw automatically the signal attenuation curve from the selected artery or vien. 

## The following notebooks shows the different segmentation tools and the results 
### \Notebooks\attcurvUsingfuzzy > 
### \Notebooks\attcurvUsingkmeans > 
##3 \Notebooks\attcurvUsingGMM > 
