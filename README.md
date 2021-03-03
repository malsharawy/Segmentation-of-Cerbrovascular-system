
# Project title: "Brain vascular system segmentation using multisweep C-Arm CT protocols.

# Aim: Investigate whether the reconstructed volume of 4D-CTA using time sepration technique provides sufficient diagnostic information by testing 

  ## 1. The ability to segment the cerebrovascular system of the brain from the reconstructed volume.
  ## 2. Extracting coordinates of different arteries to be used in estimating the AIF and generate the perfusion maps in order to investigate the influence of selecting different AIF locations on the perfusion maps.

## The developed approach uses only the specific "DEN" file format as input. 
#### Later it will be updated to use DICOM format.


### 1. First enhancement with normalization and contrast stretching are performed. 

### 2. User window will pop up to enables selecting the target vessel. 

### 3. Currently three segmentation algorithm were developed for vessels extraction:
#### 3.1. Segmentation of CTA images using Gaussian mixture model with expectation maximization.
#### 3.2. Segmentation of CTA images using Fuzzy c-means.
#### 3.3. Segmentation of CTA images using K-means. 

### 4. All the segmentation tools are followed by applying connected componant labelling algorithm for selecting target vessel.
### 5. The target vessel and it`s coordinates will be saved and the a window will pop up to show the result. 


## Files Discription:
### extractionTool.py > is a semi automatic algorithm uses GMM segmentation tool, it requires only selecting a point for the Arterial Input Function on the first frame of the loaded file, thus the script will draw automatically the signal attenuation curve from the selected artery or vien. 

## Script execution command
### The script can be executed from the terminal using the following inputs and flags,

### pythonDirectory/python segment.py -f pathToFile -fFrom firstFrameNumber -fTo LastFrameNumber -z ZframeNumber -seg segmentationType -s pathToSaveCoordinates


 ### [-f] is a required flag to assign a path to the CTA volume file in den format. 
 ### [-fFrom] and [-fTo] are required flags to select a group of frames from the volume, [-fFrom] indicates the starting frame number and [-fTo] indicates the last frame number. 
 ### [-s] is a required flag of the desired directory for saving the coordinates information.
 ### [-z] is an optional flag can be used for showing specific frame for AIF placement point selection, the default is that the middle frame will be shown.  
 ### [-seg] is an optional flag to force the script to use specific segmentation type, the default is GMM clustering algorithm. 
