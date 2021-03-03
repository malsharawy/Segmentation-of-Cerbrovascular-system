# This script is created as a part of master thesis project and only tested on 4D-CTA reconstructed volume (Volumes enhanced with Contrast agent)
# Author: Mohamed Elsharawy

# import the necessary packages
import numpy as np  
# https://numpy.org/

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor,Button
# https://matplotlib.org/

from denpy import DEN
# Provided by the supervisor RNDr. Vojtěch Kulvait, Ph.D.,vojtech.kulvait@ovgu.de

import argparse
# https://github.com/python/cpython/blob/3.9/Lib/argparse.py


import cc3d
# https://pypi.org/project/connected-components-3d/

from skimage import segmentation,measure,morphology,draw,filters,exposure
from skimage.measure import regionprops
from sklearn.mixture import GaussianMixture as gmm
# https://scikit-image.org/



# Define the necessary functions
# Function to import the den file
def getTheFile(filename,rows,cols,start,stop):
    # This function is used to import the den file and convert it to numpy array
    temp = DEN.getFrame(filename, 0, row_from=None, row_to=None, col_from=None, col_to=None) 
    arr = np.empty([temp.shape[0], temp.shape[1], 0])    
       
    for i in range(start,stop,1):
        Frame = DEN.getFrame(filename, i, row_from=None, row_to=None, col_from=None, col_to=None)
        dentoarr = np.asarray(Frame)
        dentoarr = np.expand_dims(dentoarr, axis=2)
        arr = np.concatenate((arr, dentoarr), axis=2)
    return arr
# Function to save the array in den format
def saveDENfile(arrName, arr):
    DEN.writeEmptyDEN(arrName, arr.shape[0], arr.shape[1],arr.shape[2], force=True)
    for i in range(arr.shape[2]):
        dcm = arr[:,:,i]
        DEN.writeFrame(arrName, i, dcm, force=True)

# Function to apply the GMM segmentation.
def ExtractTissueByGMM(img3d):
    arr = np.empty([img3d.shape[0], img3d.shape[1], 0]) 
    orginalShape =img3d.shape
    reshaped = img3d.reshape(-1,1)
    clustersNumber = 3
    gmm_model_CTa= gmm(n_components = clustersNumber, covariance_type='tied').fit(reshaped)
    gmm_labels =gmm_model_CTa.predict(reshaped)
    segmented_frames= gmm_labels.reshape(orginalShape[0],orginalShape[1],orginalShape[2])
    return segmented_frames

# Function to apply the kmeans segmentation.
def ExtractTissueBykmeans(img3d):
    from sklearn.cluster import KMeans
    clustersNumber = 3
    kmeans = KMeans(clustersNumber, random_state=0)
    arr = np.empty([img3d.shape[0], img3d.shape[1], 0]) 
    orginalShape =img3d.shape
    reshaped = img3d.reshape(-1,1)
    labels = kmeans.fit(reshaped).predict(reshaped)
    kmeans_segmented_frames = labels.reshape(orginalShape[0],orginalShape[1],orginalShape[2])
    return kmeans_segmented_frames

# Function to apply the fuzzy segmentation.
def ExtractTissueByfuzzy(img3d):
    from fcmeans import FCM
    print("segmenting with fuzzy ")
    original_shape = img3d.shape
    reshaped  = img3d.reshape(-1,1)
    fcm = FCM(n_clusters=3)
    fcm.fit(reshaped)
    # outputs
    fcm_centers = fcm.centers
    fcm_labels  = fcm.u.argmax(axis=1)
    VesselsMask = fcm_labels.reshape(original_shape[0],original_shape[1],original_shape[2]) 
    return VesselsMask

# Function to apply the connected components labelling algorithm with removing the small objects to reduce the output labels.
def MyCC3d(labeledarr,originalarr):
    import cc3d
    from skimage import segmentation,measure,morphology
    print("computing cc3d")
    connectivity=6
    labels_out = cc3d.connected_components(labeledarr,connectivity=connectivity)
    print("removing small size objects to decrease the output labels")
    #Removing small size objects to decrease the output labels, The following code line can be removed or modified depending on the expected size of the target object. 
    # interior_labels = morphology.remove_small_objects(labels_out, min_size=5)
    print("relabelling")
    #Relablling to obtain sequence numbering. 
    relabeled, _, _ = segmentation.relabel_sequential(labels_out)
    print("computing the objects properties")
    # regionprops = measure.regionprops(relabeled, intensity_image=originalarr)
    print("Computing the areas")
    # areas = [regionprop.area for regionprop in regionprops]
    return relabeled,labels_out

# Function to store the coordinate of the mouse click.
def onclick(event):   
    global ix, iy
    ix, iy = event.xdata, event.ydata
    global Ulabel
    Ulabel.append((ix, iy))
    ix=int(ix)
    iy=int(iy)
    print("Selected coordinates : ", iy, ix , z)
    return ix,iy


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description="extract manually the desired vessel and show the attenuation curve")
ap.add_argument("-f", "--file", required=True,	help="Path to den file")
ap.add_argument("-seg", "--segmentType", required=False,	help="indicate the segmentation type(gauss or fuzzy or kmeans)")
ap.add_argument("-fFrom", "--frameFrom", required=True,	help="indicate the starting frame number")
ap.add_argument( "-fTo" , "--frameTo", required=True,	help="indicate the last frame number")
ap.add_argument("-Z", "--Zplane", required=True,	help="indicate the desired frame number")
ap.add_argument( "-s" , "--saveLocations", required=False,	help="save the locations of the vessel voxels under the entered name")
args = vars(ap.parse_args())

# display messages to the user
print("Thank you for uploading the {} file ".format(args["file"] )) 
print("from frame No. {}".format(args["frameFrom"]))
print( "to frame No. {}".format(args["frameTo"]))
print( "segmentaing the input images with. {}".format(args["segmentType"]))

#saving the startFrame and endFrame 
StrtF = np.int(args["frameFrom"])
EndF = np.int(args["frameTo"])
segType = args["segmentType"]
z = np.int(args["Zplane"])

# Function to  get the coordinates of pixels inside a radius r and center ix, iy (selected by the user) 
def drawpoly(frame, iy,ix, rad=15):
    from skimage.draw import disk,circle,circle_perimeter
    mask = np.zeros(frame.shape, dtype=np.uint8)
    rr, cc = circle(iy, ix, radius=rad)
    rp, cp = circle_perimeter(iy, ix, radius=rad)
    rd, cd = disk((iy, ix), radius = rad)
    mask[rd, cd] = 1
    return mask

# Function to save the coordinates to txt file.
def savetxt(arrName,arr):
    from numpy import savetxt
    arr = np.transpose(arr)
    savetxt(arrName, arr, delimiter=',')

# Function to draw a circle on the selected point
def circle_points(resolution, center, radius):
    """
    Generate points which define a circle on event centre 
    """   
    radians = np.linspace(0, 2*np.pi, resolution)
    c = center[1] + radius*np.cos(radians)#polar co-ordinates
    r = center[0] + radius*np.sin(radians)
    return np.array([c, r]).T



#Getting the enhanced volume
print("File is loading ")
E_Carm = getTheFile(args["file"],500,500,StrtF,EndF)
print("File is loaded with the shape: ",E_Carm.shape)

#Getting the index of the highest normalized value
k,i,j = np.unique(np.where(E_Carm == np.max(E_Carm[:,:,z])))

#rescale contrast
reshaped = E_Carm.reshape(-1,1)

# Contrast stretching
p2, p98 = np.percentile(reshaped, (2, 98))
rescaled = exposure.rescale_intensity(reshaped, in_range=(p2, p98))
rescaled = rescaled.reshape(E_Carm.shape[0],E_Carm.shape[1],E_Carm.shape[2])
print(rescaled.shape)
print("max =",rescaled.max(),"min = ",rescaled.min()) 

# print("pixels data type", rescaled.dtype) 
rescaled = np.asarray(rescaled,np.float)
print("pixels data type", rescaled.dtype)

# Define user selection label as global label
Ulabel=[]

# store a frame for user selection
R_frame = rescaled[:,:,z].copy()

# Initiate the user window for selecting the label of the desired vessel 
print("Please select the placement location for the arterial input function then close the window") 
fig, au = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
au.imshow(R_frame, cmap="viridis")
# ax.imshow(orig_frame, cmap="viridis")

# show horz and vert lines of the mouse cursor
Ucursor = Cursor(au,horizOn=True,vertOn=True,color='red',linewidth=1.0)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
iy= int(iy)
ix= int(ix)


#Morphological operation opening to disconnect the close vessels and remove the artifacts strikes
opened = morphology.opening(rescaled,selem=morphology.ball(2))
# j,i = np.unique(np.where(opened[:,:,z] == np.max(opened[:,:,z])))
# print("j: ",j, "i: ",i,"k: ",k,"maxValue: ", np.max(rescaled[:,:,k]))

# start timing the computation time of the segmentation 
import time
start_time = time.time()
if segType == "gauss":
    #Extract Tissue mask(labeled array) from the normalized using gmm.
    print("Applying Gaussian mixture model to segment the enhanced vessels")
    img3d = ExtractTissueByGMM(opened)
elif segType == "fuzzy":
    print("segmenting with kmeans ")
    img3d = ExtractTissueByfuzzy(opened)  
elif segType == "kmeans":
    print("segmenting with kmeans ")
    img3d = ExtractTissueBykmeans(opened)
    
else:
    import sys
    sys.exit('please indicate the right segmentation type') 
print("--- %s seconds ---" % (time.time() - start_time))


                # Uncomment the below line for Saving the segmentation output to den file format
# print("Saving segmentation output to den file format and with shape: ",img3d.shape )
# saveDENfile("img3d",img3d)
# print("Segmentation output is saved")

#get the label of the tissue using the acquired index i,j,k
vLabel= img3d[iy,ix,z]
print ("vLabel : ",vLabel)

# # Use the label to get the vessels mask.
VesselsMask = img3d == vLabel
VesselsMask = VesselsMask.astype(int)

                # Uncomment the below line for Saving the VesselsMask to den file format
# print("Saving VesselsMask to den file format and with shape: ",VesselsMask.shape )
# saveDENfile("VesselsMask",VesselsMask)
# print("VesselsMask is saved")

#Run erosion operation 
Veroded = morphology.erosion(VesselsMask,morphology.ball(2))

                # Uncomment the below line for Saving the VesselsMask to den file format
# print("Saving the Veroded file to den format")
# saveDENfile("Veroded",Veroded)
# print("The Veroded file is saved")

# Compute connected components labelling algorithm
print("Running the connected components algorithm")
relabeled,labels_out = MyCC3d(Veroded,E_Carm)
print("relabeled labels: {}".format(np.unique(relabeled)))

                # Uncomment the below line for Saving the labled vessel mask to den file format
print("Saving labled vessel mask to den file format and with shape: ",relabeled.shape )
# saveDENfile("relabeled", relabeled)
print("relabeled is saved")

label = relabeled[:,:,z][iy,ix]
print("selected label:", label)
#use the selected label to get the mask of the required vessel
selectedVessel = relabeled == label

#Run dilation operation 
Vdilated = morphology.dilation(selectedVessel,morphology.ball(2))

                # Uncomment the below line for Saving the VesselsMask to den file format
print("Saving the selected region file to den format")
selectedVessel.shape
saveDENfile("selectedVessel",Vdilated)
print("The the selected region file is saved")
 
# Draw 10 Px diameter disk to get the inside Pxs coordinates
selectedFrame = Vdilated[:,:,z].copy()
R=10
mask = drawpoly(selectedFrame,iy,ix,rad = R)

# Draw the cirle for the user visualization
points = circle_points(500, [iy, ix], R)[:-1]

# print(mask.shape)
print("Please ensure your selection area is correct then close the window to contiune")
selectionImg = selectedFrame.copy()
selectionImg[np.where(mask ==0)] = 0

# plt.subplot(1,2,1)
# Initiate the user window of checking the selected vessel 

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
ax[0].imshow(rescaled[:,:,z], cmap="viridis")
ax[1].imshow(selectedFrame, cmap="viridis")
# ax.imshow(selectionImg, cmap='viridis')
# ax.axis('off')
ax[0].plot(points[:, 0], points[:, 1], '--r', lw=2)
ax[1].plot(points[:, 0], points[:, 1], '--r', lw=2)

# fig ,ax = plt.imshow(selectedFrame, cmap="viridis",interpolation='none')
# plt.imshow(mask, alpha=0.7, interpolation='none')
plt.show()

if args["saveLocations"] is not None:
    #Save the coordinates inside the selected area
    locations_values = np.asarray(np.where(mask == 1))
    b = np.full((locations_values.shape[0]+1,locations_values.shape[1]),(z+StrtF+1),dtype=int)
    b[:-1,:] = locations_values
    savetxt(args["saveLocations"],b)
    print("file is saved with the name :", args["saveLocations"] )


 