# import the necessary packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from denpy import DEN
import argparse
from sklearn.mixture import GaussianMixture as gmm
import cc3d
from skimage import segmentation,measure,morphology
from gekko import GEKKO

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description="extract manually the desired vessel and show the attenuation curve")
ap.add_argument("-f", "--file", required=True,	help="Path to den file")
ap.add_argument("-s", "--segType", required=False,	help="indicate the segmentation type(gauss or fuzzy or kmeans)")
ap.add_argument("-fFrom", "--frameFrom", required=True,	help="indicate the starting frame number")
ap.add_argument( "-fTo" , "--frameTo", required=True,	help="indicate the last frame number")
args = vars(ap.parse_args())
# display a friendly message to the user
print("Thank you for uploading the {} file ".format(args["file"] )) 
print("from frame No. {}".format(args["frameFrom"]))
print( "to frame No. {}".format(args["frameTo"]))
print( "segmentaing the input images with. {}".format(args["segType"]))

#saving the startFrame and endFrame 
StrtF = np.int(args["frameFrom"])
EndF = np.int(args["frameTo"])
segType = args["segType"]
# Define the necessary functions
def getTheFile(filename,rows,cols,start,stop):
    # This function is used to import the den file and convert it to numpy array
    arr = np.empty([rows, cols, 0])    
       
    for i in range(start,stop,1):
        Frame = DEN.getFrame(filename, i, row_from=None, row_to=None, col_from=None, col_to=None)
        dentoarr = np.asarray(Frame)
        dentoarr = np.expand_dims(dentoarr, axis=2)
        arr = np.concatenate((arr, dentoarr), axis=2)
    return arr

def ExtractTissueByGMM(img3d):
    arr = np.empty([550, 550, 0]) 
    orginalShape =img3d.shape
    reshaped = img3d.reshape(-1,1)
        
    clustersNumber = 4
    gmm_model_CTa= gmm(n_components = clustersNumber, covariance_type='tied').fit(reshaped)
    gmm_labels =gmm_model_CTa.predict(reshaped)
    segmented_frames= gmm_labels.reshape(orginalShape[0],orginalShape[1],orginalShape[2])
    arr = np.concatenate((arr, segmented_frames), axis=2)
    arr = arr.astype(int)
    return arr

def MyCC3d(labeledarr,originalarr):
    import cc3d
    from skimage import segmentation,measure,morphology
    print("computing cc3d")
    labels_out = cc3d.connected_components(labeledarr)
    print("removing small size objects to decrease the output labels")
    interior_labels = morphology.remove_small_objects(labels_out, min_size=300)
    print("relabelling")
    relabeled, _, _ = segmentation.relabel_sequential(interior_labels)
    print("computing the objects properties")
    regionprops = measure.regionprops(relabeled, intensity_image=originalarr)
    print("Computing the areas")
    areas = [regionprop.area for regionprop in regionprops]
    return relabeled,labels_out

# Simple mouse click function to store coordinates
def onclick(event):   
    global ix, iy
    ix, iy = event.xdata, event.ydata
    global label
    label.append((ix, iy))
    ix=int(ix)
    iy=int(iy)
    print(iy,ix)
    label = labels_out[:,:,0][iy,ix]
    print(label)
    return label
# Import FCM Class
import FCM

def getCentroids(volume,StrtF):
    for f in range(volume.shape[2]):
        L_ICA_mask_int = np.asarray(volume[:,:,f],dtype=int)
        L_ICA_prop = measure.regionprops(L_ICA_mask_int)
        if (f == 0):
            # print (f)
            extended_centroid = L_ICA_prop[0].centroid
            extended_centroid = extended_centroid+tuple([f])
        else:
            # print(f)
            centroid = L_ICA_prop[0].centroid  
            centroid = centroid+tuple([f])
            extended_centroid= np.dstack((extended_centroid, centroid))
        # print(extended_centroid)
    ifx = np.asarray(extended_centroid[0][0],int)
    ify = np.asarray(extended_centroid[0][1],int)
    ifz = np.asarray(extended_centroid[0][2],int)
    ifz=ifz+StrtF
    arr=[]
    arr = np.concatenate((arr, ifx), axis=0)
    arr = np.expand_dims(arr, axis=1)
    ify = np.expand_dims(ify, axis=1)
    arr = np.concatenate((arr, ify), axis=1)
    ifz = np.expand_dims(ifz, axis=1)
    arr = np.concatenate((arr, ifz), axis=1)
    arr= np.asarray(arr,dtype=int)
    return arr  

def savetxt(arrName,arr):
    from numpy import savetxt
    savetxt(arrName, arr, delimiter=',')


#Getting the 3D-array and 2D-TestingFrame for CT 
print("File is loading ")
Carm = getTheFile(args["file"],550,550,StrtF,EndF)
print("File is loaded with the shape: ",Carm.shape)
#Normalize the data  
print("Normalizing the data") 
normalized = (Carm-Carm.min())/(Carm.max()-Carm.min())

if segType == "gauss":
    #Get the index of the highest normalized value
    j,i = np.unique(np.where(normalized[:,:,0] == np.max(normalized[:,:,0])))

    #Extract Tissue mask(labeled array) from the normalized using gmm.
    print("Applying Gaussian mixture model to segment the enhanced vessels")
    img3d = ExtractTissueByGMM(normalized)

    #get the label of the tissue using the acquired index i,j 
    target = img3d[:,:,0]
    vLabel= target[j,i]

    # Use the label to get the vessels mask.
    VesselsMask = img3d == vLabel
    VesselsMask = VesselsMask.astype(int)

    #Check if the label is correct
    # plt.imshow(VesselsMask[:,:,0])
    # plt.show()

    # '''
    # #The user must select the frames which have the desired vessels
    # #RVertebral arterial = 91-107-119
    # '''
elif segType == "fuzzy":
    print("segmenting with fuzzy ")
    original_shape = normalized.shape
    reshaped  = normalized.reshape(-1,1)
    import time
    cluster = FCM.FCM(reshaped,3,m=2,epsilon=0.05,max_iter=100,kernel_shape='uniform', kernel_size=9)
    print (reshaped.shape)
    cluster.form_clusters()
    cluster.calculate_scores()
    result=cluster.result
    # print("--- %s seconds ---" % (time.time() - start_time))
    VesselsMask = result.reshape(original_shape[0],original_shape[1],original_shape[2])   
elif segType == "kmeans":
    print(kmeans)
    sys.exit('later will be add')
else:
    import sys
    sys.exit('please indicate the right segmentation type') 

print("Running the connected components algorithm")
relabeled,labels_out = MyCC3d(VesselsMask,normalized)
print("relabeled labels: {}".format(np.unique(relabeled)))
    
#Define user selection label as global label
label=[]
print("Please select the placement location for the arterial input function then close the window")

#Run the user window for selecting the label of the desired vessel 
labeled_frame = labels_out[:,:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(labeled_frame, cmap="viridis")
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

#use the selected label to get the mask of the required vessel
selectedVessel = labels_out == label
print("Please ensure your selection is right then close the window to contiune")
plt.imshow(selectedVessel[:,:,0])
plt.show()
    
# convert bool to int if needed
    
#get the orginal values
vessel = selectedVessel * Carm

# Get the centroid from the mask
arr= getCentroids(selectedVessel,StrtF)
# Save the locations of the centroids
savetxt('arrName',arr)
#calculate the average of from each frame
print("calculating the average from each frame and drawing the time attenuation curve")
yvalues =[]
xvalues = range(vessel.shape[2])
for i in range(vessel.shape[2]):
    avg = np.average(vessel[:,:,i]!= 0)
    yvalues.append(avg)

# Draw and fit cubic line
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xvalues, yvalues)
# plt.show()

xvalues = np.linspace(0,Carm.shape[2]-1,num=Carm.shape[2])
print("Fitting spline curve to the acquired points and predicting the maximum")
#GEKKO model
m=GEKKO()
#Parameter
mxspline = m.Param(value=np.linspace(0,Carm.shape[2]-1))
#Variable
myspline = m.Var()
m.cspline(mxspline,myspline,xvalues,yvalues)
#Regression Mode
m.options.IMODE = 2
#Optimize
m.solve(disp=False)

p=GEKKO()
px=p.Var(value=1 ,lb=10,ub=16)
py=p.Var()
p.Obj(-py)
p.cspline(px,py,xvalues,yvalues)
p.solve(disp=False,GUI=False)
print(" Showing the time attenuation curve, close to continue")
fig1 = plt.gcf()
plt.title('Time Attenuation Curve')
plt.plot(xvalues,yvalues,'go',label='data')
plt.plot(mxspline,myspline,'r--',label='cubic spline')
plt.plot(px,py,'bo',label='maximum')
plt.ylabel('Normalizaed values')
plt.xlabel('Time (Sec)')
fig1.savefig("selectedVessel.pdf")
fig1.savefig("selectedVessel.png")
print(" The time attenuation curve as pdf and png are saved in your hard disc")
plt.show()
print("Maximum concentration =", str(py.VALUE))
print("Time to peak =", str(px.VALUE))
from numpy import trapz
# Compute the area using the composite trapezoidal rule.
area = trapz(myspline)
print("area =", area)


    
# file = "E:/02.Python_Projects/01.BrainSegmentaion/KVA_carm.den"

# h = [(f[j] + g[j])/2 for j in range(len(x))]