# import the necessary packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from denpy import DEN
from sklearn.mixture import GaussianMixture as gmm
import cc3d
from skimage import segmentation,measure,morphology
from gekko import GEKKO

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description="extract manually the desired vessel and show the attenuation curve")
ap.add_argument("-f", "--file", required=True,	help="Path to den file")
ap.add_argument("-fFrom", "--frameFrom", required=True,	help="indicate the starting frame number")
ap.add_argument( "-fTo" , "--frameTo", required=True,	help="indicate the last frame number")
args = vars(ap.parse_args())
# display a friendly message to the user
print("Thank you for uploading the {} file ".format(args["file"] )) 
print("from frame No. {}".format(args["frameFrom"]))
print( "to frame No. {}".format(args["frameTo"]))

StrtF = np.int(args["frameFrom"])
EndF = np.int(args["frameTo"])

# Define the necessary functions
def getTheFile(filename,rows,cols,start,stop):
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

#Getting the 3D-array and 2D-TestingFrame for CT 
print("File is loading ")
Carm = getTheFile(args["file"],550,550,StrtF,EndF)
print("File is loaded with the shape: ",Carm.shape)
#Normalize the data  
print("Normalizing the data") 
normalized = (Carm-Carm.min())/(Carm.max()-Carm.min())

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
print("Running the connected components algorithm")
#Run connected component labelling to get the mask of the required vessel
labels_out = cc3d.connected_components(VesselsMask) # connectivity = 26 
#Remove small objects
interior_labels = morphology.remove_small_objects(labels_out, min_size=400)
#relabel and print labels
relabeled, _, _ = segmentation.relabel_sequential(interior_labels)

print("relabeled labels: {}".format(np.unique(relabeled)))
#get the region properties(for instance area)
# regionprops = measure.regionprops(relabeled, intensity_image=Carm[:,:,StrtF:EndF])

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

