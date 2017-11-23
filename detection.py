import time
import glob
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import sklearn.svm as svm
from scipy.ndimage.measurements import label
from collections import deque
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

CLASSIFIER_LABEL        = "Classifier"
X_SCALER_LABEL          = "XScaler"
COLORSPACE_LABEL        = "Colorspace"
ORIENT_LABEL            = "Orientation"
PIXELS_PER_CELL_LABEL   = "PixelsPerCell"
CELL_PER_BLOCK_LABEL    = "cellsPerBlock"
SPATIAL_SIZE_LABEL      = "SpatialSize"
HISTOGRAM_BINS_LABEL    = "HistogramBins"
HOG_CHANNEL_LABEL       = "HOGChannel"
SPATIAL_FEAT_LABEL      = "SpatialFeatures"
HISTOGRAM_FEAT_LABEL    = "HistogramFeatures"
HOG_FEAT_LABEL          = "HOGFeatures"
CAR_FEAT                = "ColourFeatures"
NOT_CAR_FEAT            = "NotColourFeatures"
HOG_FEAT                = "HOGCarFeatures"
HOG_NOT_FEAT            = "HOGNotFeatures"

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features #, rhist, ghist, bhist, bin_centers

# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features

# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec, block_norm="L2-Hys")
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec, block_norm="L2-Hys")
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_colour_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 1)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        feature_image = mpimg.imread(img)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            pass
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() to get color histogram features
        hist_features = color_hist(feature_image, hist_bins, hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_HOG_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img_shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, Data): #ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    
    img_tosearch = img[Data.y_start_stop[0]:Data.y_start_stop[1],:,:]
    ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)#convert_color(img_tosearch, conv='RGB2YCrCb')
    #if scale != 1:
    #    imshape = ctrans_tosearch.shape
    #    ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // Data.pix_per_cell) - Data.cell_per_block + 1
    nyblocks = (ch1.shape[0] // Data.pix_per_cell) - Data.cell_per_block + 1 
    nfeat_per_block = Data.orient*Data.cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // Data.pix_per_cell) - Data.cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, Data.orient, Data.pix_per_cell, Data.cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, Data.orient, Data.pix_per_cell, Data.cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, Data.orient, Data.pix_per_cell, Data.cell_per_block, feature_vec=False)
    
    on_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*Data.pix_per_cell
            ytop = ypos*Data.pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=Data.spatial)
            hist_features = color_hist(subimg, nbins=Data.histbin)

            # Scale features and make a prediction
            test_features = Data.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = Data.svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft) #*scale)
                ytop_draw = np.int(ytop) #*scale)
                win_draw = np.int(window) #*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                on_windows.append(((xbox_left, ytop_draw+Data.y_start_stop[0]),(xbox_left+win_draw,ytop_draw+win_draw+Data.y_start_stop[0])))
                #((startx, starty), (endx, endy))
                
    return on_windows #draw_img

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def pipeline(Data, draw_image):
    
    hot_windows_large = search_windows(draw_image, Data.windows_large, Data.svc, Data.X_scaler,
                            color_space=Data.colourspace, 
                            spatial_size=Data.spatial, hist_bins=Data.histbin, 
                            orient=Data.orient, pix_per_cell=Data.pix_per_cell, 
                            cell_per_block=Data.cell_per_block, 
                            hog_channel=Data.hog_channel, spatial_feat=True, 
                            hist_feat=True, hog_feat=True)      
    hot_windows_medium = search_windows(draw_image, Data.windows_medium, Data.svc, Data.X_scaler,
                            color_space=Data.colourspace, 
                            spatial_size=Data.spatial, hist_bins=Data.histbin, 
                            orient=Data.orient, pix_per_cell=Data.pix_per_cell, 
                            cell_per_block=Data.cell_per_block, 
                            hog_channel=Data.hog_channel, spatial_feat=True, 
                            hist_feat=True, hog_feat=True)        
    hot_windows_small = search_windows(draw_image, Data.windows_small, Data.svc, Data.X_scaler,
                            color_space=Data.colourspace, 
                            spatial_size=Data.spatial, hist_bins=Data.histbin, 
                            orient=Data.orient, pix_per_cell=Data.pix_per_cell, 
                            cell_per_block=Data.cell_per_block, 
                            hog_channel=Data.hog_channel, spatial_feat=True, 
                            hist_feat=True, hog_feat=True)

    hot_windows_large = np.array(hot_windows_large)
    hot_windows_medium = np.array(hot_windows_medium)
    hot_windows_small = np.array(hot_windows_small)
    #print(hot_windows_large.shape, hot_windows_medium.shape, hot_windows_small.shape)
    
    #windows = find_cars(draw_image, Data)
    #windows = np.array(windows)

    if sum([x.size>0 for x in [hot_windows_large, hot_windows_medium, hot_windows_small]]) > 0:
    #if windows.size>0:
        box_list = np.concatenate([x for x in [hot_windows_large, hot_windows_medium, hot_windows_small] if x.size > 0])
        #box_list = windows
        #print(box_list.shape)
        heat = np.zeros_like(draw_image[:,:,0]).astype(np.float)

        thresh = 4
        # Add heat to each box in box list
        heat = add_heat(heat,box_list)
        heat = (apply_threshold(heat,thresh)>0).astype(float)
        #plt.figure()
        #plt.title('heat')
        #plt.imshow(np.dstack((heat,heat,heat)).astype(float))
        #plt.show()
        #plt.close()
        allone = np.ones_like(heat).astype(float)
        if Data.hot_count<12:
            #thresh = 4
            Data.hot_count+=1
            #heat = (apply_threshold(heat,thresh)>0).astype(float)
            #plt.figure()
            #plt.title('heat2')
            #plt.imshow(np.dstack((heat,heat,heat)).astype(float))
            #plt.show()
            #plt.close()
            Data.hotmap.append(heat)
        else:
            Data.hotmap.popleft()
            #thresh = 4
            #heat = (apply_threshold(heat,thresh)>0).astype(float)
            Data.hotmap.append(heat)
            for i in Data.hotmap:
                #plt.figure()
                #plt.title('i')
                #plt.imshow(np.dstack((i,i,i)).astype(float))
                #plt.show()
                #plt.close()
                #plt.figure()
                #plt.title('allone')
                #plt.imshow(np.dstack((allone,allone,allone)).astype(float))
                #plt.show()
                #plt.close()
                allone = allone.astype(int) & i.astype(int) #cv2.bitwise_and(allone,test)
                #plt.figure()
                #plt.title('anded')
                #plt.imshow(np.dstack((allone,allone,allone)).astype(float))
                #plt.show()
                #plt.close()
            heat = allone
            #heat = sum(Data.hotmap)
            
        # Apply threshold to help remove false positives
        #heat2 = apply_threshold(heat,thresh)*6
        #heat2 = (heat2>0)*255
        #heat2 = heat*255

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        heatmap = np.dstack((heatmap,heatmap,heatmap))
        #heatmap2 = np.clip(heat2, 0,255)
        #heatmap2 = np.dstack((heatmap2, heatmap2, heatmap2))
        #allone = np.dstack((allone,allone,allone))

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_image = draw_labeled_bboxes(draw_image*255, labels)
    else:
        box_list = None
        draw_image = draw_image*255
        #print('no boxes')
    #window_img = draw_boxes(draw_image*255, hot_windows_large, color=(0, 0, 255), thick=6)
    #window_img = draw_boxes(window_img, hot_windows_medium, color=(0, 0, 255), thick=6) 
    #window_img = draw_boxes(window_img, hot_windows_small, color=(0, 0, 255), thick=6) 

    #print(heatmap.shape,draw_image.shape)

    return draw_image

class data():

    def __init__(self):
        # HOG parameters:
        self.colourspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        # spacial binning parameters:
        self.spatial = (10,10)
        # colour historgram parameters:
        self.histbin = 32
        self.y_start_stop = [400, None]

        self.car_features = None
        self.notcar_features = None 
        self.car_colour_features = None
        self.notcar_colour_features = None
        self.car_HOG_features = None
        self.notcar_HOG_features = None

        self.svc = None
        self.X_scaler = None

        self.windows_large = slide_window([720,1280], x_start_stop=[400, None],
                                         y_start_stop=[400, None], 
                                         xy_window=(250, 150), xy_overlap=(0.75, 0.75))
        self.windows_medium = slide_window([720,1280], x_start_stop=[400, None],
                                         y_start_stop=[400, 550], 
                                         xy_window=(150, 100), xy_overlap=(0.75, 0.75))
        self.windows_small = slide_window([720,1280], x_start_stop=[400, None],
                                         y_start_stop=[400, 500], 
                                         xy_window=(50, 50), xy_overlap=(0.75, 0.75))
        self.hot_count = 0
        self.hotmap = deque([])

    def load(self):

        dataPickle        = pickle.load(open('../classifier2', "rb" ))

        self.svc                    = dataPickle[CLASSIFIER_LABEL]       
        self.X_scaler               = dataPickle[X_SCALER_LABEL]         
        self.colourspace            = dataPickle[COLORSPACE_LABEL]       
        self.orient                 = dataPickle[ORIENT_LABEL] 
        self.pix_per_cell           = dataPickle[PIXELS_PER_CELL_LABEL]
        self.cell_per_block         = dataPickle[CELL_PER_BLOCK_LABEL]  
        self.spatial                = dataPickle[SPATIAL_SIZE_LABEL]     
        self.histbin                = dataPickle[HISTOGRAM_BINS_LABEL]   
        self.hog_channel            = dataPickle[HOG_CHANNEL_LABEL]      
        self.car_features           = dataPickle[CAR_FEAT]               
        self.notcar_features        = dataPickle[NOT_CAR_FEAT]          
        self.car_HOG_features       = dataPickle[HOG_FEAT]               
        self.notcar_HOG_features    = dataPickle[HOG_NOT_FEAT]        

    def save(self):   
        
        dataPickle = {}
        dataPickle[CLASSIFIER_LABEL]      = self.svc
        dataPickle[X_SCALER_LABEL]        = self.X_scaler
        dataPickle[COLORSPACE_LABEL]      = self.colourspace
        dataPickle[ORIENT_LABEL]          = self.orient
        dataPickle[PIXELS_PER_CELL_LABEL] = self.pix_per_cell
        dataPickle[CELL_PER_BLOCK_LABEL]  = self.cell_per_block
        dataPickle[SPATIAL_SIZE_LABEL]    = self.spatial
        dataPickle[HISTOGRAM_BINS_LABEL]  = self.histbin
        dataPickle[HOG_CHANNEL_LABEL]     = self.hog_channel
        dataPickle[CAR_FEAT]              = self.car_features
        dataPickle[NOT_CAR_FEAT]          = self.notcar_features
        dataPickle[HOG_FEAT]              = self.car_HOG_features
        dataPickle[HOG_NOT_FEAT]          = self.notcar_HOG_features

        pickle.dump(dataPickle, open('../classifier2', "wb" ))

if __name__ == "__main__":

    NewFeatures = False
    REPORT_PICTURES = False
    VIDEO = True
    TUNE = False
    
    Data = data()

    if NewFeatures:
        cars = glob.glob('../vehicles/vehicles/**/*.png')
        notcars = glob.glob('../non-vehicles/non-vehicles/**/*.png')
        print("first car image")
        print(cars[1])
        print("first not car image")
        print(notcars[1])
        #sample_size = 500
        #cars = cars[0:sample_size]
        #notcars = notcars[0:sample_size]

        data_info = data_look(cars, notcars)

        print('Your function returned a count of', 
            data_info["n_cars"], ' cars and', 
            data_info["n_notcars"], ' non-cars')
        print('of size: ',data_info["image_shape"], ' and data type:', 
            data_info["data_type"])
        # Just for fun choose random car / not-car indices and plot example images   
        car_ind = np.random.randint(0, len(cars))
        notcar_ind = np.random.randint(0, len(notcars))
            
        # Read in car / not-car images
        car_image = mpimg.imread(cars[car_ind])
        notcar_image = mpimg.imread(notcars[notcar_ind])

        # Plot the examples
        f, ((ax1, ax2)) = plt.subplots(1,2, figsize=(20,10))

        ax1.imshow(car_image)
        ax1.set_title('Example Car Image')
        ax2.imshow(notcar_image)
        ax2.set_title('Example Not-car Image')
        plt.savefig('output_images/examples.png')
        plt.close(f)

        print('Starting feature extraction')

        t1 = time.time()
        
        Data.car_features = extract_features(cars, color_space=Data.colourspace, spatial_size=Data.spatial,
                        hist_bins=Data.histbin, orient=Data.orient, 
                        pix_per_cell=Data.pix_per_cell, cell_per_block=Data.cell_per_block, hog_channel=Data.hog_channel,
                        spatial_feat=True, hist_feat=True, hog_feat=True)
        Data.notcar_features = extract_features(notcars, color_space=Data.colourspace, spatial_size=Data.spatial,
                        hist_bins=Data.histbin, orient=Data.orient, 
                        pix_per_cell=Data.pix_per_cell, cell_per_block=Data.cell_per_block, hog_channel=Data.hog_channel,
                        spatial_feat=True, hist_feat=True, hog_feat=True)
        '''
        Data.car_colour_features = extract_features(cars, cspace='RGB', spatial_size=Data.spatial,
                                hist_bins=Data.histbin, hist_range=(0, 256))
        Data.notcar_colour_features = extract_features(notcars, cspace='RGB', spatial_size=Data.spatial,
                                hist_bins=Data.histbin, hist_range=(0, 256))

        Data.car_HOG_features = extract_HOG_features(cars, cspace=Data.colorspace, orient=Data.orient, 
                            pix_per_cell=Data.pix_per_cell, cell_per_block=Data.cell_per_block, 
                            hog_channel=Data.hog_channel)
        Data.notcar_HOG_features = extract_HOG_features(notcars, cspace=Data.colorspace, orient=Data.orient, 
                            pix_per_cell=Data.pix_per_cell, cell_per_block=Data.cell_per_block, 
                            hog_channel=Data.hog_channel)
        '''

        print('Finished feature extraction')

        Data.save()

        t2 = time.time()

        print('Feature extraction/saving time: {}'.format(t2-t1))

    else:

        Data.load()

    test_imgs = glob.glob('test_images/*.jpg')
    #plt.imshow(test_img)
    #plt.show()

if len(Data.car_features) > 0:
    if Data.svc == None:
        print('Starting training')
        # Create an array stack of feature vectors
        X = np.copy(np.vstack((Data.car_features, Data.notcar_features)))
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        Data.X_scaler = X_scaler
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        # Define a labels vector based on features lists
        y = np.hstack((np.ones(len(Data.car_features)), 
                np.zeros(len(Data.notcar_features))))
        
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        # Use a linear SVC (support vector classifier)
        svc = LinearSVC()
        # Train the SVC
        t1 = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print('Fitting time: {}'.format(t2-t1))
        print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
        print('My SVC predicts: ', svc.predict(X_test[0:10]))
        print('For labels:      ', y_test[0:10])
        Data.svc = svc
        Data.save()
    else:
        svc = Data.svc
        X_scaler = Data.X_scaler

if REPORT_PICTURES:
    for count, img in enumerate(test_imgs):
        draw_image = mpimg.imread(img)

        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        draw_image = draw_image.astype(np.float32)/255

        Data.hot_count = 0
        Data.hotmap = deque([])
        window_img = pipeline(Data, draw_image)/255

        plt.figure()
        plt.imshow(window_img)
        #plt.savefig('output_images/img{}.png'.format(count))
        plt.show()
        plt.close()

if VIDEO:

    Data.hot_count = 0
    Data.hotmap = deque([])

    def process_image(image):
        # NOTE: The output you return should be a color image (3 channel) for processing video below
        # TODO: put your pipeline here,
        # you should return the final output (image where lines are drawn on lanes)
        image = image.astype(np.float32)/255
        result = pipeline(Data, image)

        return result

    first_video = 'project_video_out_good.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    clip1 = VideoFileClip('project_video.mp4') #.subclip(0,1)
    ##clip1 = VideoFileClip('project_video.mp4')
    first_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

    first_clip.write_videofile(first_video, audio=False)

if TUNE:
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]
    svr = svm.SVC()
    clf = GridSearchCV(svr, param_grid)
    clf.fit(X_train, y_train)
    print(clf.best_params_)

