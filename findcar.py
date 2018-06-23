import numpy as np
import imageio
import skimage.io
import cv2
import imageio
import function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
import glob


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,color_space,cells_per_step=3):
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch =function.convert_color(img_tosearch, conv=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1 #how many steps the window to step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1 #cells_per_step equals to blocks per step

    # Compute individual channel HOG features for the entire image
    hog1 = function.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = function.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = function.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    boxlists=[]
    scores = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = function.bin_spatial(subimg, size=spatial_size)
            hist_features = function.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            a = np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1)
            test_features = X_scaler.transform(a)
            test_prediction = svc.predict(test_features)
            score = svc.decision_function(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxlists.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                scores.append(score)
    return boxlists,scores


def process_image(img):
    scales = [[400,500,1,2],[400,500,1.3,2],[410,500,1.4,2],
            [420,556,1.6,2],[430,556,1.8,2],[430,556,2,2],[440,556,1.9,2],
            [440,556,1.3,2],[440,556,2.2,2],[500,656,3.0,2]]
    # scales = [[400,650,1.3,2],[400,650,1.8,2]]
    # scales = [[400,470,1,2],[400,500,1.5,2],[410,520,1.7,2],[420,500,1.8,2]]
    # scales = [[400,500,1.5,2]]

    boxlist = []
    score = []
    for scale in scales:
        boxlist_single,score_single = find_cars(img, scale[0], scale[1], scale[2], svc,
                X_scaler, orient, pix_per_cell, cell_per_block,
                spatial_size, hist_bins,color_space, scale[3])
        boxlist.extend(boxlist_single)
        score.extend(score_single)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = function.add_heat(heat,boxlist)
    # heat = function.add_score(heat,boxlist,score)



    # Apply threshold to help remove false positives
    heat = function.apply_threshold(heat,6)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    if len(history_heats)<10:
        history_heats.append(heatmap)
    else:
        history_heats.pop(0)
        history_heats.append(heatmap)

    # weight = [0.1,0.2,0.3,0.4]
    # for i in range(len(history_heats)):
        # history_heats[i] = history_heats[i]*weight[i]

    heatmap = np.sum(np.array(history_heats), axis=0)
    # heatmap = np.zeros_like(heatmap).astype(np.float)
    heatmap = function.apply_threshold(heatmap,20)



    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = function.draw_labeled_bboxes(np.copy(img), labels)


    # r, g, b = cv2.split(draw_img)
    # merged = cv2.merge([b,g,r])
    return draw_img

from moviepy.editor import VideoFileClip
if __name__ == "__main__":
    data = joblib.load('dump_filename')
    svc = data['classifier']
    X_scaler = data['scaler']
    color_space = data['color_space']
    orient = data['orient']

    pix_per_cell = data['pix_per_cell']
    cell_per_block = data['cell_per_block']
    spatial_size = data['spatial_size']
    hist_bins = data['hist_bins']
    print('----svm loaded done----')
    print('color_space:',color_space)
    print('orient:',orient)
    print('pix_per_cell:',pix_per_cell)
    print('cell_per_block:',cell_per_block)
    print('spatial_size:',spatial_size)
    print('hist_bins:',hist_bins)

    color_space = function.convert_format('RGB',color_space)
    filename = 'project_video'
    # filename = 'test_video'
    if 0:

        vid = imageio.get_reader(filename+'.mp4',  'ffmpeg')

        #store history heatmaps to filter the false positive
        history_heats = []
        for i,Img in enumerate(vid):
            img = skimage.img_as_ubyte(Img, True)

            draw_img =  process_image(img)

            r, g, b = cv2.split(draw_img)
            merged = cv2.merge([b,g,r])
            cv2.imshow('window',merged)
            cv2.waitKey(10)
        cv2.destroyAllWindows()
    else:

        history_heats = []
        video_output1 = filename+'_output_1.mp4'
        clip3 = VideoFileClip(filename+'.mp4')
        project_clip = clip3.fl_image(process_image)
        project_clip.write_videofile(video_output1, audio=False)

