import pickle
import glob
import function
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
def read_trainning_imgnames():
    vehicle_folders = ['GTI_Far/','GTI_Left/','GTI_MiddleClose/','GTI_Right/']
    none_vehicle_folders = ['GTI/']
    car_names = []
    no_car_names = []
    for vehicle_folder in vehicle_folders:
        images = glob.glob('vehicles/vehicles/'+vehicle_folder+'image*.png')
        car_names.extend(images)
    for none_vehicle_folder in none_vehicle_folders:
        images = glob.glob('non-vehicles/non-vehicles/'+none_vehicle_folder+'image*.png')
        no_car_names.extend(images)
    images = glob.glob('vehicles/vehicles/KITTI_extracted/*.png')
    car_names.extend(images)
    images = glob.glob('non-vehicles/non-vehicles/Extras/extra*.png')
    no_car_names.extend(images)
    return car_names,no_car_names

def train_classifier(car_names,no_car_names):
    color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 15  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [None, None] # Min and max in y to search in slide_window()

    car_features = function.extract_features(car_names, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = function.extract_features(no_car_names, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features,notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)),np.zeros(len(notcar_features))))
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler only on the training data
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to both X_train and X_test
    scaled_X_train = X_scaler.transform(X_train)
    scaled_X_test = X_scaler.transform(X_test)

    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    svc.fit(scaled_X_train, y_train)
    print('Test Accuracy of SVC = ', svc.score(scaled_X_test, y_test))
    print('My SVC predicts: ', svc.predict(X_test[0:10]))
    print('For these',10, 'labels: ', y_test[0:10])
    return svc,X_scaler,color_space,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins

from sklearn.externals import joblib
if __name__ == "__main__":
    car_names,no_car_names = read_trainning_imgnames()
    svc,X_scaler,color_space,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins = train_classifier(car_names,no_car_names)
    print('----svm trained done----')
    dd = {'classifier': svc, 'scaler': X_scaler,'color_space':color_space,
          'orient':orient,'pix_per_cell':pix_per_cell,'cell_per_block':cell_per_block,
          'spatial_size':spatial_size,'hist_bins':hist_bins}
    joblib.dump(dd, 'dump_filename_1')

