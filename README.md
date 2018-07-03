## Lucas-Kanade Tracking
The original (non Pyramidal) tracker with fixed window (its size doesnt update) is implemented.  

To run Lucas Kanade tracking type in Terminal: python3 lucas_kanade_tracking.py [--roi roi] [--dpath path_to_images]  
The process will start. After the new region of interest was found, it is printed in console.  
To stop tracking press Ctrl+C.  


Was tested on 3 datasets from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html   
The results of tracking can be compared with groundtruth_rect.txt which is provided in archive downloaded by the link.  

1. Coke  
 Run with: python3 lucas_kanade_tracking.py --roi 298 160 48 80 --dpath 'Coke/img/'  
 The tracker should track the can, but in 6 iterations after the start finds a hand and tracks the hand.   

2. DragonBaby  
    Run with: python3 lucas_kanade_tracking.py --roi 160 83 56 65 --dpath 'DragonBaby/img/'
    Completely misses a baby when it starts moving fast.

3. Football  
    Run with: python3 lucas_kanade_tracking.py --roi 310 102 39 50 --dpath 'Football/img/'  
    On this dataset tracker performs the best because the region of interest changes a little on each iteration.
    
    
## Meanshift tracking
The original meanshift tracker is implemented and compared with OpenCV Meanshift and Camshift.

To run Meanshift tracking type in Terminal: python3 mean_shift.py [--roi roi] [--dpath path_to_images]
The process will start. Three tracking windows are shown on image:  
    - Red - custom Meanshift  
    - Blue - OpenCV Meanshift  
    - Green - OpenCV Camshift  

Was tested on 2 datasets from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html   
The methods don't work properly with small window, so the window should be larger.
To track the process in details change the last line in show_image_with_rect function to: cv2.waitKey(0)
The results of tracking can be copared with groundtruth_rect.txt which is provided in archive downloaded by the link.  

1. Coke  
    Run with: python3 mean_shift.py --roi 270 160 80 80 --dpath 'Coke/img/'  
    (the results won't match with the groundtruth but with parameters from groundtruth_rect.txt trackers miss the can from the very beginning)
    All trackers cope with the task of can tracking.    

2. DragonBaby  
    Run with: python3 mean_shift.py --roi 160 83 100 100 --dpath 'DragonBaby/img/'
    As movements are fast CamShift window becomes large on the erty beginning and remains same throughut all sequence of images.
    Custom MeanShift goes out of window soon and the process interupts.

**Note**: 
Custom mean shift sometimes fails (centroid receives NaN coordinates; I wish I had time to investigate this). In this case program crashes with exception.
  
- for Coke dataset program doesn't crash when centroids are initialized as 
centroid = np.zeros(2)

But in this case meanshift converges much slower (up to 20 iterations) than it can.

- for DragonBaby dataset program doesn't crash when centroids are initialized as 
centroid = get_central_point(input_roi_box)

So we take the current central point of region of interest and meanshift coverges in up to 3 iterations on each step.