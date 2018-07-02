## Lucas-Kanade Tracking
The original (non Pyramidal) tracker with fixed window (its size doesnt update) is implemented.  

To run tracking type in Terminal: python3 lucas_kanade_tracking.py [--roi roi] [--dpath path_to_images]  
The process will start. After the new region of interest was found, it is printed in console.  
To stop tracking press Ctrl+C.  


Was tested on 3 datasets from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html   
The results of tracking can be copared with groundtruth_rect.txt whihc is provided in archive downloaded by the link.  

1. Coke  
 Run with: python3 lucas_kanade_tracking.py --roi 298 160 48 80 --dpath 'Coke/img/'  
 The tracker should track the can, but in 6 iterations after the start finds a hand and tracks the hand.   

2. DragonBaby  
    Run with: python3 lucas_kanade_tracking.py --roi 160 83 56 65 --dpath 'DragonBaby/img/'  
    Completely misses a baby when it starts moving fast.  

3. 