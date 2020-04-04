
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:14:19 2020

@author: taiki
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:30:39 2020

@author: taiki
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:18:05 2020

@author: taiki
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

name = 'Sakai'
No = 9
syn = 'syn1'
D_frames2 = []
D_fig2 = plt.figure()
D_ax2 = Axes3D(D_fig2)

#%%
#import angles
#dat_angles = np.genfromtxt('angles_self_' + name + str(No) + '.csv', skip_header = 1, delimiter = ',')
dat_angles = np.genfromtxt('SVD_jointangles_self_' + name + str(No) + '_' + syn + '.csv', skip_header = 1, delimiter = ',')

for i in range(0,len(dat_angles[:,0])):#len(dat_angles[:,0])
#%%
# position of static pose

# trunk
    S_IJ = np.array([-22,19,145])
    S_PX = np.array([-22,19,130])
    S_C7 = np.array([-22,4,145])
    S_T7 = np.array([-22,7,130])
    S_R_shoulder = np.array([3,6,155])
    S_L_shoulder = np.array([-47,6,155])
    S_M_shoulder = np.array((S_R_shoulder + S_R_shoulder) / 2)
    S_M_C7IJ = np.array((S_IJ+ S_C7) / 2)
    S_M_T7PX = np.array((S_PX+ S_T7) / 2)
    
    # pelvis
    S_R_ASIS = np.array([-7,16,100])
    S_L_ASIS = np.array([-37,16,100])
    S_M_PSIS = np.array([-22,1,105])
    S_R_GT = np.array([3,6,90])
    S_L_GT = np.array([-47,6,90])
    S_M_ASIS = np.array((S_R_ASIS + S_L_ASIS) / 2)
    
    # R_upper_arm
    S_R_elbow_lat = np.array([38,3,150])
    S_R_elbow_med = np.array([38,3,148])
    S_M_R_elbow = np.array((S_R_elbow_lat + S_R_elbow_med) / 2)
    
    # L_upper_arm
    S_L_elbow_lat = np.array([-85,3,150])
    S_L_elbow_med = np.array([-85,3,148])
    S_M_L_elbow = np.array((S_L_elbow_lat + S_L_elbow_med) / 2)
    
    # R_forearm
    S_R_wrist_lat = np.array([63,0,154])
    S_R_wrist_med = np.array([63,0,148])
    S_M_R_wrist = np.array((S_R_wrist_lat + S_R_wrist_med) / 2)
    
    # L_forearm
    S_L_wrist_lat = np.array([-110,0,154])
    S_L_wrist_med = np.array([-110,0,148])
    S_M_L_wrist = np.array((S_L_wrist_lat + S_L_wrist_med) / 2)
    
    # R_hand
    S_R_MC3 = np.array([73,-3,151])
    
    # L_hand
    S_L_MC3 = np.array([-120,-3,151])
    
    # R_thigh
    S_R_knee_lat = np.array([3,6,50])
    S_R_knee_med = np.array([-3,6,50])
    S_M_R_knee = np.array((S_R_knee_lat + S_R_knee_med) / 2)
    
    # R_ankle
    S_R_ankle_lat = np.array([3,6,10])
    S_R_ankle_med = np.array([-3,6,10])
    S_M_R_ankle = np.array((S_R_ankle_lat + S_R_ankle_med) / 2)
    
    # L_thigh
    S_L_knee_lat = np.array([-47,6,50])
    S_L_knee_med = np.array([-41,6,50])
    S_M_L_knee = np.array((S_L_knee_lat + S_L_knee_med) / 2)
    
    # L_ankle
    S_L_ankle_lat = np.array([-47,6,10])
    S_L_ankle_med = np.array([-41,6,10])
    S_M_L_ankle = np.array((S_L_ankle_lat + S_L_ankle_med) / 2)
    
    R_leg_lat_length = np.linalg.norm((S_R_GT - S_R_knee_lat))
    R_leg_med_length = np.linalg.norm((S_R_GT - S_R_knee_med))
    L_leg_lat_length = np.linalg.norm((S_L_GT - S_L_knee_lat))
    L_leg_med_length = np.linalg.norm((S_L_GT - S_L_knee_med))
    
    # R_foot
    S_R_toe = np.array([0,27,0])
    S_R_heel = np.array([0,0,0])
    
    # L_foot
    S_L_toe = np.array([-44,27,0])
    S_L_heel = np.array([-44,0,0])
    
    R_HC1 = (2/3 * S_R_GT) + (1/3 * S_R_ASIS)
    R_HC2 = (0.18 * S_L_ASIS) + (0.82 * S_R_ASIS)
    R_HC3 = (R_HC2 - S_R_ASIS)
    S_R_HC =  R_HC1 + R_HC3
    
    L_HC1 = (2/3 * S_L_GT) + (1/3 * S_L_ASIS)
    L_HC2 = (0.18 * S_R_ASIS) + (0.82 * S_L_ASIS)
    L_HC3 = (L_HC2 - S_L_ASIS)
    S_L_HC =  L_HC1 + L_HC3
    
    #%%
    # Static LCS
    # pelvis LCS
    S_pelvis_x = (S_R_ASIS - S_L_ASIS) / np.linalg.norm((S_R_ASIS - S_L_ASIS), ord=2)
    S_pelvis_1 = (S_M_ASIS - S_M_PSIS) / np.linalg.norm((S_M_ASIS - S_M_PSIS), ord=2)
    S_pelvis_z = np.cross(S_pelvis_x,S_pelvis_1) / np.linalg.norm(np.cross(S_pelvis_x,S_pelvis_1), ord=2)
    S_pelvis_y = np.cross(S_pelvis_z,S_pelvis_x) / np.linalg.norm(np.cross(S_pelvis_z,S_pelvis_x), ord=2)
    
    S_LCS_pelvis = np.array([S_pelvis_x, S_pelvis_y, S_pelvis_z])
    S_Veri_pelvis = np.linalg.det(S_LCS_pelvis)
    
    # trunk LCS
    S_trunk_y = (S_PX - S_T7) / np.linalg.norm((S_PX - S_T7), ord=2)
    S_trunk_1 = (S_M_C7IJ - S_M_T7PX) / np.linalg.norm((S_M_C7IJ - S_M_T7PX), ord=2)
    S_trunk_x = np.cross(S_trunk_y,S_trunk_1) / np.linalg.norm(np.cross(S_trunk_y,S_trunk_1), ord=2)
    S_trunk_z = np.cross(S_trunk_x,S_trunk_y) / np.linalg.norm(np.cross(S_trunk_x,S_trunk_y), ord=2)
        
    S_LCS_trunk = np.array([S_trunk_x,S_trunk_y,S_trunk_z])
    S_Veri_trunk = np.linalg.det(S_LCS_trunk)
    
    # L_upper_arm LCS
    S_L_upper_z = (S_L_elbow_lat - S_L_elbow_med) / np.linalg.norm((S_L_elbow_lat - S_L_elbow_med), ord=2)
    S_L_upper_1 = (S_L_shoulder - S_M_L_elbow) / np.linalg.norm((S_L_shoulder - S_M_L_elbow), ord=2)
    S_L_upper_y = np.cross(S_L_upper_z,S_L_upper_1) / np.linalg.norm(np.cross(S_L_upper_z,S_L_upper_1), ord=2)
    S_L_upper_x = np.cross(S_L_upper_y,S_L_upper_z) / np.linalg.norm(np.cross(S_L_upper_y,S_L_upper_z), ord=2)
        
    S_LCS_L_upper = np.array([S_L_upper_x,S_L_upper_y,S_L_upper_z])
    S_Veri_L_upper = np.linalg.det(S_LCS_L_upper)
        
    # R_upper_arm LCS
    S_R_upper_z = (S_R_elbow_lat - S_R_elbow_med) / np.linalg.norm((S_R_elbow_lat - S_R_elbow_med), ord=2)
    S_R_upper_1 = (S_R_shoulder - S_M_R_elbow) / np.linalg.norm((S_R_shoulder - S_M_R_elbow), ord=2)
    S_R_upper_y = np.cross(S_R_upper_1,S_R_upper_z) / np.linalg.norm(np.cross(S_R_upper_1,S_R_upper_z), ord=2)
    S_R_upper_x = np.cross(S_R_upper_y,S_R_upper_z) / np.linalg.norm(np.cross(S_R_upper_y,S_R_upper_z), ord=2)
        
    S_LCS_R_upper = np.array([S_R_upper_x,S_R_upper_y,S_R_upper_z])
    S_Veri_R_upper = np.linalg.det(S_LCS_R_upper)
    
    # L_lower_arm LCS
    S_L_lower_z = (S_L_wrist_lat - S_L_wrist_med) / np.linalg.norm((S_L_wrist_lat - S_L_wrist_med), ord=2)
    S_L_lower_1 = (S_M_L_elbow - S_M_L_wrist) / np.linalg.norm((S_M_L_elbow - S_M_L_wrist), ord=2)
    S_L_lower_y = np.cross(S_L_lower_z,S_L_lower_1) / np.linalg.norm(np.cross(S_L_lower_z,S_L_lower_1), ord=2)
    S_L_lower_x = np.cross(S_L_lower_y,S_L_lower_z) / np.linalg.norm(np.cross(S_L_lower_y,S_L_lower_z), ord=2)
        
    S_LCS_L_lower = np.array([S_L_lower_x,S_L_lower_y,S_L_lower_z])
    S_Veri_L_lower = np.linalg.det(S_LCS_L_lower)
    
    # R_lower_arm LCS
    S_R_lower_z = (S_R_wrist_lat - S_R_wrist_med) / np.linalg.norm((S_R_wrist_lat - S_R_wrist_med), ord=2)
    S_R_lower_1 = (S_M_R_elbow - S_M_R_wrist) / np.linalg.norm((S_M_R_elbow - S_M_R_wrist), ord=2)
    S_R_lower_y = np.cross(S_R_lower_1,S_R_lower_z) / np.linalg.norm(np.cross(S_R_lower_1,S_R_lower_z), ord=2)
    S_R_lower_x = np.cross(S_R_lower_y,S_R_lower_z) / np.linalg.norm(np.cross(S_R_lower_y,S_R_lower_z), ord=2)
        
    S_LCS_R_lower = np.array([S_R_lower_x,S_R_lower_y,S_R_lower_z])
    S_Veri_R_lower = np.linalg.det(S_LCS_R_lower)
    
    # L_hand LCS
    S_L_hand_x = (S_M_L_wrist - S_L_MC3) / np.linalg.norm((S_M_L_wrist - S_L_MC3), ord=2)
    S_L_hand_y = np.cross(S_L_lower_z,S_L_hand_x) / np.linalg.norm(np.cross(S_L_lower_z,S_L_hand_x), ord=2)
    S_L_hand_z = np.cross(S_L_hand_x,S_L_hand_y) / np.linalg.norm(np.cross(S_L_hand_x,S_L_hand_y), ord=2)
        
    S_LCS_L_hand = np.array([S_L_hand_x,S_L_hand_y,S_L_hand_z])
    S_Veri_L_hand = np.linalg.det(S_LCS_L_hand)
        
    # R_hand LCS 
    S_R_hand_x = (S_R_MC3 - S_M_R_wrist) / np.linalg.norm((S_M_R_wrist - S_R_MC3), ord=2)
    S_R_hand_y = np.cross(S_R_lower_z,S_R_hand_x) / np.linalg.norm(np.cross(S_R_lower_z,S_R_hand_x), ord=2)
    S_R_hand_z = np.cross(S_R_hand_x,S_R_hand_y) / np.linalg.norm(np.cross(S_R_hand_x,S_R_hand_y), ord=2)
    
    S_LCS_R_hand = np.array([S_R_hand_x,S_R_hand_y,S_R_hand_z])
    S_Veri_R_hand = np.linalg.det(S_LCS_R_hand)
        
    # L_thigh LCS
    S_L_thigh_x = (S_L_knee_med - S_L_knee_lat) / np.linalg.norm((S_L_knee_med - S_L_knee_lat), ord=2)
    S_L_thigh_1 = (S_L_GT - S_M_L_knee) / np.linalg.norm((S_L_GT - S_M_L_knee), ord=2)
    S_L_thigh_y = np.cross(S_L_thigh_1,S_L_thigh_x) / np.linalg.norm(np.cross(S_L_thigh_1,S_L_thigh_x), ord=2)
    S_L_thigh_z = np.cross(S_L_thigh_x,S_L_thigh_y) / np.linalg.norm(np.cross(S_L_thigh_x,S_L_thigh_y), ord=2)
        
    S_LCS_L_thigh = np.array([S_L_thigh_x,S_L_thigh_y,S_L_thigh_z])
    S_Veri_L_thigh = np.linalg.det(S_LCS_L_thigh)
        
    # R_thigh LCS
    S_R_thigh_x = (S_R_knee_lat - S_R_knee_med) / np.linalg.norm((S_R_knee_lat - S_R_knee_med), ord=2)
    S_R_thigh_1 = (S_R_GT - S_M_R_knee) / np.linalg.norm((S_R_GT - S_M_R_knee), ord=2)
    S_R_thigh_y = np.cross(S_R_thigh_1,S_R_thigh_x) / np.linalg.norm(np.cross(S_R_thigh_1,S_R_thigh_x), ord=2)
    S_R_thigh_z = np.cross(S_R_thigh_x,S_R_thigh_y) / np.linalg.norm(np.cross(S_R_thigh_x,S_R_thigh_y), ord=2)
        
    S_LCS_R_thigh = np.array([S_R_thigh_x,S_R_thigh_y,S_R_thigh_z])
    S_Veri_R_thigh = np.linalg.det(S_LCS_R_thigh)
        
    # L_leg LCS
    S_L_leg_x = (S_L_ankle_med - S_L_ankle_lat) / np.linalg.norm((S_L_ankle_med - S_L_ankle_lat), ord=2)
    S_L_leg_1 = (S_M_L_knee - S_M_L_ankle) / np.linalg.norm((S_M_L_knee - S_M_L_ankle), ord=2)
    S_L_leg_y = np.cross(S_L_leg_1,S_L_leg_x) / np.linalg.norm(np.cross(S_L_leg_1,S_L_leg_x), ord=2)
    S_L_leg_z = np.cross(S_L_leg_x,S_L_leg_y) / np.linalg.norm(np.cross(S_L_leg_x,S_L_leg_y), ord=2)
    S_LCS_L_leg = np.array([S_L_leg_x,S_L_leg_y,S_L_leg_z])
    S_Veri_L_leg = np.linalg.det(S_LCS_L_leg)
        
    # R_leg LCS
    S_R_leg_x = (S_R_ankle_lat - S_R_ankle_med) / np.linalg.norm((S_R_ankle_lat - S_R_ankle_med), ord=2)
    S_R_leg_1 = (S_M_R_knee - S_M_R_ankle) / np.linalg.norm((S_M_R_knee - S_M_R_ankle), ord=2)
    S_R_leg_y = np.cross(S_R_leg_1,S_R_leg_x) / np.linalg.norm(np.cross(S_R_leg_1,S_R_leg_x), ord=2)
    S_R_leg_z = np.cross(S_R_leg_x,S_R_leg_y) / np.linalg.norm(np.cross(S_R_leg_x,S_R_leg_y), ord=2)
        
    S_LCS_R_leg = np.array([S_R_leg_x,S_R_leg_y,S_R_leg_z])
    S_Veri_R_leg = np.linalg.det(S_LCS_R_leg)
    
    # L_foot LCS
    S_L_foot_y = (S_L_toe - S_L_heel) /  np.linalg.norm((S_L_toe - S_L_heel), ord=2)
    S_L_foot_1 = (S_L_toe - S_M_L_ankle) / np.linalg.norm((S_L_toe - S_M_L_ankle), ord=2)
    S_L_foot_x = np.cross(S_L_foot_1,S_L_foot_y) / np.linalg.norm(np.cross(S_L_foot_1,S_L_foot_y), ord=2)
    S_L_foot_z = np.cross(S_L_foot_x,S_L_foot_y) / np.linalg.norm(np.cross(S_L_foot_x,S_L_foot_y), ord=2)
        
    S_LCS_L_foot = np.array([S_L_foot_x,S_L_foot_y,S_L_foot_z])
    S_Veri_L_foot = np.linalg.det(S_LCS_L_foot)
        
    # R_foot LCS        
    S_R_foot_y = (S_R_toe - S_R_heel) /  np.linalg.norm((S_R_toe - S_R_heel), ord=2)
    S_R_foot_1 = (S_R_toe - S_M_R_ankle) / np.linalg.norm((S_R_toe - S_M_R_ankle), ord=2)
    S_R_foot_x = np.cross(S_R_foot_1,S_R_foot_y) / np.linalg.norm(np.cross(S_R_foot_1,S_R_foot_y), ord=2)
    S_R_foot_z = np.cross(S_R_foot_x,S_R_foot_y) / np.linalg.norm(np.cross(S_R_foot_x,S_R_foot_y), ord=2)
        
    S_LCS_R_foot = np.array([S_R_foot_x,S_R_foot_y,S_R_foot_z])
    S_Veri_R_foot = np.linalg.det(S_LCS_R_foot)
    #%%
    #Dynamic position
    D_R_knee_lat = np.array([43,6,90])
    D_R_knee_med = np.array([43,6,84])
    D_M_R_knee = np.array((D_R_knee_lat + D_R_knee_med) / 2)
    D_R_ankle_lat = np.array([43,-34,90])
    D_R_ankle_med = np.array([43,-34,84])
    D_M_R_ankle = np.array((D_R_ankle_lat + D_R_ankle_med) / 2)
    
    D_R_ASIS = np.array([-7,16,100])
    D_L_ASIS = np.array([-37,16,100])
    D_M_PSIS = np.array([-22,1,105])
    D_R_GT = np.array([3,6,90])
    D_L_GT = np.array([-47,6,90])
    
    D_R_shoulder = np.array([3,71,110])
    D_L_shoulder = np.array([-47,71,110])
    D_IJ = np.array([-22,61,97])
    D_PX = np.array([-22,46,97])
    D_C7 = np.array([-22,61,112])
    D_T7 = np.array([-22,46,109])
    D_M_C7IJ = (D_C7 + D_T7) / 2
    D_M_T7PX = (D_T7 + D_PX) / 2
    
    D_R_HC1 = (2/3 * S_R_GT) + (1/3 * S_R_ASIS)
    D_R_HC2 = (0.18 * S_L_ASIS) + (0.82 * S_R_ASIS)
    D_R_HC3 = (D_R_HC2 - S_R_ASIS)
    D_R_HC =  D_R_HC1 + D_R_HC3
    
    D_L_HC1 = (2/3 * S_L_GT) + (1/3 * S_L_ASIS)
    D_L_HC2 = (0.18 * S_R_ASIS) + (0.82 * S_L_ASIS)
    D_L_HC3 = (D_L_HC2 - S_L_ASIS)
    D_L_HC =  D_L_HC1 + D_L_HC3
    
    
    #%%

    #Static rotation matrix
    S_pel_R_tr = np.dot(S_LCS_trunk, S_LCS_pelvis.T)
    S_R_tr_R_up = np.dot(S_LCS_R_upper, S_LCS_trunk.T)
    S_L_tr_R_up = np.dot(S_LCS_L_upper, S_LCS_trunk.T)
    S_R_pel_R_th = np.dot(S_LCS_R_thigh, S_LCS_pelvis.T)
    S_L_pel_R_th = np.dot(S_LCS_L_thigh, S_LCS_pelvis.T)
    
    # transform from pel to distal
    a1 = math.radians(dat_angles[i,37])
    b1 = math.radians(dat_angles[i,38])
    c1 = math.radians(dat_angles[i,39])
        
    a2 = math.radians(dat_angles[i,1])
    b2 = math.radians(dat_angles[i,2])
    c2 = math.radians(dat_angles[i,3]-10)
        
    a3 = math.radians(dat_angles[i,7])
    b3 = math.radians(dat_angles[i,8]-10)
    c3 = math.radians(dat_angles[i,9]-20)
    
    a4 = math.radians(dat_angles[i,13])
    b4 = math.radians(dat_angles[i,14])
    c4 = math.radians(dat_angles[i,15]-30)
        
    # first rotation
    RM_x1 = np.array([[1, 0, 0],
                      [0, np.cos(a1), np.sin(a1)],
                      [0, -1*np.sin(a1), np.cos(a1)]])
        
    RM_y1 = np.array([[np.cos(b1), 0, -1*np.sin(b1)],
                       [0, 1, 0],
                       [np.sin(b1), 0, np.cos(b1)]])
        
    RM_z1 = np.array([[np.cos(c1), np.sin(c1), 0],
                       [-1*np.sin(c1), np.cos(c1), 0],
                       [0, 0, 1]])
        
    RM_1 = np.array(np.dot(np.dot(RM_z1, RM_y1), RM_x1))
        
    # second rotation    
    RM_x2 = np.array([[1, 0, 0],
                      [0, np.cos(a2), np.sin(a2)],
                      [0, -1*np.sin(a2), np.cos(a2)]])
        
    RM_y2 = np.array([[np.cos(b2), 0, -1*np.sin(b2)],
                       [0, 1, 0],
                       [np.sin(b2), 0, np.cos(b2)]])
        
    RM_z2 = np.array([[np.cos(c2), np.sin(c2), 0],
                       [-1*np.sin(c2), np.cos(c2), 0],
                       [0, 0, 1]])
        
    RM_2 = np.array(np.dot(np.dot(RM_z2, RM_y2), RM_x2))
        
    # third rotation    
    RM_x3 = np.array([[1, 0, 0],
                      [0, np.cos(a3), np.sin(a3)],
                      [0, -1*np.sin(a3), np.cos(a3)]])
        
    RM_y3 = np.array([[np.cos(b3), 0, -1*np.sin(b3)],
                       [0, 1, 0],
                       [np.sin(b3), 0, np.cos(b3)]])
        
    RM_z3 = np.array([[np.cos(c3), np.sin(c3), 0],
                       [-1*np.sin(c3), np.cos(c3), 0],
                       [0, 0, 1]])
    
    RM_3 = np.array(np.dot(np.dot(RM_z3, RM_y3), RM_x3))
    
    # fourth rotation     
    RM_x4 = np.array([[1, 0, 0],
                      [0, np.cos(a4), np.sin(a4)],
                      [0, -1*np.sin(a4), np.cos(a4)]])
        
    RM_y4 = np.array([[np.cos(b4), 0, -1*np.sin(b4)], 
                       [0, 1, 0],
                       [np.sin(b4), 0, np.cos(b4)]])
        
    RM_z4 = np.array([[np.cos(c4), np.sin(c4), 0],
                       [-1*np.sin(c4), np.cos(c4), 0],       
                       [0, 0, 1]])
        
    RM_4 = np.array(np.dot(np.dot(RM_z4, RM_y4), RM_x4))
    
    #%%
    # position
    # first rotation of upper-body
    # position from trunk LCS
    tr_R_ASIS = np.dot(S_LCS_trunk.T,S_R_ASIS)
    tr_L_ASIS = np.dot(S_LCS_trunk.T,S_L_ASIS)
    tr_M_PSIS = np.dot(S_LCS_trunk.T,S_M_PSIS)
    tr_IJ = np.dot(S_LCS_trunk.T,S_IJ)
    tr_PX = np.dot(S_LCS_trunk.T,S_PX)
    tr_C7 = np.dot(S_LCS_trunk.T,S_C7)
    tr_T7 = np.dot(S_LCS_trunk.T,S_T7)
    tr_R_shoulder = np.dot(S_LCS_trunk.T,S_R_shoulder)
    tr_L_shoulder = np.dot(S_LCS_trunk.T,S_L_shoulder)
    
    tr_R_elbow_lat = np.dot(S_LCS_trunk.T,S_R_elbow_lat)
    tr_R_elbow_med = np.dot(S_LCS_trunk.T,S_R_elbow_med)
    tr_L_elbow_lat = np.dot(S_LCS_trunk.T,S_L_elbow_lat)
    tr_L_elbow_med = np.dot(S_LCS_trunk.T,S_L_elbow_med)
    
    tr_R_wrist_lat = np.dot(S_LCS_trunk.T,S_R_wrist_lat)
    tr_R_wrist_med = np.dot(S_LCS_trunk.T,S_R_wrist_med)
    tr_L_wrist_lat = np.dot(S_LCS_trunk.T,S_L_wrist_lat)
    tr_L_wrist_med = np.dot(S_LCS_trunk.T,S_L_wrist_med)
    
    tr_R_MC3 = np.dot(S_LCS_trunk.T,S_R_MC3)
    tr_L_MC3 = np.dot(S_LCS_trunk.T,S_L_MC3)
    
    # origin point as lumber joint
    lumber = (S_R_ASIS + S_L_ASIS) / 2
    
    tr_R_ASIS2 = tr_R_ASIS - lumber
    tr_L_ASIS2 = tr_L_ASIS - lumber
    tr_M_PSIS2 = tr_M_PSIS - lumber
    tr_IJ2 = tr_IJ - lumber
    tr_PX2 = tr_PX - lumber
    tr_C72 = tr_C7 - lumber
    tr_T72 = tr_T7 - lumber
    tr_R_shoulder2 = tr_R_shoulder - lumber
    tr_L_shoulder2 = tr_L_shoulder - lumber
    
    tr_R_elbow_lat2 = tr_R_elbow_lat - lumber
    tr_R_elbow_med2 = tr_R_elbow_med - lumber
    tr_L_elbow_lat2 = tr_L_elbow_lat - lumber
    tr_L_elbow_med2 = tr_L_elbow_med - lumber
    
    tr_R_wrist_lat2 = tr_R_wrist_lat - lumber
    tr_R_wrist_med2 = tr_R_wrist_med - lumber
    tr_L_wrist_lat2 = tr_L_wrist_lat - lumber
    tr_L_wrist_med2 = tr_L_wrist_med - lumber
    
    tr_R_MC32 = tr_R_MC3 - lumber
    tr_L_MC32 = tr_L_MC3 - lumber
    
    # Dynamic position
    D_lumber = (D_R_ASIS + D_L_ASIS ) / 2
    
    D_G_R_ASIS = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_R_ASIS2) + D_lumber
    D_G_L_ASIS = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_L_ASIS2) + D_lumber
    D_G_M_PSIS = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_M_PSIS2) + D_lumber
    D_G_IJ = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_IJ2) + D_lumber
    D_G_PX = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_PX2) + D_lumber
    D_G_C7 = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_C72) + D_lumber
    D_G_T7 = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_T72) + D_lumber
    D_G_R_shoulder = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_R_shoulder2) + D_lumber
    D_G_L_shoulder = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_L_shoulder2) + D_lumber
    D_G_M_C7IJ = (D_G_C7 + D_G_IJ) / 2
    D_G_M_T7PX = (D_G_T7 + D_G_PX) / 2
    
    D_G_R_elbow_lat = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_R_elbow_lat2) + D_lumber
    D_G_R_elbow_med = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_R_elbow_med2) + D_lumber
    D_G_L_elbow_lat = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_L_elbow_lat2) + D_lumber
    D_G_L_elbow_med = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_L_elbow_med2) + D_lumber
    D_G_M_R_elbow = (D_G_R_elbow_lat + D_G_R_elbow_med) / 2
    D_G_M_L_elbow = (D_G_L_elbow_lat + D_G_L_elbow_med) / 2
    
    D_G_R_wrist_lat = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_R_wrist_lat2) + D_lumber
    D_G_R_wrist_med = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_R_wrist_med2) + D_lumber
    D_G_L_wrist_lat = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_L_wrist_lat2) + D_lumber
    D_G_L_wrist_med = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_L_wrist_med2) + D_lumber
    D_G_M_R_wrist = (D_G_R_wrist_lat + D_G_R_wrist_med) / 2
    D_G_M_L_wrist = (D_G_L_wrist_lat + D_G_L_wrist_med) / 2
    
    D_G_R_MC3 = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_R_MC32) + D_lumber
    D_G_L_MC3 = np.dot(np.dot(RM_1,S_pel_R_tr.T),tr_L_MC32) + D_lumber
    #%%
    # second rotation of L_upper-body
    # position from upper arm LCS
    up_L_elbow_lat = np.dot(S_LCS_R_upper.T,D_G_L_elbow_lat)
    up_L_elbow_med = np.dot(S_LCS_R_upper.T,D_G_L_elbow_med)
    
    up_L_wrist_lat = np.dot(S_LCS_R_upper.T,D_G_L_wrist_lat)
    up_L_wrist_med = np.dot(S_LCS_R_upper.T,D_G_L_wrist_med)
    
    up_L_MC3 = np.dot(S_LCS_R_upper.T,D_G_L_MC3)
    
    # origin point as shoulder joint
    L_shoulder = D_G_L_shoulder
    
    up_L_elbow_lat2 = up_L_elbow_lat - L_shoulder
    up_L_elbow_med2 = up_L_elbow_med - L_shoulder
    
    up_L_wrist_lat2 = up_L_wrist_lat - L_shoulder
    up_L_wrist_med2 = up_L_wrist_med - L_shoulder
    
    up_L_MC32 = up_L_MC3 - L_shoulder
    
    # Dynamic position
    # trunk LCS
    S_trunk_y = (D_G_PX- D_G_T7) / np.linalg.norm((D_G_PX - D_G_T7), ord=2)
    S_trunk_1 = (D_G_M_C7IJ - D_G_M_T7PX) / np.linalg.norm((D_G_M_C7IJ - D_G_M_T7PX), ord=2)
    S_trunk_x = np.cross(S_trunk_y,S_trunk_1) / np.linalg.norm(np.cross(S_trunk_y,S_trunk_1), ord=2)
    S_trunk_z = np.cross(S_trunk_x,S_trunk_y) / np.linalg.norm(np.cross(S_trunk_x,S_trunk_y), ord=2)
        
    S_LCS_trunk = np.array([S_trunk_x,S_trunk_y,S_trunk_z])
    S_Veri_trunk = np.linalg.det(S_LCS_trunk)
    
    # L_upper_arm LCS
    S_L_upper_z = (D_G_L_elbow_lat - D_G_L_elbow_med) / np.linalg.norm((D_G_L_elbow_lat - D_G_L_elbow_med), ord=2)
    S_L_upper_1 = (D_G_L_shoulder - D_G_M_L_elbow) / np.linalg.norm((D_G_L_shoulder - D_G_M_L_elbow), ord=2)
    S_L_upper_y = np.cross(S_L_upper_z,S_L_upper_1) / np.linalg.norm(np.cross(S_L_upper_z,S_L_upper_1), ord=2)
    S_L_upper_x = np.cross(S_L_upper_y,S_L_upper_z) / np.linalg.norm(np.cross(S_L_upper_y,S_L_upper_z), ord=2)
        
    S_LCS_L_upper = np.array([S_L_upper_x,S_L_upper_y,S_L_upper_z])
    S_Veri_L_upper = np.linalg.det(S_LCS_L_upper)
        
    D_L_shoulderjoint = D_G_L_shoulder
    
    D_G_L_elbow_lat = np.dot(np.dot(RM_2,S_L_tr_R_up.T),up_L_elbow_lat2) + D_L_shoulderjoint
    D_G_L_elbow_med = np.dot(np.dot(RM_2,S_L_tr_R_up.T),up_L_elbow_med2) + D_L_shoulderjoint
    
    D_G_L_wrist_lat = np.dot(np.dot(RM_2,S_L_tr_R_up.T),up_L_wrist_lat2) + D_L_shoulderjoint
    D_G_L_wrist_med = np.dot(np.dot(RM_2,S_L_tr_R_up.T),up_L_wrist_med2) + D_L_shoulderjoint
    
    D_G_L_MC3 = np.dot(np.dot(RM_2,S_L_tr_R_up.T),up_L_MC32) + D_L_shoulderjoint
    
    #%%
    # third rotation of upper-body
    # position from lower arm LCS
    lw_L_wrist_lat = np.dot(S_LCS_R_lower.T,D_G_L_wrist_lat)
    lw_L_wrist_med = np.dot(S_LCS_R_lower.T,D_G_L_wrist_med)
    
    lw_L_MC3 = np.dot(S_LCS_R_lower.T,D_G_L_MC3)
    
    # origin point as elbow joint
    D_G_M_L_elbow = (D_G_L_elbow_lat + D_G_L_elbow_med) / 2
    L_elbow = D_G_M_L_elbow
    
    lw_L_wrist_lat2 = lw_L_wrist_lat - L_elbow
    lw_L_wrist_med2 = lw_L_wrist_med - L_elbow
    
    lw_L_MC32 = lw_L_MC3 - L_elbow
    
    # Dynamic position
    # L_upper_arm LCS
    S_L_upper_z = (D_G_L_elbow_lat - D_G_L_elbow_med) / np.linalg.norm((D_G_L_elbow_lat - D_G_L_elbow_med), ord=2)
    S_L_upper_1 = (D_G_L_shoulder - D_G_M_L_elbow) / np.linalg.norm((D_G_L_shoulder - D_G_M_L_elbow), ord=2)
    S_L_upper_y = np.cross(S_L_upper_z,S_L_upper_1) / np.linalg.norm(np.cross(S_L_upper_z,S_L_upper_1), ord=2)
    S_L_upper_x = np.cross(S_L_upper_y,S_L_upper_z) / np.linalg.norm(np.cross(S_L_upper_y,S_L_upper_z), ord=2)
        
    S_LCS_L_upper = np.array([S_L_upper_x,S_L_upper_y,S_L_upper_z])
    S_Veri_L_upper = np.linalg.det(S_LCS_L_upper)
        
    # L_lower_arm LCS
    D_G_L_lower_z = (D_G_L_wrist_lat - D_G_L_wrist_med) / np.linalg.norm((D_G_L_wrist_lat - D_G_L_wrist_med), ord=2)
    D_G_L_lower_1 = (D_G_M_L_elbow - D_G_M_L_wrist) / np.linalg.norm((D_G_M_L_elbow - D_G_M_L_wrist), ord=2)
    S_L_lower_y = np.cross(S_L_lower_z,S_L_lower_1) / np.linalg.norm(np.cross(S_L_lower_z,S_L_lower_1), ord=2)
    S_L_lower_x = np.cross(S_L_lower_y,S_L_lower_z) / np.linalg.norm(np.cross(S_L_lower_y,S_L_lower_z), ord=2)
        
    S_LCS_L_lower = np.array([S_L_lower_x,S_L_lower_y,S_L_lower_z])
    S_Veri_L_lower = np.linalg.det(S_LCS_L_lower)
    
    S_L_up_R_lw = np.dot(S_LCS_L_lower, S_LCS_L_upper.T)
    
    D_L_elbowjoint = D_G_M_L_elbow
    
    D_G_L_wrist_lat = np.dot(np.dot(RM_3,S_L_up_R_lw.T),lw_L_wrist_lat2) + D_L_elbowjoint
    D_G_L_wrist_med = np.dot(np.dot(RM_3,S_L_up_R_lw.T),lw_L_wrist_med2) + D_L_elbowjoint
    
    D_G_L_MC3 = np.dot(np.dot(RM_3,S_L_up_R_lw.T),lw_L_MC32) + D_L_elbowjoint
    
    #%%
    # fourth rotation of upper-body
    # position from hand LCS
    ha_L_MC3 = np.dot(S_LCS_R_hand.T,D_G_L_MC3)
    
    # origin point as elbow joint
    D_G_M_L_wrist = (D_G_L_wrist_lat + D_G_L_wrist_med) / 2
    L_wrist = D_G_M_L_wrist
    D_G_M_L_elbow = (D_G_L_elbow_lat + D_G_L_elbow_med) / 2
    
    ha_L_MC32 = ha_L_MC3 - L_wrist
    
    # Dynamic position
    # L_lower_arm LCS
    D_G_L_lower_z = (D_G_L_wrist_lat - D_G_L_wrist_med) / np.linalg.norm((D_G_L_wrist_lat - D_G_L_wrist_med), ord=2)
    D_G_L_lower_1 = (D_G_M_L_elbow - D_G_M_L_wrist) / np.linalg.norm((D_G_M_L_elbow - D_G_M_L_wrist), ord=2)
    S_L_lower_y = np.cross(S_L_lower_z,S_L_lower_1) / np.linalg.norm(np.cross(S_L_lower_z,S_L_lower_1), ord=2)
    S_L_lower_x = np.cross(S_L_lower_y,S_L_lower_z) / np.linalg.norm(np.cross(S_L_lower_y,S_L_lower_z), ord=2)
        
    S_LCS_L_lower = np.array([S_L_lower_x,S_L_lower_y,S_L_lower_z])
    S_Veri_L_lower = np.linalg.det(S_LCS_L_lower)
    
    S_L_up_R_lw = np.dot(S_LCS_L_lower, S_LCS_L_upper.T)
    
    # L_hand LCS
    S_L_hand_x = (D_G_M_L_wrist - D_G_L_MC3) / np.linalg.norm((D_G_M_L_wrist - D_G_L_MC3), ord=2)
    S_L_hand_y = np.cross(D_G_L_lower_z,S_L_hand_x) / np.linalg.norm(np.cross(D_G_L_lower_z,S_L_hand_x), ord=2)
    S_L_hand_z = np.cross(S_L_hand_x,S_L_hand_y) / np.linalg.norm(np.cross(S_L_hand_x,S_L_hand_y), ord=2)
        
    S_LCS_L_hand = np.array([S_L_hand_x,S_L_hand_y,S_L_hand_z])
    S_Veri_L_hand = np.linalg.det(S_LCS_L_hand)
        
    S_L_lw_R_ha = np.dot(S_LCS_L_hand, S_LCS_L_lower.T)
    
    D_L_wristjoint = D_G_M_L_wrist
    
    D_G_L_MC3 = np.dot(np.dot(RM_4,S_L_lw_R_ha.T),ha_L_MC32) + D_L_wristjoint
    
    #%%
    a2 = math.radians(dat_angles[i,4])
    b2 = math.radians(dat_angles[i,5])
    c2 = math.radians(dat_angles[i,6]-10)
        
    a3 = math.radians(dat_angles[i,10])
    b3 = math.radians(dat_angles[i,11]-10)
    c3 = math.radians(dat_angles[i,12]-20)
        
    a4 = math.radians(dat_angles[i,16])
    b4 = math.radians(dat_angles[i,17])
    c4 = math.radians(dat_angles[i,18]-30)
        
    # first rotation
    RM_x1 = np.array([[1, 0, 0],
                      [0, np.cos(a1), np.sin(a1)],
                      [0, -1*np.sin(a1), np.cos(a1)]])
        
    RM_y1 = np.array([[np.cos(b1), 0, -1*np.sin(b1)],
                       [0, 1, 0],
                       [np.sin(b1), 0, np.cos(b1)]])
        
    RM_z1 = np.array([[np.cos(c1), np.sin(c1), 0],
                       [-1*np.sin(c1), np.cos(c1), 0],
                       [0, 0, 1]])
        
    RM_1 = np.array(np.dot(np.dot(RM_z1, RM_y1), RM_x1))
        
    # second rotation    
    RM_x2 = np.array([[1, 0, 0],
                      [0, np.cos(a2), np.sin(a2)],
                      [0, -1*np.sin(a2), np.cos(a2)]])
        
    RM_y2 = np.array([[np.cos(b2), 0, -1*np.sin(b2)],
                       [0, 1, 0],
                       [np.sin(b2), 0, np.cos(b2)]])
        
    RM_z2 = np.array([[np.cos(c2), np.sin(c2), 0],
                       [-1*np.sin(c2), np.cos(c2), 0],
                       [0, 0, 1]])
        
    RM_2 = np.array(np.dot(np.dot(RM_z2, RM_y2), RM_x2))
        
    # third rotation    
    RM_x3 = np.array([[1, 0, 0],
                      [0, np.cos(a3), np.sin(a3)],
                      [0, -1*np.sin(a3), np.cos(a3)]])
        
    RM_y3 = np.array([[np.cos(b3), 0, -1*np.sin(b3)],
                       [0, 1, 0],
                       [np.sin(b3), 0, np.cos(b3)]])
        
    RM_z3 = np.array([[np.cos(c3), np.sin(c3), 0],
                       [-1*np.sin(c3), np.cos(c3), 0],
                       [0, 0, 1]])

    RM_3 = np.array(np.dot(np.dot(RM_z3, RM_y3), RM_x3))
    
    # fourth rotation     
    RM_x4 = np.array([[1, 0, 0],
                      [0, np.cos(a4), np.sin(a4)],
                      [0, -1*np.sin(a4), np.cos(a4)]])
        
    RM_y4 = np.array([[np.cos(b4), 0, -1*np.sin(b4)], 
                       [0, 1, 0],
                       [np.sin(b4), 0, np.cos(b4)]])
        
    RM_z4 = np.array([[np.cos(c4), np.sin(c4), 0],
                       [-1*np.sin(c4), np.cos(c4), 0],       
                       [0, 0, 1]])
        
    RM_4 = np.array(np.dot(np.dot(RM_z4, RM_y4), RM_x4))
    #%%
    # second rotation of R_upper-body
    # position from upper arm LCS
    up_R_elbow_lat = np.dot(S_LCS_R_upper.T,D_G_R_elbow_lat)
    up_R_elbow_med = np.dot(S_LCS_R_upper.T,D_G_R_elbow_med)
    
    up_R_wrist_lat = np.dot(S_LCS_R_upper.T,D_G_R_wrist_lat)
    up_R_wrist_med = np.dot(S_LCS_R_upper.T,D_G_R_wrist_med)
    
    up_R_MC3 = np.dot(S_LCS_R_upper.T,D_G_R_MC3)
    
    # origin point as shoulder joint
    R_shoulder = D_G_R_shoulder
    
    up_R_elbow_lat2 = up_R_elbow_lat - R_shoulder
    up_R_elbow_med2 = up_R_elbow_med - R_shoulder
    
    up_R_wrist_lat2 = up_R_wrist_lat - R_shoulder
    up_R_wrist_med2 = up_R_wrist_med - R_shoulder
    
    up_R_MC32 = up_R_MC3 - R_shoulder
    
    # Dynamic position
    # trunk LCS
    S_trunk_y = (D_G_PX- D_G_T7) / np.linalg.norm((D_G_PX - D_G_T7), ord=2)
    S_trunk_1 = (D_G_M_C7IJ - D_G_M_T7PX) / np.linalg.norm((D_G_M_C7IJ - D_G_M_T7PX), ord=2)
    S_trunk_x = np.cross(S_trunk_y,S_trunk_1) / np.linalg.norm(np.cross(S_trunk_y,S_trunk_1), ord=2)
    S_trunk_z = np.cross(S_trunk_x,S_trunk_y) / np.linalg.norm(np.cross(S_trunk_x,S_trunk_y), ord=2)
        
    S_LCS_trunk = np.array([S_trunk_x,S_trunk_y,S_trunk_z])
    S_Veri_trunk = np.linalg.det(S_LCS_trunk)
    
    # R_upper_arm LCS
    S_R_upper_z = (D_G_R_elbow_lat - D_G_R_elbow_med) / np.linalg.norm((D_G_R_elbow_lat - D_G_R_elbow_med), ord=2)
    S_R_upper_1 = (D_G_R_shoulder - D_G_M_R_elbow) / np.linalg.norm((D_G_R_shoulder - D_G_M_R_elbow), ord=2)
    S_R_upper_y = np.cross(S_R_upper_1,S_R_upper_z) / np.linalg.norm(np.cross(S_R_upper_1,S_R_upper_z), ord=2)
    S_R_upper_x = np.cross(S_R_upper_y,S_R_upper_z) / np.linalg.norm(np.cross(S_R_upper_y,S_R_upper_z), ord=2)
        
    S_LCS_R_upper = np.array([S_R_upper_x,S_R_upper_y,S_R_upper_z])
    S_Veri_R_upper = np.linalg.det(S_LCS_R_upper)
    
    S_R_tr_R_up = np.dot(S_LCS_R_upper, S_LCS_trunk.T)
    S_L_tr_R_up = np.dot(S_LCS_L_upper, S_LCS_trunk.T)
    
    D_R_shoulderjoint = D_G_R_shoulder
    
    D_G_R_elbow_lat = np.dot(np.dot(RM_2,S_R_tr_R_up.T),up_R_elbow_lat2) + D_R_shoulderjoint
    D_G_R_elbow_med = np.dot(np.dot(RM_2,S_R_tr_R_up.T),up_R_elbow_med2) + D_R_shoulderjoint
    
    D_G_R_wrist_lat = np.dot(np.dot(RM_2,S_R_tr_R_up.T),up_R_wrist_lat2) + D_R_shoulderjoint
    D_G_R_wrist_med = np.dot(np.dot(RM_2,S_R_tr_R_up.T),up_R_wrist_med2) + D_R_shoulderjoint
    
    D_G_R_MC3 = np.dot(np.dot(RM_2,S_R_tr_R_up.T),up_R_MC32) + D_R_shoulderjoint
    
    #%%
    # third rotation of upper-body
    # position from lower arm LCS
    lw_R_wrist_lat = np.dot(S_LCS_R_lower.T,D_G_R_wrist_lat)
    lw_R_wrist_med = np.dot(S_LCS_R_lower.T,D_G_R_wrist_med)
    
    lw_R_MC3 = np.dot(S_LCS_R_lower.T,D_G_R_MC3)
    
    # origin point as elbow joint
    D_G_M_R_elbow = (D_G_R_elbow_lat + D_G_R_elbow_med) / 2
    R_elbow = D_G_M_R_elbow
    
    lw_R_wrist_lat2 = lw_R_wrist_lat - R_elbow
    lw_R_wrist_med2 = lw_R_wrist_med - R_elbow
    
    lw_R_MC32 = lw_R_MC3 - R_elbow
    
    # Dynamic position
    # R_upper_arm LCS
    S_R_upper_z = (D_G_R_elbow_lat - D_G_R_elbow_med) / np.linalg.norm((D_G_R_elbow_lat - D_G_R_elbow_med), ord=2)
    S_R_upper_1 = (D_G_R_shoulder - D_G_M_R_elbow) / np.linalg.norm((D_G_R_shoulder - D_G_M_R_elbow), ord=2)
    S_R_upper_y = np.cross(S_R_upper_1,S_R_upper_z) / np.linalg.norm(np.cross(S_R_upper_1,S_R_upper_z), ord=2)
    S_R_upper_x = np.cross(S_R_upper_y,S_R_upper_z) / np.linalg.norm(np.cross(S_R_upper_y,S_R_upper_z), ord=2)
        
    S_LCS_R_upper = np.array([S_R_upper_x,S_R_upper_y,S_R_upper_z])
    S_Veri_R_upper = np.linalg.det(S_LCS_R_upper)
    
    # R_lower_arm LCS
    D_G_R_lower_z = (D_G_R_wrist_lat - D_G_R_wrist_med) / np.linalg.norm((D_G_R_wrist_lat - D_G_R_wrist_med), ord=2)
    D_G_R_lower_1 = (D_G_M_R_elbow - D_G_M_R_wrist) / np.linalg.norm((D_G_M_R_elbow - D_G_M_R_wrist), ord=2)
    S_R_lower_y = np.cross(S_R_lower_1,S_R_lower_z) / np.linalg.norm(np.cross(S_R_lower_1,S_R_lower_z), ord=2)
    S_R_lower_x = np.cross(S_R_lower_y,S_R_lower_z) / np.linalg.norm(np.cross(S_R_lower_y,S_R_lower_z), ord=2)
        
    S_LCS_R_lower = np.array([S_R_lower_x,S_R_lower_y,S_R_lower_z])
    S_Veri_R_lower = np.linalg.det(S_LCS_R_lower)
    
    S_R_up_R_lw = np.dot(S_LCS_R_lower, S_LCS_R_upper.T)
    
    D_R_elbowjoint = D_G_M_R_elbow
    
    D_G_R_wrist_lat = np.dot(np.dot(RM_3,S_R_up_R_lw.T),lw_R_wrist_lat2) + D_R_elbowjoint
    D_G_R_wrist_med = np.dot(np.dot(RM_3,S_R_up_R_lw.T),lw_R_wrist_med2) + D_R_elbowjoint
    
    D_G_R_MC3 = np.dot(np.dot(RM_3,S_R_up_R_lw.T),lw_R_MC32) + D_R_elbowjoint
    
    #%%
    # fourth rotation of upper-body
    # position from hand LCS
    ha_R_MC3 = np.dot(S_LCS_R_hand.T,D_G_R_MC3)
    
    # origin point as elbow joint
    D_G_M_R_wrist = (D_G_R_wrist_lat + D_G_R_wrist_med) / 2
    R_wrist = D_G_M_R_wrist
    D_G_M_R_elbow = (D_G_R_elbow_lat + D_G_R_elbow_med) / 2
    
    ha_R_MC32 = ha_R_MC3 - R_wrist
    
    # Dynamic position
    # R_lower_arm LCS
    D_G_R_lower_z = (D_G_R_wrist_lat - D_G_R_wrist_med) / np.linalg.norm((D_G_R_wrist_lat - D_G_R_wrist_med), ord=2)
    D_G_R_lower_1 = (D_G_M_R_elbow - D_G_M_R_wrist) / np.linalg.norm((D_G_M_R_elbow - D_G_M_R_wrist), ord=2)
    S_R_lower_y = np.cross(S_R_lower_1,S_R_lower_z) / np.linalg.norm(np.cross(S_R_lower_1,S_R_lower_z), ord=2)
    S_R_lower_x = np.cross(S_R_lower_y,S_R_lower_z) / np.linalg.norm(np.cross(S_R_lower_y,S_R_lower_z), ord=2)
        
    S_LCS_R_lower = np.array([S_R_lower_x,S_R_lower_y,S_R_lower_z])
    S_Veri_R_lower = np.linalg.det(S_LCS_R_lower)
    
    S_R_up_R_lw = np.dot(S_LCS_R_lower, S_LCS_R_upper.T)
    
    # R_hand LCS 
    S_R_hand_x = (D_G_R_MC3 - D_G_M_R_wrist) / np.linalg.norm((D_G_M_R_wrist - D_G_R_MC3), ord=2)
    S_R_hand_y = np.cross(D_G_R_lower_z,S_R_hand_x) / np.linalg.norm(np.cross(D_G_R_lower_z,S_R_hand_x), ord=2)
    S_R_hand_z = np.cross(S_R_hand_x,S_R_hand_y) / np.linalg.norm(np.cross(S_R_hand_x,S_R_hand_y), ord=2)
    
    S_LCS_R_hand = np.array([S_R_hand_x,S_R_hand_y,S_R_hand_z])
    S_Veri_R_hand = np.linalg.det(S_LCS_R_hand)
    
    S_R_lw_R_ha = np.dot(S_LCS_R_hand, S_LCS_R_lower.T)
    
    D_R_wristjoint = D_G_M_R_wrist
    
    D_G_R_MC3 = np.dot(np.dot(RM_4,S_R_lw_R_ha.T),ha_R_MC32) + D_R_wristjoint
    
    #%%
    #Static rotation matrix
    S_R_pel_R_th = np.dot(S_LCS_R_thigh, S_LCS_pelvis.T)
    S_L_pel_R_th = np.dot(S_LCS_L_thigh, S_LCS_pelvis.T)
    
    # transform from pel to distal
    a1 = math.radians(dat_angles[i,19])
    b1 = math.radians(dat_angles[i,20])
    c1 = math.radians(dat_angles[i,21])
        
    a2 = math.radians(dat_angles[i,25])
    b2 = math.radians(dat_angles[i,26])
    c2 = math.radians(dat_angles[i,27])
        
    a3 = math.radians(dat_angles[i,31])
    b3 = math.radians(dat_angles[i,32])
    c3 = math.radians(dat_angles[i,33])
    
    # first rotation
    RM_x1 = np.array([[1, 0, 0],
                      [0, np.cos(a1), np.sin(a1)],
                      [0, -1*np.sin(a1), np.cos(a1)]])
        
    RM_y1 = np.array([[np.cos(b1), 0, -1*np.sin(b1)],
                       [0, 1, 0],
                       [np.sin(b1), 0, np.cos(b1)]])
        
    RM_z1 = np.array([[np.cos(c1), np.sin(c1), 0],
                       [-1*np.sin(c1), np.cos(c1), 0],
                       [0, 0, 1]])
        
    RM_1 = np.array(np.dot(np.dot(RM_z1, RM_y1), RM_x1))
        
    # second rotation    
    RM_x2 = np.array([[1, 0, 0],
                      [0, np.cos(a2), np.sin(a2)],
                      [0, -1*np.sin(a2), np.cos(a2)]])
        
    RM_y2 = np.array([[np.cos(b2), 0, -1*np.sin(b2)],
                       [0, 1, 0],
                       [np.sin(b2), 0, np.cos(b2)]])
        
    RM_z2 = np.array([[np.cos(c2), np.sin(c2), 0],
                       [-1*np.sin(c2), np.cos(c2), 0],
                       [0, 0, 1]])
        
    RM_2 = np.array(np.dot(np.dot(RM_z2, RM_y2), RM_x2))
        
    # third rotation    
    RM_x3 = np.array([[1, 0, 0],
                      [0, np.cos(a3), np.sin(a3)],
                      [0, -1*np.sin(a3), np.cos(a3)]])
        
    RM_y3 = np.array([[np.cos(b3), 0, -1*np.sin(b3)],
                       [0, 1, 0],
                       [np.sin(b3), 0, np.cos(b3)]])
        
    RM_z3 = np.array([[np.cos(c3), np.sin(c3), 0],
                       [-1*np.sin(c3), np.cos(c3), 0],
                       [0, 0, 1]])
    
    RM_3 = np.array(np.dot(np.dot(RM_z3, RM_y3), RM_x3))
    
    
    #%%
    # first rotation of L_lower-body
    # position from thigh LCS
    
    th_L_GT = np.dot(S_LCS_L_thigh.T,S_L_GT)
    th_L_knee_lat = np.dot(S_LCS_L_thigh.T,S_L_knee_lat)
    th_L_knee_med = np.dot(S_LCS_L_thigh.T,S_L_knee_med)
    
    th_L_ankle_lat = np.dot(S_LCS_L_thigh.T,S_L_ankle_lat)
    th_L_ankle_med = np.dot(S_LCS_L_thigh.T,S_L_ankle_med)
    
    th_L_toe = np.dot(S_LCS_L_thigh.T,S_L_toe)
    th_L_heel = np.dot(S_LCS_L_thigh.T,S_L_heel)
    
    # origin point as hip joint
    L_hip = S_L_GT
    
    th_L_knee_lat2 = th_L_knee_lat - L_hip
    th_L_knee_med2 = th_L_knee_med - L_hip
    
    th_L_ankle_lat2 = th_L_ankle_lat - L_hip
    th_L_ankle_med2 = th_L_ankle_med - L_hip
    
    th_L_toe2 = th_L_toe - L_hip
    th_L_heel2 = th_L_heel - L_hip
    
    # Dynamic position
    D_L_hip = D_L_GT
    
    D_G_L_knee_lat = np.dot(np.dot(RM_1,S_L_pel_R_th.T),th_L_knee_lat2) + D_L_hip
    D_G_L_knee_med = np.dot(np.dot(RM_1,S_L_pel_R_th.T),th_L_knee_med2) + D_L_hip
    D_G_M_L_knee = (D_G_L_knee_lat + D_G_L_knee_med) / 2
    
    D_G_L_ankle_lat = np.dot(np.dot(RM_1,S_L_pel_R_th.T),th_L_ankle_lat2) + D_L_hip
    D_G_L_ankle_med = np.dot(np.dot(RM_1,S_L_pel_R_th.T),th_L_ankle_med2) + D_L_hip
    D_G_M_L_ankle = (D_G_L_ankle_lat + D_G_L_ankle_med) / 2
    
    D_G_L_toe = np.dot(np.dot(RM_1,S_L_pel_R_th.T),th_L_toe2) + D_L_hip
    D_G_L_heel = np.dot(np.dot(RM_1,S_L_pel_R_th.T),th_L_heel2) + D_L_hip
    
    #%%
    # second rotation of R_lower-body
    # position from leg LCS
    le_L_ankle_lat = np.dot(S_LCS_L_leg.T,D_G_L_ankle_lat)
    le_L_ankle_med = np.dot(S_LCS_L_leg.T,D_G_L_ankle_med)
    
    le_L_toe = np.dot(S_LCS_L_leg.T,D_G_L_toe)
    le_L_heel = np.dot(S_LCS_L_leg.T,D_G_L_heel)
    
    # origin point as lumber joint
    L_knee = D_G_M_L_knee
    
    le_L_ankle_lat2 = le_L_ankle_lat - L_knee
    le_L_ankle_med2 = le_L_ankle_med - L_knee
    
    le_L_toe2 = le_L_toe - L_knee
    le_L_heel2 = le_L_heel - L_knee
    
    # Dynamic position
    # L_thigh LCS
    S_L_thigh_x = (D_G_L_knee_med - D_G_L_knee_lat) / np.linalg.norm((D_G_L_knee_med - D_G_L_knee_lat), ord=2)
    S_L_thigh_1 = (S_L_GT - D_G_M_L_knee) / np.linalg.norm((S_L_GT - D_G_M_L_knee), ord=2)
    S_L_thigh_y = np.cross(S_L_thigh_1,S_L_thigh_x) / np.linalg.norm(np.cross(S_L_thigh_1,S_L_thigh_x), ord=2)
    S_L_thigh_z = np.cross(S_L_thigh_x,S_L_thigh_y) / np.linalg.norm(np.cross(S_L_thigh_x,S_L_thigh_y), ord=2)
        
    S_LCS_L_thigh = np.array([S_L_thigh_x,S_L_thigh_y,S_L_thigh_z])
    S_Veri_L_thigh = np.linalg.det(S_LCS_L_thigh)
        
        
    # L_leg LCS
    S_L_leg_x = (D_G_L_ankle_med - D_G_L_ankle_lat) / np.linalg.norm((D_G_L_ankle_med - D_G_L_ankle_lat), ord=2)
    S_L_leg_1 = (D_G_M_L_knee - D_G_M_L_ankle) / np.linalg.norm((D_G_M_L_knee - D_G_M_L_ankle), ord=2)
    S_L_leg_y = np.cross(S_L_leg_1,S_L_leg_x) / np.linalg.norm(np.cross(S_L_leg_1,S_L_leg_x), ord=2)
    S_L_leg_z = np.cross(S_L_leg_x,S_L_leg_y) / np.linalg.norm(np.cross(S_L_leg_x,S_L_leg_y), ord=2)
    S_LCS_L_leg = np.array([S_L_leg_x,S_L_leg_y,S_L_leg_z])
    S_Veri_L_leg = np.linalg.det(S_LCS_L_leg)
        
    S_L_th_R_le = np.dot(S_LCS_L_leg, S_LCS_L_thigh.T)
    
    D_L_kneejoint = D_G_M_L_knee
    
    D_G_L_ankle_lat = np.dot(np.dot(RM_2,S_L_th_R_le.T),le_L_ankle_lat2) + D_L_kneejoint
    D_G_L_ankle_med = np.dot(np.dot(RM_2,S_L_th_R_le.T),le_L_ankle_med2) + D_L_kneejoint
    
    D_G_L_toe = np.dot(np.dot(RM_2,S_L_th_R_le.T),le_L_toe2) + D_L_kneejoint
    D_G_L_heel = np.dot(np.dot(RM_2,S_L_th_R_le.T),le_L_heel2) + D_L_kneejoint
    
    #%%
    # third rotation of upper-body
    # position from foot LCS
    fo_L_toe = np.dot(S_LCS_R_foot.T,D_G_L_toe)
    fo_L_heel = np.dot(S_LCS_R_foot.T,D_G_L_heel)
    
    # origin point as ankle joint
    D_G_M_L_ankle = (D_G_L_ankle_lat + D_G_L_ankle_med) / 2
    L_ankle = D_G_M_L_ankle
    
    fo_L_toe2 = fo_L_toe - L_ankle
    fo_L_heel2 = fo_L_heel - L_ankle
    
    # Dynamic position
    # L_leg LCS
    S_L_leg_x = (D_G_L_ankle_med - D_G_L_ankle_lat) / np.linalg.norm((D_G_L_ankle_med - D_G_L_ankle_lat), ord=2)
    S_L_leg_1 = (D_G_M_L_knee - D_G_M_L_ankle) / np.linalg.norm((D_G_M_L_knee - D_G_M_L_ankle), ord=2)
    S_L_leg_y = np.cross(S_L_leg_1,S_L_leg_x) / np.linalg.norm(np.cross(S_L_leg_1,S_L_leg_x), ord=2)
    S_L_leg_z = np.cross(S_L_leg_x,S_L_leg_y) / np.linalg.norm(np.cross(S_L_leg_x,S_L_leg_y), ord=2)
    S_LCS_L_leg = np.array([S_L_leg_x,S_L_leg_y,S_L_leg_z])
    S_Veri_L_leg = np.linalg.det(S_LCS_L_leg)
        
    # L_foot LCS
    S_L_foot_y = (D_G_L_toe - D_G_L_heel) /  np.linalg.norm((D_G_L_toe - D_G_L_heel), ord=2)
    S_L_foot_1 = (D_G_L_toe - D_G_M_L_ankle) / np.linalg.norm((D_G_L_toe - D_G_M_L_ankle), ord=2)
    S_L_foot_x = np.cross(S_L_foot_1,S_L_foot_y) / np.linalg.norm(np.cross(S_L_foot_1,S_L_foot_y), ord=2)
    S_L_foot_z = np.cross(S_L_foot_x,S_L_foot_y) / np.linalg.norm(np.cross(S_L_foot_x,S_L_foot_y), ord=2)
        
    S_LCS_L_foot = np.array([S_L_foot_x,S_L_foot_y,S_L_foot_z])
    S_Veri_L_foot = np.linalg.det(S_LCS_L_foot)
        
    S_L_le_R_fo = np.dot(S_LCS_L_foot, S_LCS_L_leg.T)
    
    D_L_anklejoint = D_G_M_L_ankle
    
    D_G_L_toe = np.dot(np.dot(RM_3,S_L_le_R_fo.T),fo_L_toe2) + D_L_anklejoint
    D_G_L_heel = np.dot(np.dot(RM_3,S_L_le_R_fo.T),fo_L_heel2) + D_L_anklejoint
    
    #%%
    #R_lower_body
    #Static rotation matrix
    S_R_pel_R_th = np.dot(S_LCS_R_thigh, S_LCS_pelvis.T)
    S_L_pel_R_th = np.dot(S_LCS_L_thigh, S_LCS_pelvis.T)
    
    # transform from pel to distal
    a1 = math.radians(dat_angles[i,22])
    b1 = math.radians(dat_angles[i,23])
    c1 = math.radians(dat_angles[i,24])
        
    a2 = math.radians(dat_angles[i,28])
    b2 = math.radians(dat_angles[i,29])
    c2 = math.radians(dat_angles[i,30])
        
    a3 = math.radians(dat_angles[i,34])
    b3 = math.radians(dat_angles[i,35])
    c3 = math.radians(dat_angles[i,36])
    
    # first rotation
    RM_x1 = np.array([[1, 0, 0],
                      [0, np.cos(a1), np.sin(a1)],
                      [0, -1*np.sin(a1), np.cos(a1)]])
        
    RM_y1 = np.array([[np.cos(b1), 0, -1*np.sin(b1)],
                       [0, 1, 0],
                       [np.sin(b1), 0, np.cos(b1)]])
        
    RM_z1 = np.array([[np.cos(c1), np.sin(c1), 0],
                       [-1*np.sin(c1), np.cos(c1), 0],
                       [0, 0, 1]])
        
    RM_1 = np.array(np.dot(np.dot(RM_z1, RM_y1), RM_x1))
        
    # second rotation    
    RM_x2 = np.array([[1, 0, 0],
                      [0, np.cos(a2), np.sin(a2)],
                      [0, -1*np.sin(a2), np.cos(a2)]])
        
    RM_y2 = np.array([[np.cos(b2), 0, -1*np.sin(b2)],
                       [0, 1, 0],
                       [np.sin(b2), 0, np.cos(b2)]])
        
    RM_z2 = np.array([[np.cos(c2), np.sin(c2), 0],
                       [-1*np.sin(c2), np.cos(c2), 0],
                       [0, 0, 1]])
        
    RM_2 = np.array(np.dot(np.dot(RM_z2, RM_y2), RM_x2))
        
    # third rotation    
    RM_x3 = np.array([[1, 0, 0],
                      [0, np.cos(a3), np.sin(a3)],
                      [0, -1*np.sin(a3), np.cos(a3)]])
        
    RM_y3 = np.array([[np.cos(b3), 0, -1*np.sin(b3)],
                       [0, 1, 0],
                       [np.sin(b3), 0, np.cos(b3)]])
        
    RM_z3 = np.array([[np.cos(c3), np.sin(c3), 0],
                       [-1*np.sin(c3), np.cos(c3), 0],
                       [0, 0, 1]])
    
    RM_3 = np.array(np.dot(np.dot(RM_z3, RM_y3), RM_x3))
    
    
    #%%
    # first rotation of R_lower-body
    # position from thigh LCS
    th_R_GT = np.dot(S_LCS_R_thigh.T,S_R_GT)
    th_R_knee_lat = np.dot(S_LCS_R_thigh.T,S_R_knee_lat)
    th_R_knee_med = np.dot(S_LCS_R_thigh.T,S_R_knee_med)
    
    th_R_ankle_lat = np.dot(S_LCS_R_thigh.T,S_R_ankle_lat)
    th_R_ankle_med = np.dot(S_LCS_R_thigh.T,S_R_ankle_med)
    
    th_R_toe = np.dot(S_LCS_R_thigh.T,S_R_toe)
    th_R_heel = np.dot(S_LCS_R_thigh.T,S_R_heel)
    
    # origin point as hip joint
    R_hip = S_R_GT
    
    th_R_knee_lat2 = th_R_knee_lat - R_hip
    th_R_knee_med2 = th_R_knee_med - R_hip
    
    th_R_ankle_lat2 = th_R_ankle_lat - R_hip
    th_R_ankle_med2 = th_R_ankle_med - R_hip
    
    th_R_toe2 = th_R_toe - R_hip
    th_R_heel2 = th_R_heel - R_hip
    
    # Dynamic position
    D_R_hip = D_R_GT
    
    D_G_R_knee_lat = np.dot(np.dot(RM_1,S_R_pel_R_th.T),th_R_knee_lat2) + D_R_hip
    D_G_R_knee_med = np.dot(np.dot(RM_1,S_R_pel_R_th.T),th_R_knee_med2) + D_R_hip
    D_G_M_R_knee = (D_G_R_knee_lat + D_G_R_knee_med) / 2
    
    D_G_R_ankle_lat = np.dot(np.dot(RM_1,S_R_pel_R_th.T),th_R_ankle_lat2) + D_R_hip
    D_G_R_ankle_med = np.dot(np.dot(RM_1,S_R_pel_R_th.T),th_R_ankle_med2) + D_R_hip
    D_G_M_R_ankle = (D_G_R_ankle_lat + D_G_R_ankle_med) / 2
    
    D_G_R_toe = np.dot(np.dot(RM_1,S_R_pel_R_th.T),th_R_toe2) + D_R_hip
    D_G_R_heel = np.dot(np.dot(RM_1,S_R_pel_R_th.T),th_R_heel2) + D_R_hip
    
    #%%
    # second rotation of R_lower-body
    # position from leg LCS
    le_R_ankle_lat = np.dot(S_LCS_R_leg.T,D_G_R_ankle_lat)
    le_R_ankle_med = np.dot(S_LCS_R_leg.T,D_G_R_ankle_med)
    
    le_R_toe = np.dot(S_LCS_R_leg.T,D_G_R_toe)
    le_R_heel = np.dot(S_LCS_R_leg.T,D_G_R_heel)
    
    # origin point as lumber joint
    R_knee = D_G_M_R_knee
    
    le_R_ankle_lat2 = le_R_ankle_lat - R_knee
    le_R_ankle_med2 = le_R_ankle_med - R_knee
    
    le_R_toe2 = le_R_toe - R_knee
    le_R_heel2 = le_R_heel - R_knee
    
    # Dynamic position
    # R_thigh LCS
    S_R_thigh_x = (D_G_R_knee_lat - D_G_R_knee_med) / np.linalg.norm((D_G_R_knee_lat - D_G_R_knee_med), ord=2)
    S_R_thigh_1 = (S_R_GT - D_G_M_R_knee) / np.linalg.norm((S_R_GT - D_G_M_R_knee), ord=2)
    S_R_thigh_y = np.cross(S_R_thigh_1,S_R_thigh_x) / np.linalg.norm(np.cross(S_R_thigh_1,S_R_thigh_x), ord=2)
    S_R_thigh_z = np.cross(S_R_thigh_x,S_R_thigh_y) / np.linalg.norm(np.cross(S_R_thigh_x,S_R_thigh_y), ord=2)
        
    S_LCS_R_thigh = np.array([S_R_thigh_x,S_R_thigh_y,S_R_thigh_z])
    S_Veri_R_thigh = np.linalg.det(S_LCS_R_thigh)
            
    # R_leg LCS
    S_R_leg_x = (D_G_R_ankle_lat - D_G_R_ankle_med) / np.linalg.norm((D_G_R_ankle_lat - D_G_R_ankle_med), ord=2)
    S_R_leg_1 = (D_G_M_R_knee - D_G_M_R_ankle) / np.linalg.norm((D_G_M_R_knee - D_G_M_R_ankle), ord=2)
    S_R_leg_y = np.cross(S_R_leg_1,S_R_leg_x) / np.linalg.norm(np.cross(S_R_leg_1,S_R_leg_x), ord=2)
    S_R_leg_z = np.cross(S_R_leg_x,S_R_leg_y) / np.linalg.norm(np.cross(S_R_leg_x,S_R_leg_y), ord=2)
        
    S_LCS_R_leg = np.array([S_R_leg_x,S_R_leg_y,S_R_leg_z])
    S_Veri_R_leg = np.linalg.det(S_LCS_R_leg)
    
    S_R_th_R_le = np.dot(S_LCS_R_leg, S_LCS_R_thigh.T)
    
    D_R_kneejoint = D_G_M_R_knee
    
    D_G_R_ankle_lat = np.dot(np.dot(RM_2,S_R_th_R_le.T),le_R_ankle_lat2) + D_R_kneejoint
    D_G_R_ankle_med = np.dot(np.dot(RM_2,S_R_th_R_le.T),le_R_ankle_med2) + D_R_kneejoint
    
    D_G_R_toe = np.dot(np.dot(RM_2,S_R_th_R_le.T),le_R_toe2) + D_R_kneejoint
    D_G_R_heel = np.dot(np.dot(RM_2,S_R_th_R_le.T),le_R_heel2) + D_R_kneejoint
    
    #%%
    # third rotation of upper-body
    # position from foot LCS
    fo_R_toe = np.dot(S_LCS_R_foot.T,D_G_R_toe)
    fo_R_heel = np.dot(S_LCS_R_foot.T,D_G_R_heel)
    
    # origin point as ankle joint
    D_G_M_R_ankle = (D_G_R_ankle_lat + D_G_R_ankle_med) / 2
    R_ankle = D_G_M_R_ankle
    
    fo_R_toe2 = fo_R_toe - R_ankle
    fo_R_heel2 = fo_R_heel - R_ankle
    
    # Dynamic position
    # R_leg LCS
    S_R_leg_x = (D_G_R_ankle_lat - D_G_R_ankle_med) / np.linalg.norm((D_G_R_ankle_lat - D_G_R_ankle_med), ord=2)
    S_R_leg_1 = (D_G_M_R_knee - D_G_M_R_ankle) / np.linalg.norm((D_G_M_R_knee - D_G_M_R_ankle), ord=2)
    S_R_leg_y = np.cross(S_R_leg_1,S_R_leg_x) / np.linalg.norm(np.cross(S_R_leg_1,S_R_leg_x), ord=2)
    S_R_leg_z = np.cross(S_R_leg_x,S_R_leg_y) / np.linalg.norm(np.cross(S_R_leg_x,S_R_leg_y), ord=2)
        
    S_LCS_R_leg = np.array([S_R_leg_x,S_R_leg_y,S_R_leg_z])
    S_Veri_R_leg = np.linalg.det(S_LCS_R_leg)
    
    # R_foot LCS        
    S_R_foot_y = (D_G_R_toe - D_G_R_heel) /  np.linalg.norm((D_G_R_toe - D_G_R_heel), ord=2)
    S_R_foot_1 = (D_G_R_toe - D_G_M_R_ankle) / np.linalg.norm((D_G_R_toe - D_G_M_R_ankle), ord=2)
    S_R_foot_x = np.cross(S_R_foot_1,S_R_foot_y) / np.linalg.norm(np.cross(S_R_foot_1,S_R_foot_y), ord=2)
    S_R_foot_z = np.cross(S_R_foot_x,S_R_foot_y) / np.linalg.norm(np.cross(S_R_foot_x,S_R_foot_y), ord=2)
        
    S_LCS_R_foot = np.array([S_R_foot_x,S_R_foot_y,S_R_foot_z])
    S_Veri_R_foot = np.linalg.det(S_LCS_R_foot)
    
    S_R_le_R_fo = np.dot(S_LCS_R_foot, S_LCS_R_leg.T)
    
    D_R_anklejoint = D_G_M_R_ankle
    
    D_G_R_toe = np.dot(np.dot(RM_3,S_R_le_R_fo.T),fo_R_toe2) + D_R_anklejoint
    D_G_R_heel = np.dot(np.dot(RM_3,S_R_le_R_fo.T),fo_R_heel2) + D_R_anklejoint
    
    #%%
    a = math.radians(-30)
    b = math.radians(0)
    c = math.radians(0)
    
    RM_x = np.array([[1, 0, 0],
                     [0, np.cos(a), np.sin(a)],
                     [0, -1*np.sin(a), np.cos(a)]])
        
    RM_y = np.array([[np.cos(b), 0, -1*np.sin(b)],
                      [0, 1, 0],
                      [np.sin(b), 0, np.cos(b)]])
        
    RM_z = np.array([[np.cos(c), np.sin(c), 0],
                      [-1*np.sin(c), np.cos(c), 0],
                      [0, 0, 1]])
        
    RM = np.array(np.dot(np.dot(RM_z, RM_y), RM_x))
    
    D_G_IJ = np.dot(D_G_IJ, RM)
    D_G_PX = np.dot(D_G_PX, RM)
    D_G_C7 = np.dot(D_G_C7, RM)
    D_G_T7 = np.dot(D_G_T7, RM)
    D_G_R_shoulder = np.dot(D_G_R_shoulder, RM)
    D_G_L_shoulder = np.dot(D_G_L_shoulder, RM) 
    D_R_ASIS = np.dot(D_R_ASIS, RM)
    D_L_ASIS = np.dot(D_L_ASIS, RM)
    D_M_PSIS = np.dot(D_M_PSIS, RM)
    
    S_R_GT = np.dot(S_R_GT, RM)
    S_L_GT = np.dot(S_L_GT, RM)
    D_G_R_knee_lat = np.dot(D_G_R_knee_lat, RM)
    D_G_R_knee_med = np.dot(D_G_R_knee_med, RM)
    D_G_L_knee_lat = np.dot(D_G_L_knee_lat, RM)
    D_G_L_knee_med = np.dot(D_G_L_knee_med, RM)
    D_G_R_ankle_lat = np.dot(D_G_R_ankle_lat, RM)
    D_G_R_ankle_med = np.dot(D_G_R_ankle_med, RM)
    D_G_L_ankle_lat = np.dot(D_G_L_ankle_lat, RM)
    D_G_L_ankle_med = np.dot(D_G_L_ankle_med, RM)
    D_G_R_toe = np.dot(D_G_R_toe, RM)
    D_G_R_heel = np.dot(D_G_R_heel, RM)
    D_G_L_toe = np.dot(D_G_L_toe, RM)
    D_G_L_heel = np.dot(D_G_L_heel, RM)
    
    D_G_R_elbow_lat = np.dot(D_G_R_elbow_lat, RM)
    D_G_R_elbow_med = np.dot(D_G_R_elbow_med, RM)
    D_G_L_elbow_lat = np.dot(D_G_L_elbow_lat, RM)
    D_G_L_elbow_med = np.dot(D_G_L_elbow_med, RM)
    D_G_R_wrist_lat = np.dot(D_G_R_wrist_lat, RM)
    D_G_R_wrist_med = np.dot(D_G_R_wrist_med, RM)
    D_G_L_wrist_lat = np.dot(D_G_L_wrist_lat, RM)
    D_G_L_wrist_med = np.dot(D_G_L_wrist_med, RM)
    D_G_R_MC3 = np.dot(D_G_R_MC3, RM)
    D_G_L_MC3 = np.dot(D_G_L_MC3, RM)   
       
    #%%

# D_G_R_leg
    D_R_leg1 = D_ax2.plot([D_R_ASIS[0],S_R_GT[0]],[D_R_ASIS[1],S_R_GT[1]],[D_R_ASIS[2],S_R_GT[2]],marker="o",color="g",lw=2)
    D_R_leg2 = D_ax2.plot([S_R_GT[0],D_G_R_knee_lat[0]],[S_R_GT[1],D_G_R_knee_lat[1]],[S_R_GT[2],D_G_R_knee_lat[2]],marker="o",color="g",lw=2)            
    D_R_leg3 = D_ax2.plot([S_R_GT[0],D_G_R_knee_med[0]],[S_R_GT[1],D_G_R_knee_med[1]],[S_R_GT[2],D_G_R_knee_med[2]],marker="o",color="g",lw=2)
    D_R_leg4 = D_ax2.plot([D_G_R_knee_lat[0],D_G_R_ankle_lat[0]],[D_G_R_knee_lat[1],D_G_R_ankle_lat[1]],[D_G_R_knee_lat[2],D_G_R_ankle_lat[2]],marker="o",color="g",lw=2)
    D_R_leg5 = D_ax2.plot([D_G_R_knee_med[0],D_G_R_ankle_med[0]],[D_G_R_knee_med[1],D_G_R_ankle_med[1]],[D_G_R_knee_med[2],D_G_R_ankle_med[2]],marker="o",color="g",lw=2)
    D_R_leg6 = D_ax2.plot([D_G_R_ankle_lat[0],D_G_R_ankle_med[0]],[D_G_R_ankle_lat[1],D_G_R_ankle_med[1]],[D_G_R_ankle_lat[2],D_G_R_ankle_med[2]],marker="o",color="g",lw=2)
    D_R_leg7 = D_ax2.plot([D_G_R_ankle_lat[0],D_G_R_toe[0]],[D_G_R_ankle_lat[1],D_G_R_toe[1]],[D_G_R_ankle_lat[2],D_G_R_toe[2]],marker="o",color="g",lw=2)
    D_R_leg8 = D_ax2.plot([D_G_R_ankle_lat[0],D_G_R_heel[0]],[D_G_R_ankle_lat[1],D_G_R_heel[1]],[D_G_R_ankle_lat[2],D_G_R_heel[2]],marker="o",color="g",lw=2)
    D_R_leg9 = D_ax2.plot([D_G_R_ankle_med[0],D_G_R_toe[0]],[D_G_R_ankle_med[1],D_G_R_toe[1]],[D_G_R_ankle_med[2],D_G_R_toe[2]],marker="o",color="g",lw=2)
    D_R_leg10 = D_ax2.plot([D_G_R_ankle_med[0],D_G_R_heel[0]],[D_G_R_ankle_med[1],D_G_R_heel[1]],[D_G_R_ankle_med[2],D_G_R_heel[2]],marker="o",color="g",lw=2)
    D_R_leg11 = D_ax2.plot([D_G_R_toe[0],D_G_R_heel[0]],[D_G_R_toe[1],D_G_R_heel[1]],[D_G_R_toe[2],D_G_R_heel[2]],marker="o",color="g",lw=2)

# D_G_L_D_G_Leg
    D_L_leg1 = D_ax2.plot([D_L_ASIS[0],S_L_GT[0]],[D_L_ASIS[1],S_L_GT[1]],[D_L_ASIS[2],S_L_GT[2]],marker="o",color="b",lw=2)
    D_L_leg2 = D_ax2.plot([S_L_GT[0],D_G_L_knee_lat[0]],[S_L_GT[1],D_G_L_knee_lat[1]],[S_L_GT[2],D_G_L_knee_lat[2]],marker="o",color="b",lw=2)    
    D_L_leg3 = D_ax2.plot([S_L_GT[0],D_G_L_knee_med[0]],[S_L_GT[1],D_G_L_knee_med[1]],[S_L_GT[2],D_G_L_knee_med[2]],marker="o",color="b",lw=2)
    D_L_leg4 = D_ax2.plot([D_G_L_knee_lat[0],D_G_L_ankle_lat[0]],[D_G_L_knee_lat[1],D_G_L_ankle_lat[1]],[D_G_L_knee_lat[2],D_G_L_ankle_lat[2]],marker="o",color="b",lw=2)
    D_L_leg5 = D_ax2.plot([D_G_L_knee_med[0],D_G_L_ankle_med[0]],[D_G_L_knee_med[1],D_G_L_ankle_med[1]],[D_G_L_knee_med[2],D_G_L_ankle_med[2]],marker="o",color="b",lw=2)
    D_L_leg6 = D_ax2.plot([D_G_L_ankle_lat[0],D_G_L_ankle_med[0]],[D_G_L_ankle_lat[1],D_G_L_ankle_med[1]],[D_G_L_ankle_lat[2],D_G_L_ankle_med[2]],marker="o",color="b",lw=2)
    D_L_leg7 = D_ax2.plot([D_G_L_ankle_lat[0],D_G_L_toe[0]],[D_G_L_ankle_lat[1],D_G_L_toe[1]],[D_G_L_ankle_lat[2],D_G_L_toe[2]],marker="o",color="b",lw=2)
    D_L_leg8 = D_ax2.plot([D_G_L_ankle_lat[0],D_G_L_heel[0]],[D_G_L_ankle_lat[1],D_G_L_heel[1]],[D_G_L_ankle_lat[2],D_G_L_heel[2]],marker="o",color="b",lw=2)
    D_L_leg9 = D_ax2.plot([D_G_L_ankle_med[0],D_G_L_toe[0]],[D_G_L_ankle_med[1],D_G_L_toe[1]],[D_G_L_ankle_med[2],D_G_L_toe[2]],marker="o",color="b",lw=2)
    D_L_leg10 = D_ax2.plot([D_G_L_ankle_med[0],D_G_L_heel[0]],[D_G_L_ankle_med[1],D_G_L_heel[1]],[D_G_L_ankle_med[2],D_G_L_heel[2]],marker="o",color="b",lw=2)
    D_L_leg11 = D_ax2.plot([D_G_L_toe[0],D_G_L_heel[0]],[D_G_L_toe[1],D_G_L_heel[1]],[D_G_L_toe[2],D_G_L_heel[2]],marker="o",color="b",lw=2)
    
# R_arm
    D_R_arm1 = D_ax2.plot([D_G_R_shoulder[0],D_G_R_elbow_lat[0]],[D_G_R_shoulder[1],D_G_R_elbow_lat[1]],[D_G_R_shoulder[2],D_G_R_elbow_lat[2]],marker="o",color="r",lw=2)
    D_R_arm2 = D_ax2.plot([D_G_R_shoulder[0],D_G_R_elbow_med[0]],[D_G_R_shoulder[1],D_G_R_elbow_med[1]],[D_G_R_shoulder[2],D_G_R_elbow_med[2]],marker="o",color="r",lw=2)
    D_R_arm3 = D_ax2.plot([D_G_R_elbow_lat[0],D_G_R_elbow_med[0]],[D_G_R_elbow_lat[1],D_G_R_elbow_med[1]],[D_G_R_elbow_lat[2],D_G_R_elbow_med[2]],marker="o",color="r",lw=2)    
    D_R_arm4 = D_ax2.plot([D_G_R_elbow_lat[0],D_G_R_wrist_lat[0]],[D_G_R_elbow_lat[1],D_G_R_wrist_lat[1]],[D_G_R_elbow_lat[2],D_G_R_wrist_lat[2]],marker="o",color="r",lw=2)
    D_R_arm5 = D_ax2.plot([D_G_R_elbow_med[0],D_G_R_wrist_med[0]],[D_G_R_elbow_med[1],D_G_R_wrist_med[1]],[D_G_R_elbow_med[2],D_G_R_wrist_med[2]],marker="o",color="r",lw=2)
    D_R_arm6 = D_ax2.plot([D_G_R_wrist_lat[0],D_G_R_wrist_med[0]],[D_G_R_wrist_lat[1],D_G_R_wrist_med[1]],[D_G_R_wrist_lat[2],D_G_R_wrist_med[2]],marker="o",color="r",lw=2)    
    D_R_arm7 = D_ax2.plot([D_G_R_wrist_lat[0],D_G_R_MC3[0]],[D_G_R_wrist_lat[1],D_G_R_MC3[1]],[D_G_R_wrist_lat[2],D_G_R_MC3[2]],marker="o",color="r",lw=2)
    D_R_arm8 = D_ax2.plot([D_G_R_wrist_med[0],D_G_R_MC3[0]],[D_G_R_wrist_med[1],D_G_R_MC3[1]],[D_G_R_wrist_med[2],D_G_R_MC3[2]],marker="o",color="r",lw=2)

# L_arm
    D_L_arm1 = D_ax2.plot([D_G_L_shoulder[0],D_G_L_elbow_lat[0]],[D_G_L_shoulder[1],D_G_L_elbow_lat[1]],[D_G_L_shoulder[2],D_G_L_elbow_lat[2]],marker="o",color="y",lw=2)
    D_L_arm2 = D_ax2.plot([D_G_L_shoulder[0],D_G_L_elbow_med[0]],[D_G_L_shoulder[1],D_G_L_elbow_med[1]],[D_G_L_shoulder[2],D_G_L_elbow_med[2]],marker="o",color="y",lw=2)
    D_L_arm3 = D_ax2.plot([D_G_L_elbow_lat[0],D_G_L_elbow_med[0]],[D_G_L_elbow_lat[1],D_G_L_elbow_med[1]],[D_G_L_elbow_lat[2],D_G_L_elbow_med[2]],marker="o",color="y",lw=2)    
    D_L_arm4 = D_ax2.plot([D_G_L_elbow_lat[0],D_G_L_wrist_lat[0]],[D_G_L_elbow_lat[1],D_G_L_wrist_lat[1]],[D_G_L_elbow_lat[2],D_G_L_wrist_lat[2]],marker="o",color="y",lw=2)
    D_L_arm5 = D_ax2.plot([D_G_L_elbow_med[0],D_G_L_wrist_med[0]],[D_G_L_elbow_med[1],D_G_L_wrist_med[1]],[D_G_L_elbow_med[2],D_G_L_wrist_med[2]],marker="o",color="y",lw=2)
    D_L_arm6 = D_ax2.plot([D_G_L_wrist_lat[0],D_G_L_wrist_med[0]],[D_G_L_wrist_lat[1],D_G_L_wrist_med[1]],[D_G_L_wrist_lat[2],D_G_L_wrist_med[2]],marker="o",color="y",lw=2)    
    D_L_arm7 = D_ax2.plot([D_G_L_wrist_lat[0],D_G_L_MC3[0]],[D_G_L_wrist_lat[1],D_G_L_MC3[1]],[D_G_L_wrist_lat[2],D_G_L_MC3[2]],marker="o",color="y",lw=2)
    D_L_arm8 = D_ax2.plot([D_G_L_wrist_med[0],D_G_L_MC3[0]],[D_G_L_wrist_med[1],D_G_L_MC3[1]],[D_G_L_wrist_med[2],D_G_L_MC3[2]],marker="o",color="y",lw=2)
  
# D_body
    D_body1 = D_ax2.plot([D_G_R_shoulder[0],D_G_IJ[0]],[D_G_R_shoulder[1],D_G_IJ[1]],[D_G_R_shoulder[2],D_G_IJ[2]],marker="o",color="k",lw=2)    
    D_body2 = D_ax2.plot([D_G_L_shoulder[0],D_G_IJ[0]],[D_G_L_shoulder[1],D_G_IJ[1]],[D_G_L_shoulder[2],D_G_IJ[2]],marker="o",color="k",lw=2)
    D_body3 = D_ax2.plot([D_G_R_shoulder[0],D_G_C7[0]],[D_G_R_shoulder[1],D_G_C7[1]],[D_G_R_shoulder[2],D_G_C7[2]],marker="o",color="k",lw=2)    
    D_body4 = D_ax2.plot([D_G_L_shoulder[0],D_G_C7[0]],[D_G_L_shoulder[1],D_G_C7[1]],[D_G_L_shoulder[2],D_G_C7[2]],marker="o",color="k",lw=2)
    D_body5 = D_ax2.plot([D_G_R_shoulder[0],D_R_ASIS[0]],[D_G_R_shoulder[1],D_R_ASIS[1]],[D_G_R_shoulder[2],D_R_ASIS[2]],marker="o",color="k",lw=2)    
    D_body6 = D_ax2.plot([D_G_L_shoulder[0],D_L_ASIS[0]],[D_G_L_shoulder[1],D_L_ASIS[1]],[D_G_L_shoulder[2],D_L_ASIS[2]],marker="o",color="k",lw=2)
    D_body7 = D_ax2.plot([D_G_IJ[0],D_G_PX[0]],[D_G_IJ[1],D_G_PX[1]],[D_G_IJ[2],D_G_PX[2]],marker="o",color="k",lw=2)
    D_body8 = D_ax2.plot([D_G_C7[0],D_G_T7[0]],[D_G_C7[1],D_G_T7[1]],[D_G_C7[2],D_G_T7[2]],marker="o",color="k",lw=2)    
    D_body9 = D_ax2.plot([D_G_T7[0],D_M_PSIS[0]],[D_G_T7[1],D_M_PSIS[1]],[D_G_T7[2],D_M_PSIS[2]],marker="o",color="k",lw=2)
    D_body10 = D_ax2.plot([D_G_PX[0],D_R_ASIS[0]],[D_G_PX[1],D_R_ASIS[1]],[D_G_PX[2],D_R_ASIS[2]],marker="o",color="k",lw=2)
    D_body11 = D_ax2.plot([D_G_PX[0],D_L_ASIS[0]],[D_G_PX[1],D_L_ASIS[1]],[D_G_PX[2],D_L_ASIS[2]],marker="o",color="k",lw=2)
    D_body12 = D_ax2.plot([D_R_ASIS[0],D_M_PSIS[0]],[D_R_ASIS[1],D_M_PSIS[1]],[D_R_ASIS[2],D_M_PSIS[2]],marker="o",color="k",lw=2)    
    D_body13 = D_ax2.plot([D_L_ASIS[0],D_M_PSIS[0]],[D_L_ASIS[1],D_M_PSIS[1]],[D_L_ASIS[2],D_M_PSIS[2]],marker="o",color="k",lw=2)
    D_body14 = D_ax2.plot([D_L_ASIS[0],D_R_ASIS[0]],[D_L_ASIS[1],D_R_ASIS[1]],[D_L_ASIS[2],D_R_ASIS[2]],marker="o",color="k",lw=2)
    D_body15 = D_ax2.plot([D_G_R_shoulder[0],D_G_L_shoulder[0]],[D_G_R_shoulder[1],D_G_L_shoulder[1]],[D_G_R_shoulder[2],D_G_L_shoulder[2]],marker="o",color="k",lw=2)    
    
    D_frames2.append(D_body1+D_body2+D_body3+D_body4+D_body5+D_body6+D_body7+D_body8+D_body9+D_body10+D_body11+D_body12+D_body13+D_body14+D_body15+
                  D_R_arm1+D_R_arm2+D_R_arm3+D_R_arm4+D_R_arm5+D_R_arm6+D_R_arm7+D_R_arm8+
                  D_L_arm1+D_L_arm2+D_L_arm3+D_L_arm4+D_L_arm5+D_L_arm6+D_L_arm7+D_L_arm8+
                  D_R_leg1+D_R_leg2+D_R_leg3+D_R_leg4+D_R_leg5+D_R_leg6+D_R_leg7+D_R_leg8+D_R_leg9+D_R_leg10+D_R_leg11+
                  D_L_leg1+D_L_leg2+D_L_leg3+D_L_leg4+D_L_leg5+D_L_leg6+D_L_leg7+D_L_leg8+D_L_leg9+D_L_leg10+D_L_leg11)
                 
# 軸ラベルの設定
D_ax2.set_xlabel("X-axis(mm)")
D_ax2.set_ylabel("Y-axis(mm)")
D_ax2.set_zlabel("Z-axis(mm)")

D_ax2.set_xticks([-150, 0.0, 50])
D_ax2.set_yticks([-50, 0.0, 150])
D_ax2.set_zticks([-10, 0.0, 150])

plt.title("Right arm")
#plt.axis('equal')

D_ani2 = animation.ArtistAnimation(D_fig2, D_frames2, interval=0.01)
plt.show()

#D_ani2.save(name + '_' + syn + ".gif", writer="imagemagick")
#D_ani2.save(name + '_' + syn + ".mp4", writer="ffmpeg")