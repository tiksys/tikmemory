#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:58:02 2020

@author: taiki
"""

import numpy as np
import pandas as pd
import math

name = 'Sakai' 
No = 9
data_val = np.genfromtxt('Pitching_' + name + '0' + str(No) + '_trimed.csv', skip_header = 6, delimiter = ',')

angles_df = pd.DataFrame({'L_shoulder_exro':[],'L_shoulder_add':[],'L_shoulder_Had':[],
                       'R_shoulder_inro':[],'R_shoulder_abd':[],'R_shoulder_Hab':[],
                       'L_elbow_rot':[],'L_elbow_add':[],'L_elbow_fl':[],
                       'R_elbow_rot':[],'R_elbow_abd':[],'R_elbow_ex':[],
                       'L_wrist_rot':[],'L_wrist_abd':[],'L_wrist_fl':[],
                       'R_wrist_rot':[],'R_wrist_add':[],'R_wrist_ex':[],
                       'L_hip_ex':[],'L_hip_add':[],'L_hip_inro':[],
                       'R_hip_ex':[],'R_hip_abd':[],'R_hip_exro':[],
                       'L_knee_fl':[],'L_knee_add':[],'L_knee_inro':[],
                       'R_knee_fl':[],'R_knee_abd':[],'R_knee_exro':[],
                       'L_ankle_ex':[],'L_ankle_add':[],'L_ankle_inrr':[],
                       'R_ankle_ex':[],'R_ankle_abd':[],'R_ankle_exro':[],
                       'forward_tilt':[],'L_tilt':[],'R_rotation':[]})

for i in range(0,len(data_val[:,0])):
    
    print('***************', str(i) , 'frame ***************')
#%%    
    Top_head = np.array([data_val[i,2],data_val[i,3],data_val[i,4]])
    R_head = np.array([data_val[i,11],data_val[i,12],data_val[i,13]])
    L_head = np.array([data_val[i,14],data_val[i,15],data_val[i,16]])

# body
    IJ = np.array([data_val[i,17],data_val[i,18],data_val[i,19]])
    PX = np.array([data_val[i,20],data_val[i,21],data_val[i,22]])
    C7= np.array([data_val[i,23],data_val[i,24],data_val[i,25]])
    T7 = np.array([data_val[i,26],data_val[i,27],data_val[i,28]])
    R_shoulder = np.array([data_val[i,29],data_val[i,30],data_val[i,31]])
    L_shoulder = np.array([data_val[i,32],data_val[i,33],data_val[i,34]])
    R_ASIS = np.array([data_val[i,65],data_val[i,66],data_val[i,67]])
    L_ASIS = np.array([data_val[i,68],data_val[i,69],data_val[i,70]])
    M_PSIS = np.array([data_val[i,71],data_val[i,72],data_val[i,73]])
    M_ASIS = np.array((L_ASIS + R_ASIS) / 2)
    M_shoulder = (R_shoulder + L_shoulder) / 2
    M_C7IJ = np.array((C7 + IJ) / 2)
    M_T7PX = np.array((T7 + PX) / 2)

# R_arm
    R_elbow_lat = np.array([data_val[i,35],data_val[i,36],data_val[i,37]])
    R_elbow_med = np.array([data_val[i,38],data_val[i,39],data_val[i,40]])
    R_wrist_lat = np.array([data_val[i,47],data_val[i,48],data_val[i,49]])
    R_wrist_med = np.array([data_val[i,50],data_val[i,51],data_val[i,52]])
    R_MC3 = np.array([data_val[i,59],data_val[i,60],data_val[i,61]])
    M_R_elbow = np.array((R_elbow_lat + R_elbow_med) / 2)
    M_R_wrist = np.array((R_wrist_lat + R_wrist_med) / 2)

# L_arm
    L_elbow_lat = np.array([data_val[i,41],data_val[i,42],data_val[i,43]])
    L_elbow_med = np.array([data_val[i,44],data_val[i,45],data_val[i,46]])
    L_wrist_lat = np.array([data_val[i,53],data_val[i,54],data_val[i,55]])
    L_wrist_med = np.array([data_val[i,56],data_val[i,57],data_val[i,58]])
    L_MC3 = np.array([data_val[i,62],data_val[i,63],data_val[i,64]])
    M_L_elbow = np.array((L_elbow_lat + L_elbow_med) / 2)
    M_L_wrist = np.array((L_wrist_lat + L_wrist_med) / 2)

# R_leg
    R_GT = np.array([data_val[i,74],data_val[i,75],data_val[i,76]])
    R_knee_lat = np.array([data_val[i,80],data_val[i,81],data_val[i,82]])
    R_knee_med = np.array([data_val[i,83],data_val[i,84],data_val[i,85]])
    R_shank = np.array([data_val[i,92],data_val[i,93],data_val[i,94]])
    R_ankle_lat = np.array([data_val[i,98],data_val[i,99],data_val[i,100]])
    R_ankle_med = np.array([data_val[i,101],data_val[i,102],data_val[i,103]])
    R_MT3 = np.array([data_val[i,110],data_val[i,111],data_val[i,112]])
    R_heel = np.array([data_val[i,116],data_val[i,117],data_val[i,118]])
    M_R_knee = np.array((R_knee_lat + R_knee_med) / 2)
    M_R_ankle = np.array((R_ankle_lat + R_ankle_med) / 2)
    

# L_leg
    L_GT = np.array([data_val[i,77],data_val[i,78],data_val[i,79]])
    L_knee_lat = np.array([data_val[i,86],data_val[i,87],data_val[i,88]])
    L_knee_med = np.array([data_val[i,89],data_val[i,90],data_val[i,91]])
    L_shank = np.array([data_val[i,95],data_val[i,96],data_val[i,97]])
    L_ankle_lat = np.array([data_val[i,104],data_val[i,105],data_val[i,106]])
    L_ankle_med = np.array([data_val[i,107],data_val[i,108],data_val[i,109]])
    L_MT3 = np.array([data_val[i,113],data_val[i,114],data_val[i,115]])
    L_heel = np.array([data_val[i,119],data_val[i,120],data_val[i,121]])
    M_L_knee = np.array((L_knee_lat + L_knee_med) / 2)
    M_L_ankle = np.array((L_ankle_lat + L_ankle_med) / 2)

# hip_center
    #R_hipD = np.array(R_GT /3 + R_ASIS *2/3)
    #L_hipD = np.array(L_GT /3 + L_ASIS *2/3)
    #R_C_hip = (L_hipD - R_hipD) * 0.18
    #L_C_hip = (R_hipD - L_hipD) * 0.18
    
#%%
# make LCS
# pelvis LCS
    pelvis_x = (R_ASIS - L_ASIS) / np.linalg.norm((R_ASIS - L_ASIS), ord=2)
    pelvis_1 = (M_ASIS - M_PSIS) / np.linalg.norm((M_ASIS - M_PSIS), ord=2)
    pelvis_z = np.cross(pelvis_x,pelvis_1) / np.linalg.norm(np.cross(pelvis_x,pelvis_1), ord=2)
    pelvis_y = np.cross(pelvis_z,pelvis_x) / np.linalg.norm(np.cross(pelvis_z,pelvis_x), ord=2)
    
    LCS_pelvis = np.matrix([pelvis_x, pelvis_y, pelvis_z])
    Veri_pelvis = np.linalg.det(LCS_pelvis)
    
# L_thigh LCS
    L_thigh_x = (L_knee_med - L_knee_lat) / np.linalg.norm((L_knee_med - L_knee_lat), ord=2)
    L_thigh_1 = (L_GT - M_L_knee) / np.linalg.norm((L_GT - M_L_knee), ord=2)
    L_thigh_y = np.cross(L_thigh_1,L_thigh_x) / np.linalg.norm(np.cross(L_thigh_1,L_thigh_x), ord=2)
    L_thigh_z = np.cross(L_thigh_x,L_thigh_y) / np.linalg.norm(np.cross(L_thigh_x,L_thigh_y), ord=2)
    

    LCS_L_thigh = np.matrix([L_thigh_x,L_thigh_y,L_thigh_z])
    Veri_L_thigh = np.linalg.det(LCS_L_thigh)
    
# R_thigh LCS
    R_thigh_x = (R_knee_lat - R_knee_med) / np.linalg.norm((R_knee_lat - R_knee_med), ord=2)
    R_thigh_1 = (R_GT - M_R_knee) / np.linalg.norm((R_GT - M_R_knee), ord=2)
    R_thigh_y = np.cross(R_thigh_1,R_thigh_x) / np.linalg.norm(np.cross(R_thigh_1,R_thigh_x), ord=2)
    R_thigh_z = np.cross(R_thigh_x,R_thigh_y) / np.linalg.norm(np.cross(R_thigh_x,R_thigh_y), ord=2)
    
    LCS_R_thigh = np.matrix([R_thigh_x,R_thigh_y,R_thigh_z])
    Veri_R_thigh = np.linalg.det(LCS_R_thigh)
    
# L_leg LCS
    L_leg_x = (L_ankle_med - L_ankle_lat) / np.linalg.norm((L_ankle_med - L_ankle_lat), ord=2)
    L_leg_1 = (M_L_knee - M_L_ankle) / np.linalg.norm((M_L_knee - M_L_ankle), ord=2)
    L_leg_y = np.cross(L_leg_1,L_leg_x) / np.linalg.norm(np.cross(L_leg_1,L_leg_x), ord=2)
    L_leg_z = np.cross(L_leg_x,L_leg_y) / np.linalg.norm(np.cross(L_leg_x,L_leg_y), ord=2)
    
    LCS_L_leg = np.matrix([L_leg_x,L_leg_y,L_leg_z])
    Veri_L_leg = np.linalg.det(LCS_L_leg)
    
# R_leg LCS
    R_leg_x = (R_ankle_lat - R_ankle_med) / np.linalg.norm((R_ankle_lat - R_ankle_med), ord=2)
    R_leg_1 = (M_R_knee - M_R_ankle) / np.linalg.norm((M_R_knee - M_R_ankle), ord=2)
    R_leg_y = np.cross(R_leg_1,R_leg_x) / np.linalg.norm(np.cross(R_leg_1,R_leg_x), ord=2)
    R_leg_z = np.cross(R_leg_x,R_leg_y) / np.linalg.norm(np.cross(R_leg_x,R_leg_y), ord=2)
    
    LCS_R_leg = np.matrix([R_leg_x,R_leg_y,R_leg_z])
    Veri_R_leg = np.linalg.det(LCS_R_leg)

# L_foot LCS
    L_foot_y = (L_MT3 - L_heel) /  np.linalg.norm((L_MT3 - L_heel), ord=2)
    L_foot_1 = (L_MT3 - M_L_ankle) / np.linalg.norm((L_MT3 - M_L_ankle), ord=2)
    L_foot_x = np.cross(L_foot_1,L_foot_y) / np.linalg.norm(np.cross(L_foot_1,L_foot_y), ord=2)
    L_foot_z = np.cross(L_foot_x,L_foot_y) / np.linalg.norm(np.cross(L_foot_x,L_foot_y), ord=2)
    
    LCS_L_foot = np.matrix([L_foot_x,L_foot_y,L_foot_z])
    Veri_L_foot = np.linalg.det(LCS_L_foot)
    
# R_foot LCS        
    R_foot_y = (R_MT3 - R_heel) /  np.linalg.norm((R_MT3 - R_heel), ord=2)
    R_foot_1 = (R_MT3 - M_R_ankle) / np.linalg.norm((R_MT3 - M_R_ankle), ord=2)
    R_foot_x = np.cross(R_foot_1,R_foot_y) / np.linalg.norm(np.cross(R_foot_1,R_foot_y), ord=2)
    R_foot_z = np.cross(R_foot_x,R_foot_y) / np.linalg.norm(np.cross(R_foot_x,R_foot_y), ord=2)
    
    LCS_R_foot = np.matrix([R_foot_x,R_foot_y,R_foot_z])
    Veri_R_foot = np.linalg.det(LCS_R_foot)

# trunk LCS
    trunk_y = (PX - T7) / np.linalg.norm((PX - T7), ord=2)
    trunk_1 = (M_C7IJ - M_T7PX) / np.linalg.norm((M_C7IJ - M_T7PX), ord=2)
    trunk_x = np.cross(trunk_y,trunk_1) / np.linalg.norm(np.cross(trunk_y,trunk_1), ord=2)
    trunk_z = np.cross(trunk_x,trunk_y) / np.linalg.norm(np.cross(trunk_x,trunk_y), ord=2)
    
    LCS_trunk = np.matrix([trunk_x,trunk_y,trunk_z])
    Veri_trunk = np.linalg.det(LCS_trunk)
     
# L_upper_arm LCS
    L_upper_z = (L_elbow_lat - L_elbow_med) / np.linalg.norm((L_elbow_lat - L_elbow_med), ord=2)
    L_upper_1 = (L_shoulder - M_L_elbow) / np.linalg.norm((L_shoulder - M_L_elbow), ord=2)
    L_upper_y = np.cross(L_upper_z,L_upper_1) / np.linalg.norm(np.cross(L_upper_z,L_upper_1), ord=2)
    L_upper_x = np.cross(L_upper_y,L_upper_z) / np.linalg.norm(np.cross(L_upper_y,L_upper_z), ord=2)
    
    LCS_L_upper = np.matrix([L_upper_x,L_upper_y,L_upper_z])
    Veri_L_upper = np.linalg.det(LCS_L_upper)
    
# R_upper_arm LCS
    R_upper_z = (R_elbow_lat - R_elbow_med) / np.linalg.norm((R_elbow_lat - R_elbow_med), ord=2)
    R_upper_1 = (R_shoulder - M_R_elbow) / np.linalg.norm((R_shoulder - M_R_elbow), ord=2)
    R_upper_y = np.cross(R_upper_1,R_upper_z) / np.linalg.norm(np.cross(R_upper_1,R_upper_z), ord=2)
    R_upper_x = np.cross(R_upper_y,R_upper_z) / np.linalg.norm(np.cross(R_upper_y,R_upper_z), ord=2)
    
    LCS_R_upper = np.matrix([R_upper_x,R_upper_y,R_upper_z])
    Veri_R_upper = np.linalg.det(LCS_R_upper)
    
# L_lower_arm LCS
    L_lower_z = (L_wrist_lat - L_wrist_med) / np.linalg.norm((L_wrist_lat - L_wrist_med), ord=2)
    L_lower_1 = (M_L_elbow - M_L_wrist) / np.linalg.norm((M_L_elbow - M_L_wrist), ord=2)
    L_lower_y = np.cross(L_lower_z,L_lower_1) / np.linalg.norm(np.cross(L_lower_z,L_lower_1), ord=2)
    L_lower_x = np.cross(L_lower_y,L_lower_z) / np.linalg.norm(np.cross(L_lower_y,L_lower_z), ord=2)
    
    LCS_L_lower = np.matrix([L_lower_x,L_lower_y,L_lower_z])
    Veri_L_lower = np.linalg.det(LCS_L_lower)

# R_lower_arm LCS
    R_lower_z = (R_wrist_lat - R_wrist_med) / np.linalg.norm((R_wrist_lat - R_wrist_med), ord=2)
    R_lower_1 = (M_R_elbow - M_R_wrist) / np.linalg.norm((M_R_elbow - M_R_wrist), ord=2)
    R_lower_y = np.cross(R_lower_1,R_lower_z) / np.linalg.norm(np.cross(R_lower_1,R_lower_z), ord=2)
    R_lower_x = np.cross(R_lower_y,R_lower_z) / np.linalg.norm(np.cross(R_lower_y,R_lower_z), ord=2)
    
    LCS_R_lower = np.matrix([R_lower_x,R_lower_y,R_lower_z])
    Veri_R_lower = np.linalg.det(LCS_R_lower)
    
# L_hand LCS
    #L_hand_1 = (L_wrist_lat - L_MC3) / np.linalg.norm((L_wrist_lat - L_MC3))
    #L_hand_2 = (L_wrist_med - L_MC3) / np.linalg.norm((L_wrist_med - L_MC3))
    #L_hand_y = np.cross(L_hand_1,L_hand_2) / np.linalg.norm(np.cross(L_hand_1,L_hand_2), ord=2)
    #L_hand_3 = (M_L_wrist - L_MC3) / np.linalg.norm((M_L_wrist - L_MC3))
    #L_hand_z = np.cross(L_hand_3,L_hand_y) / np.linalg.norm(np.cross(L_hand_3,L_hand_y), ord=2)    
    #L_hand_x = np.cross(L_hand_y,L_hand_z) / np.linalg.norm(np.cross(L_hand_y,L_hand_z), ord=2)    
    
    L_hand_x = (M_L_wrist - L_MC3) / np.linalg.norm((M_L_wrist - L_MC3), ord=2)
    L_hand_y = np.cross(L_lower_z,L_hand_x) / np.linalg.norm(np.cross(L_lower_z,L_hand_x), ord=2)
    L_hand_z = np.cross(L_hand_x,L_hand_y) / np.linalg.norm(np.cross(L_hand_x,L_hand_y), ord=2)
    LCS_L_hand = np.matrix([L_hand_x,L_hand_y,L_hand_z])
    Veri_L_hand = np.linalg.det(LCS_L_hand)
    
# R_hand LCS
    #R_hand_1 = (R_wrist_lat - R_MC3) / np.linalg.norm((R_wrist_lat - R_MC3))
    #R_hand_2 = (R_wrist_med - R_MC3) / np.linalg.norm((R_wrist_med - R_MC3))
    #R_hand_y = np.cross(R_hand_2,R_hand_1) / np.linalg.norm(np.cross(R_hand_2,R_hand_1), ord=2)
    #R_hand_3 = (R_MC3 - M_R_wrist) / np.linalg.norm((R_MC3 - M_R_wrist))
    #R_hand_z = np.cross(R_hand_3,R_hand_y) / np.linalg.norm(np.cross(R_hand_3,R_hand_y), ord=2)    
    #R_hand_x = np.cross(R_hand_y,R_hand_z) / np.linalg.norm(np.cross(R_hand_y,R_hand_z), ord=2)  
    
    R_hand_x = (R_MC3 - M_R_wrist) / np.linalg.norm((M_R_wrist - R_MC3), ord=2)
    R_hand_y = np.cross(R_lower_z,R_hand_x) / np.linalg.norm(np.cross(R_lower_z,R_hand_x), ord=2)
    R_hand_z = np.cross(R_hand_x,R_hand_y) / np.linalg.norm(np.cross(R_hand_x,R_hand_y), ord=2)

    LCS_R_hand = np.matrix([R_hand_x,R_hand_y,R_hand_z])
    Veri_R_hand = np.linalg.det(LCS_R_hand)
#%%
# calculate angles tan
    
# L_hip joint
    L_pel_RM_th = np.dot(LCS_pelvis,LCS_L_thigh.T) # plrvis to thigh
    L_hip_x_rad = math.atan2(-1 * L_pel_RM_th[2,1] , L_pel_RM_th[2,2])
    L_hip_z_rad = math.atan2(-1 * L_pel_RM_th[1,0] , L_pel_RM_th[0,0])
    c1 = math.cos(L_hip_x_rad)
    c2 = L_pel_RM_th[2,2]/c1
    L_hip_y_rad = math.atan2(L_pel_RM_th[2,0] ,c2)
    L_hip_x_deg = math.degrees(L_hip_x_rad)
    L_hip_y_deg = math.degrees(L_hip_y_rad)
    L_hip_z_deg = math.degrees(L_hip_z_rad) 
    
# R_hip joint
    R_pel_RM_th = np.dot(LCS_pelvis,LCS_R_thigh.T) # plrvis to thigh    
    R_hip_x_rad = math.atan2(-1 * R_pel_RM_th[2,1] , R_pel_RM_th[2,2])
    R_hip_z_rad = math.atan2(-1 * R_pel_RM_th[1,0] , R_pel_RM_th[0,0])
    c1 = math.cos(R_hip_x_rad)
    c2 = R_pel_RM_th[2,2]/c1
    R_hip_y_rad = math.atan2(R_pel_RM_th[2,0] ,c2)
    R_hip_x_deg = math.degrees(R_hip_x_rad)
    R_hip_y_deg = math.degrees(R_hip_y_rad)
    R_hip_z_deg = math.degrees(R_hip_z_rad) 
    
#L_knee joint
    L_th_RM_le = np.dot(LCS_L_thigh,LCS_L_leg.T) #thigh to leg
    L_knee_x_rad = math.atan2(-1 * L_th_RM_le[2,1] , L_th_RM_le[2,2])
    L_knee_z_rad = math.atan2(-1 * L_th_RM_le[1,0] , L_th_RM_le[0,0])
    c1 = math.cos(L_knee_x_rad)
    c2 = L_th_RM_le[2,2]/c1
    L_knee_y_rad = math.atan2(L_th_RM_le[2,0] ,c2)
    L_knee_x_deg = math.degrees(L_knee_x_rad)
    L_knee_y_deg = math.degrees(L_knee_y_rad)
    L_knee_z_deg = math.degrees(L_knee_z_rad) 
    
# R_knee joint
    R_th_RM_le = np.dot(LCS_R_thigh,LCS_R_leg.T) #thigh to leg    
    R_knee_x_rad = math.atan2(-1 * R_th_RM_le[2,1] , R_th_RM_le[2,2])
    R_knee_z_rad = math.atan2(-1 * R_th_RM_le[1,0] , R_th_RM_le[0,0])
    c1 = math.cos(R_knee_x_rad)
    c2 = R_th_RM_le[2,2]/c1
    R_knee_y_rad = math.atan2(R_th_RM_le[2,0] ,c2)
    R_knee_x_deg = math.degrees(R_knee_x_rad)
    R_knee_y_deg = math.degrees(R_knee_y_rad)
    R_knee_z_deg = math.degrees(R_knee_z_rad)  
    
# L_ankle joint
    L_le_RM_fo = np.dot(LCS_L_leg,LCS_L_foot.T) #leg to foot
    L_ankle_x_rad = math.atan2(-1 * L_le_RM_fo[2,1] , L_le_RM_fo[2,2])
    L_ankle_z_rad = math.atan2(-1 * L_le_RM_fo[1,0] , L_le_RM_fo[0,0])
    c1 = math.cos(L_ankle_x_rad)
    c2 = L_le_RM_fo[2,2]/c1
    L_ankle_y_rad = math.atan2(L_le_RM_fo[2,0] ,c2)
    L_ankle_x_deg = math.degrees(L_ankle_x_rad)
    L_ankle_y_deg = math.degrees(L_ankle_y_rad)
    L_ankle_z_deg = math.degrees(L_ankle_z_rad)  
    
# R_ankle joint
    R_le_RM_fo = np.dot(LCS_R_leg,LCS_R_foot.T) #leg to foot    
    R_ankle_x_rad = math.atan2(-1 * R_le_RM_fo[2,1] , R_le_RM_fo[2,2])
    R_ankle_z_rad = math.atan2(-1 * R_le_RM_fo[1,0] , R_le_RM_fo[0,0])
    c1 = math.cos(R_ankle_x_rad)
    c2 = R_le_RM_fo[2,2]/c1
    R_ankle_y_rad = math.atan2(R_le_RM_fo[2,0] ,c2)
    R_ankle_x_deg = math.degrees(R_ankle_x_rad)
    R_ankle_y_deg = math.degrees(R_ankle_y_rad)
    R_ankle_z_deg = math.degrees(R_ankle_z_rad)  
    
# trunk joint
    pel_RM_tr = np.dot(LCS_pelvis,LCS_trunk.T) #pelvis to trunk
    trunk_x_rad = math.atan2(-1 * pel_RM_tr[2,1] , pel_RM_tr[2,2])
    trunk_z_rad = math.atan2(-1 * pel_RM_tr[1,0] , pel_RM_tr[0,0])
    c1 = math.cos(trunk_x_rad)
    c2 = pel_RM_tr[2,2]/c1
    trunk_y_rad = math.atan2(pel_RM_tr[2,0] ,c2)
    trunk_x_deg = math.degrees(trunk_x_rad)
    trunk_y_deg = math.degrees(trunk_y_rad)
    trunk_z_deg = math.degrees(trunk_z_rad)
    
# L_shoulder
    L_tr_RM_up = np.dot(LCS_trunk,LCS_L_upper.T) #trunk to upper arm
    L_shoulder_x_rad = math.atan2(-1 * L_tr_RM_up[2,1] , L_tr_RM_up[2,2])
    #L_shoulder_y_rad = math.atan2(-1 * L_tr_RM_up[2,0] , L_tr_RM_up[2,1] * np.sin(L_shoulder_x_rad))
    L_shoulder_z_rad = math.atan2(-1 * L_tr_RM_up[1,0] , L_tr_RM_up[0,0])
    c1 = math.cos(L_shoulder_x_rad)
    c2 = L_tr_RM_up[2,2]/c1
    L_shoulder_y_rad = math.atan2(L_tr_RM_up[2,0] ,c2)
    L_shoulder_x_deg = math.degrees(L_shoulder_x_rad)
    L_shoulder_y_deg = math.degrees(L_shoulder_y_rad)
    L_shoulder_z_deg = math.degrees(L_shoulder_z_rad)
    
# R_shoulder joint
    R_tr_RM_up = np.dot(LCS_trunk,LCS_R_upper.T) #trunk to upper arm
    R_shoulder_x_rad = math.atan2(-1 * R_tr_RM_up[2,1] , R_tr_RM_up[2,2])
    #R_shoulder_y_rad = math.atan2(-1 * R_tr_RM_up[2,0] , R_tr_RM_up[2,1] * np.sin(R_shoulder_x_rad))
    R_shoulder_z_rad = math.atan2(-1 * R_tr_RM_up[1,0] , R_tr_RM_up[0,0])
    c1 = math.cos(R_shoulder_x_rad)
    c2 = R_tr_RM_up[2,2]/c1
    R_shoulder_y_rad = math.atan2(R_tr_RM_up[2,0] ,c2)
    R_shoulder_x_deg = math.degrees(R_shoulder_x_rad)
    R_shoulder_y_deg = math.degrees(R_shoulder_y_rad)
    R_shoulder_z_deg = math.degrees(R_shoulder_z_rad)    
    
# L_elbow joint
    L_up_RM_lw = np.dot(LCS_L_upper,LCS_L_lower.T) #upper to lower
    L_elbow_x_rad = math.atan2(-1 * L_up_RM_lw[2,1] , L_up_RM_lw[2,2])
    #L_elbow_y_rad = math.atan2(-1 * L_up_RM_lw[2,0] , L_up_RM_lw[2,1] * np.sin(L_elbow_x_rad))
    L_elbow_z_rad = math.atan2(-1 * L_up_RM_lw[1,0] , L_up_RM_lw[0,0])
    c1 = math.cos(L_elbow_x_rad)
    c2 = L_up_RM_lw[2,2]/c1
    L_elbow_y_rad = math.atan2(L_up_RM_lw[2,0] ,c2)
    L_elbow_x_deg = math.degrees(L_elbow_x_rad)
    L_elbow_y_deg = math.degrees(L_elbow_y_rad)
    L_elbow_z_deg = math.degrees(L_elbow_z_rad)
    
# R_elbow joint
    R_up_RM_lw = np.dot(LCS_R_upper,LCS_R_lower.T) #upper to lower    
    R_elbow_x_rad = math.atan2(-1 * R_up_RM_lw[2,1] , R_up_RM_lw[2,2])
    #R_elbow_y_rad = math.atan2(-1 * R_up_RM_lw[2,0] , R_up_RM_lw[2,1] * np.sin(R_elbow_x_rad))
    R_elbow_z_rad = math.atan2(-1 * R_up_RM_lw[1,0] , R_up_RM_lw[0,0])
    c1 = math.cos(R_elbow_x_rad)
    c2 = R_up_RM_lw[2,2]/c1
    R_elbow_y_rad = math.atan2(R_up_RM_lw[2,0] ,c2)
    R_elbow_x_deg = math.degrees(R_elbow_x_rad)
    R_elbow_y_deg = math.degrees(R_elbow_y_rad)
    R_elbow_z_deg = math.degrees(R_elbow_z_rad)
    
# L_wrist joint
    L_lw_RM_ha = np.dot(LCS_L_lower,LCS_L_hand.T) #lower arm to hand    
    L_wrist_x_rad = math.atan2(-1 * L_lw_RM_ha[2,1] , L_lw_RM_ha[2,2])
    #L_wrist_y_rad = math.atan2(-1 * L_lw_RM_ha[2,0] , L_lw_RM_ha[2,1] * np.sin(L_wrist_x_rad))
    L_wrist_z_rad = math.atan2(-1 * L_lw_RM_ha[1,0] , L_lw_RM_ha[0,0])
    c1 = math.cos(L_elbow_x_rad)
    c2 = L_lw_RM_ha[2,2]/c1
    L_wrist_y_rad = math.atan2(L_lw_RM_ha[2,0] ,c2)
    L_wrist_x_deg = math.degrees(L_wrist_x_rad)
    L_wrist_y_deg = math.degrees(L_wrist_y_rad)
    L_wrist_z_deg = math.degrees(L_wrist_z_rad)
    
# R_wrist joint
    R_lw_RM_ha = np.dot(LCS_R_lower,LCS_R_hand.T) #lower arm to hand
    R_wrist_x_rad = math.atan2(-1 * R_lw_RM_ha[2,1] , R_lw_RM_ha[2,2])
   #R_wrist_y_rad = math.atan2(-1 * R_lw_RM_ha[2,0] , R_lw_RM_ha[2,1] * np.sin(R_wrist_x_rad))
    R_wrist_z_rad = math.atan2(-1 * R_lw_RM_ha[1,0] , R_lw_RM_ha[0,0])
    c1 = math.cos(R_elbow_x_rad)
    c2 = R_lw_RM_ha[2,2]/c1
    R_wrist_y_rad = math.atan2(R_lw_RM_ha[2,0] ,c2)
    R_wrist_x_deg = math.degrees(R_wrist_x_rad)
    R_wrist_y_deg = math.degrees(R_wrist_y_rad)
    R_wrist_z_deg = math.degrees(R_wrist_z_rad)
#%%
    # make dataframe
    angles = [L_shoulder_x_deg, L_shoulder_y_deg, L_shoulder_z_deg,
                  R_shoulder_x_deg, R_shoulder_y_deg, R_shoulder_z_deg,
                  L_elbow_x_deg, L_elbow_y_deg, L_elbow_z_deg,
                  R_elbow_x_deg, R_elbow_y_deg, R_elbow_z_deg,
                  L_wrist_x_deg, L_wrist_y_deg, L_wrist_z_deg,
                  R_wrist_x_deg, R_wrist_y_deg, R_wrist_z_deg,
                  L_hip_x_deg, L_hip_y_deg, L_hip_z_deg,
                  R_hip_x_deg, R_hip_y_deg, R_hip_z_deg,
                  L_knee_x_deg, L_knee_y_deg, L_knee_z_deg,
                  R_knee_x_deg, R_knee_y_deg, R_knee_z_deg,
                  L_ankle_x_deg, L_ankle_y_deg, L_ankle_z_deg,
                  R_ankle_x_deg, R_ankle_y_deg, R_ankle_z_deg, 
                  trunk_x_deg, trunk_y_deg, trunk_z_deg]
    
    angles_df.loc[i] = angles # append to df
angles_df.to_csv('angles_self_' + name + str(No) + '.csv')    
