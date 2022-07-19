#!/usr/bin/env python

from tokenize import Pointfloat
import rospy
import sys
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_msgs.msg import Int32
import moveit_msgs.msg
import numpy as np
import math
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose

from tf.transformations import *

import time
from collections import deque
import moveit_commander
from mtx_driver.msg import imu_msg

import copy
from math import pi
from math import tan

from moveit_commander.conversions import pose_to_list

def Upper_extremity_Fk(th_c):
    th1 = np.deg2rad(th_c[0])
    th2 = np.deg2rad(th_c[1])
    th3 = np.deg2rad(th_c[2])
    th4 = np.deg2rad(th_c[3])
    th5 = np.deg2rad(th_c[4])

    l1 = 0.3
    l2 = 0.27

    M = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -(l1+l2)],
                [0, 0, 0, 1]])

    inv_M = np.linalg.inv(M)
    
    w1 = np.array([0, 0, 1])
    q1 = np.array([0, 0, 0])
    v1 = np.cross(-w1, q1)
    S1_2 = np.array([[w1[0],w1[1],w1[2],v1[0],v1[1],v1[2]]])
    S1 = S1_2.T
    TS1 = MatrixExp6(np.dot(VecTose3(S1),th1))

    w2 = np.array([0, 1, 0])
    q2 = np.array([0, 0, 0])
    v2 = np.cross(-w2, q2)
    S2_2 = np.array([[w2[0],w2[1],w2[2],v2[0],v2[1],v2[2]]])
    S2 = S2_2.T
    TS2 = MatrixExp6(np.dot(VecTose3(S2),th2))

    w3 = np.array([1, 0, 0])
    q3 = np.array([0, 0, 0])
    v3 = np.cross(-w3, q3)
    S3_2 = np.array([[w3[0],w3[1],w3[2],v3[0],v3[1],v3[2]]])
    S3 = S3_2.T
    TS3 = MatrixExp6(np.dot(VecTose3(S3), th3))

    # w4 = np.array([0, 1, 0])
    # q4 = np.array([0, 0, -l1])
    # v4 = np.cross(-w4, q4)
    # S4_2 = np.array([[w4[0],w4[1],w4[2],v4[0],v4[1],v4[2]]])
    # S4 = S4_2.T
    # TS4 = MatrixExp6(np.dot(VecTose3(S4), th4))
    
    w4 = np.array([0, 1, 0])
    q4 = np.array([0, 0, -l1])
    v4 = np.cross(-w4, q4)
    S4_2 = np.array([[w4[0],w4[1],w4[2],v4[0],v4[1],v4[2]]])
    S4 = S4_2.T
    TS4 = MatrixExp6(np.dot(VecTose3(S4), th4))

    w5 = np.array([0, 0, -1])
    q5 = np.array([0, 0, -l1])    ##equal q5
    v5 = np.cross(-w5, q5)
    S5_2 = np.array([[w5[0],w5[1],w5[2],v5[0],v5[1],v5[2]]])
    S5 = S5_2.T
    TS5 = MatrixExp6(np.dot(VecTose3(S5), th5))

    Tsb = np.dot(np.dot(np.dot(np.dot(np.dot(TS1,TS2),TS3),TS4),TS5),M)
    return Tsb

def Upper_extremity_Ja(th_c):
    th1 = np.deg2rad(th_c[0])
    th2 = np.deg2rad(th_c[1])
    th3 = np.deg2rad(th_c[2])
    th4 = np.deg2rad(th_c[3])
    th5 = np.deg2rad(th_c[4])

    l1 = 0.3
    l2 = 0.27

    M = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -(l1+l2)],
                [0, 0, 0, 1]])

    inv_M = np.linalg.inv(M)

    w1 = np.array([0, 0, 1])
    q1 = np.array([0, 0, 0])
    v1 = np.cross(-w1, q1)
    S1_2 = np.array([[w1[0],w1[1],w1[2],v1[0],v1[1],v1[2]]])
    S1 = S1_2.T
    TS1 = np.dot(VecTose3(S1),th1)
    B1 = np.dot(np.dot(inv_M,VecTose3(S1)),M)
    TB1 = MatrixExp6(np.dot(B1,th1))

    w2 = np.array([0, 1, 0])
    q2 = np.array([0, 0, 0])
    v2 = np.cross(-w2, q2)
    S2_2 = np.array([[w2[0],w2[1],w2[2],v2[0],v2[1],v2[2]]])
    S2 = S2_2.T
    TS2 = np.dot(VecTose3(S2),th2)
    B2 = np.dot(np.dot(inv_M,VecTose3(S2)),M)
    TB2_m = MatrixExp6(np.dot(-B2, th2))
    TB2 = MatrixExp6(np.dot(B2,th2))

    w3 = np.array([1, 0, 0])
    q3 = np.array([0, 0, 0])
    v3 = np.cross(-w3, q3)
    S3_2 = np.array([[w3[0],w3[1],w3[2],v3[0],v3[1],v3[2]]])
    S3 = S3_2.T
    TS3 = np.dot(VecTose3(S3),th3)
    B3 = np.dot(np.dot(inv_M,VecTose3(S3)),M)
    TB3_m = MatrixExp6(np.dot(-B3, th3))
    TB3 = MatrixExp6(np.dot(B3,th3))

    w4 = np.array([0, 1, 0])
    q4 = np.array([0, 0, -l1])
    v4 = np.cross(-w4, q4)
    S4_2 = np.array([[w4[0],w4[1],w4[2],v4[0],v4[1],v4[2]]])
    S4 = S4_2.T
    TS4 = VecTose3(S4)
    B4 = np.dot(np.dot(inv_M,VecTose3(S4)),M)
    TB4_m = MatrixExp6(np.dot(-B4, th4))
    TB4 = MatrixExp6(np.dot(B4,th4))

    w5 = np.array([0, 0, -1])
    q5 = np.array([0, 0, -l1])
    v5 = np.cross(-w5, q5)
    S5_2 = np.array([[w5[0],w5[1],w5[2],v5[0],v5[1],v5[2]]])
    S5 = S5_2.T
    TS5 = VecTose3(S5)
    B5 = np.dot(np.dot(inv_M,VecTose3(S5)),M)
    TB5_m = MatrixExp6(np.dot(-B5, th5))
    TB5 = MatrixExp6(np.dot(B5,th5))

    Jb_1 = np.dot(Adjoint(np.dot(np.dot(np.dot(TB5_m,TB4_m),TB3_m),TB2_m)),se3ToVec(B1))
    Jb_2 = np.dot(Adjoint(np.dot(np.dot(TB5_m,TB4_m),TB3_m)),se3ToVec(B2))
    Jb_3 = np.dot(Adjoint(np.dot(TB5_m,TB4_m)),se3ToVec(B3))
    Jb_4 = np.dot(Adjoint(TB5_m),se3ToVec(B4))
    Jb_5 = se3ToVec(B5)

    # print(Jb_1.shape)
    # print(Jb_2.shape)
    # print(Jb_3.shape)
    # print(Jb_4.shape)
    # print(Jb_5.shape)

    Jb = np.hstack((Jb_1,Jb_2,Jb_3,Jb_4,Jb_5))
    # print(Jb.shape)

    Tsb = np.dot(np.dot(np.dot(np.dot(np.dot(M,TB1),TB2),TB3),TB4),TB5)

    JA = np.dot(Adjoint(Tsb),Jb)
    return JA

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[2], r[1], r[0])
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]

def se3ToVec(se3mat):
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]


def VecTose3(V):
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                np.zeros((1, 4))]


def VecToso3(omg):
    return np.array([[0, -omg[2], omg[1]],
                    [omg[2], 0, -omg[0]],
                    [-omg[1], omg[0], 0]])


def Adjoint(T):
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                np.c_[np.dot(VecToso3(p), R), R]]


def TransToRp(T):
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]


def AxisAng3(expc3):
    return (Normalize(expc3), np.linalg.norm(expc3))


def so3ToVec(so3mat):
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])


def NearZero(z):
    return abs(z) < 1e-6


def MatrixLog6(T):
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)),
                        [T[0][3], T[1][3], T[2][3]]],
                    [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat,
                        np.dot(np.eye(3) - omgmat / 2.0 \
                                + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                                * np.dot(omgmat, omgmat) / theta, [T[0][3],
                                                                    T[1][3],
                                                                    T[2][3]])],
                    [[0, 0, 0, 0]]]


def MatrixLog3(R):
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)


def MatrixExp6(se3mat):
    se3mat = np.array(se3mat)
    omgtheta = so3ToVec(se3mat[0: 3, 0: 3])
    if NearZero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        return np.r_[np.c_[MatrixExp3(se3mat[0: 3, 0: 3]),
                        np.dot(np.eye(3) * theta \
                                + (1 - np.cos(theta)) * omgmat \
                                + (theta - np.sin(theta)) \
                                * np.dot(omgmat, omgmat),
                                se3mat[0: 3, 3]) / theta],
                    [[0, 0, 0, 1]]]


def MatrixExp3(so3mat):
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
            + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)


def Normalize(V):
    return V / np.linalg.norm(V)


def rotz(th):
    T = [[np.cos(th), -np.sin(th), 0],
        [np.sin(th), np.cos(th), 0],
        [0, 0, 1]]
    return T


def rotx(th):
    T = [[1, 0, 0],
        [0, np.cos(th), -np.sin(th)],
        [0, np.sin(th), np.cos(th)]]
    return T


def roty(th):
    T = [[np.cos(th), 0, np.sin(th)],
        [0, 1, 0],
        [-np.sin(th), 0, np.cos(th)]]
    return T

f1 = open("/home/irol/catkin_ws/src/mtx_driver_stand/ang_velocity.txt","w")
f2 = open("/home/irol/catkin_ws/src/mtx_driver_stand/lin_velocity.txt","w")
f3 = open("/home/irol/catkin_ws/src/mtx_driver_stand/threshold.txt","w")
f4 = open("/home/irol/catkin_ws/src/mtx_driver_stand/EEF_position.txt","w")

initial_position = np.array([0,0,0])

def callback(msg):
    th1 = msg.s_yaw
    th2 = msg.s_pitch
    th3 = msg.s_roll
    pre_th4 = msg.w_pitch
    pre_th5 = msg.w_yaw
    th4 = th2 - pre_th4
    th5 = th1 -pre_th5

    o1 = msg.s_ang_z
    o2 = msg.s_ang_y
    o3 = msg.s_ang_x
    o4 = msg.w_ang_y
    o5 = -msg.w_ang_z

    theta = np.array([th1, th2, th3, th4, th5])
    omega = np.array([o1, o2, o3, o4, o5])

    UEFK = Upper_extremity_Fk(theta)
    UEJ = Upper_extremity_Ja(theta)

    sv = np.dot(UEJ,omega)

    ang_v = np.array([sv[0][0],sv[1][0],sv[2][0]])
    lin_v_s = np.array([sv[3][0],sv[4][0],sv[5][0]])
    q_2 = np.array([UEFK[0][3],UEFK[1][3],UEFK[2][3]])
    f4.write(str(q_2[0][0])+' ') #position_x
    f4.write(str(q_2[1][0])+' ') #position_y
    f4.write(str(q_2[2][0])+' '+'\n') #position_z
    q = q_2.T

    some_pre = np.cross(ang_v,q)
    some = some_pre[0]

    ang_v = ang_v
    f1.write(str(ang_v[0])+' ') #ang_velocity_x
    f1.write(str(ang_v[1])+' ') #ang_velocity_y
    f1.write(str(ang_v[2])+' '+'\n') #ang_velocity_z
    print('ang_v')
    print(ang_v)

    lin_v = lin_v_s+some
    print('lin_v')
    print(lin_v)
    f2.write(str(lin_v[0])+' ') #lin_velocity_x
    f2.write(str(lin_v[1])+' ') #lin_velocity_y
    f2.write(str(lin_v[2])+' '+'\n') #lin_velocity_z

    ###EEF spatial velocity end

    ###intensity estimator

    ##first_rotation
    tuning_r = 0.1
    Rk = tuning_r*(0.01*VecToso3(ang_v)) + np.eye(3,3) #0.01 is sampling time
    pre_vec_Rk = so3ToVec(Rk) #3x3 > 3x1, XYZ, Rk(theta*tuning_parameter)
    # print('pre_vec_rk')
    # print(pre_vec_Rk)

    # print(pre_vec_Rk)
    q_Rk = quaternion_from_euler(pre_vec_Rk[0],pre_vec_Rk[1],pre_vec_Rk[2] )
    # print('q_Rk')
    # print(q_Rk)
    # vec_Rk = np.array([pre_vec_Rk[2],pre_vec_Rk[1],pre_vec_Rk[0]]) #ZYX



    # c_orientation = np.array([0.707,0,0,0.707]) #robot_initial_pose
    q_c_orientation = quaternion_from_euler(0,0,0)
    # quaternion = euler_to_quaternion(vec_Rk) #ZYX > quaternion(x,y,z,w)
    target_orientation = quaternion_multiply(q_Rk, q_c_orientation) #x,y,z,w
    # print("target_o : x,y,z,w")
    # print(target_orientation)


    ##second_translation
    direction = lin_v/np.linalg.norm(lin_v)
    threshold = np.linalg.norm(lin_v)
    f3.write(str(threshold)+'\n')
    # print(threshold)
    # print(threshold.shape)
    tuning_l = 0.001 #mm
    intensity = tuning_l*threshold
    # print(lin_v)
    initial_position = np.array([0,0,0])
    target_position = initial_position + (direction * intensity)
    # print('threshold')
    # print(threshold)

    # rot_z = rotz(-90)
    # rot_z_q = euler_to_quaternion([-90,0,0])
    

    if threshold < 1.5:
        pub3.publish(Pose(Point(0,0,0),Quaternion(0,0,0,0)))
        
    elif threshold >= 1.5:
        # Point.data = np.dot(rot_z*target_position)
        # Quaternion.data = quaternion_multiply(rot_z_q,target_orientation)
        Point.data = target_position
        Quaternion.data = target_orientation
        pub3.publish(Pose(Point(target_position[0],target_position[1], target_position[2]),Quaternion(target_orientation[0], target_orientation[1], target_orientation[2], target_orientation[3])))

    # pub1.publish(target_position[0],target_position[1], target_position[2])
    # pub2.publish(target_orientation[0], target_orientation[1], target_orientation[2], target_orientation[3])
    # pub3.publish(Pose(Point(target_position[0],target_position[1], target_position[2]),Quaternion(target_orientation[0], target_orientation[1], target_orientation[2], target_orientation[3])))

    
    
rospy.init_node('target_pose')
sub = rospy.Subscriber('imu_msg', imu_msg, callback)
pub1 = rospy.Publisher('target_position', Point)
pub2 = rospy.Publisher('target_orientation', Quaternion)
pub3 = rospy.Publisher('target_pose_dh',Pose)
rospy.spin()
