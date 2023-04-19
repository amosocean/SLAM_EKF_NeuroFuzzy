#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :rootFunctions.py
# @Time      :2023/4/14 10:00 PM
# @Author    :Kinddle
import numpy as np

from tools import *


def prediction_step(mu, sigma, u, isFirst):
    """
    Compute the new mu based on the noise-free (odometry-based) motion model
    Remember to normalize theta after the update
    (hint: use the function normalize_angle available in tools)
    :param mu:(2N+3) array. representing the state mean
    :param sigma: (2N+3,2N+3) array covariance matrix
    :param u: odometry reading (r1, t, r2) ,Use them to access the rotation and translation values
    :param isFirst: -
    :return: new mu and new sigma
    """
    N = (len(mu) - 3) // 2
    Fx = np.c_[np.eye(3), np.zeros([3, N * 2])]
    new_mu = mu + Fx.T.dot(np.r_[u["t"] * np.cos(mu[2] + u["r1"]),
                                 u["t"] * np.sin(mu[2] + u["r1"]),
                                 normalize_angle(u["r1"] + u["r2"])])
    # Compute the 3*3 Jacobi Gx of the motion model
    Gx = np.array([[0, 0, -u["t"] * np.sin(mu[2] + u["r1"])],
                   [0, 0, u["t"] * np.sin(mu[2] + u["r1"])],
                   [0, 0, 0], ])

    G = np.eye(2 * N + 3) + Fx.T.dot(Gx).dot(Fx)
    # notion noise
    Sigma = 0.25
    R3 = np.diag([Sigma, Sigma, Sigma * 0.1])
    R = np.zeros(sigma.shape)
    R[:3, :3] = R3

    if not isFirst:
        new_sigma = G.dot(sigma).dot(G.T) + R
    else:
        new_sigma = sigma
    return new_mu, new_sigma


def correction_step(mu, sigma, z, observedLandmarks):
    """
    Updates the belief, i. e., mu and sigma after observing landmarks, according to the sensor model
    :param mu: array [2N+3]
    :param sigma: array [2N+3,2N+3]
    :param z: information of observe
    :param observedLandmarks: to make sure which landmarks have been observed
    :return:new mu sigma observedLandmarks
    """
    N = (len(mu) - 3) // 2
    m = len(z)
    Z = np.zeros([m*2])  # range bearing
    Z_expected = np.zeros([m*2])
    H = np.zeros([m*2, 2 * N + 3])

    for i in range(m):
        z_now = z[i]
        landmarkID = z_now["id"]
        if not observedLandmarks[landmarkID]:
            mu[3+landmarkID*2] = mu[0] + z_now["range"] * np.cos(z_now["bearing"] + mu[2])
            mu[3+landmarkID*2+1] = mu[1] + z_now["range"] * np.sin(z_now["bearing"] + mu[2])
            observedLandmarks[landmarkID] = True
        Z[i*2:i*2+2] = z_now['range'], z_now["bearing"]
        # Z[i,1] = z_now["bearing"]

        b = np.r_[mu[3+landmarkID*2] - mu[0], mu[3+landmarkID*2+1] - mu[1]]
        q = b.T.dot(b)
        Z_expected[i*2:i*2+2] = np.sqrt(q),normalize_angle(np.arctan2(b[1],b[0])-mu[2])

        Fx = np.r_[np.c_[np.eye(3),np.zeros([3,2*N])],
                   np.c_[np.zeros([2,2*landmarkID+3]),np.eye(2),np.zeros([2,2*N-2*landmarkID-2])]]
        Hi = 1/q * np.r_[np.c_[-np.sqrt(q)*b[0], -np.sqrt(q)*b[1], 0, +np.sqrt(q)*b[0], +np.sqrt(q)*b[1]],
                         np.c_[b[1],-b[0], -q ,-b[1],b[0]]] .dot(Fx)
        H[i*2:i*2+2,:] = Hi

    Q = np.eye(2*m) * 0.25
    K = sigma.dot(H.T).dot(np.linalg.inv(H.dot(sigma).dot(H.T)+Q))

    diff_Z = normalize_all_bearings(Z-Z_expected)
    new_mu = mu + K.dot(diff_Z)
    new_sigma = (np.eye(sigma.shape[0])-K.dot(H)).dot(sigma)
    return new_mu,new_sigma,observedLandmarks

    pass
