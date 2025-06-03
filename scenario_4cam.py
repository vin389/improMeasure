import cv2
import numpy as np
from Camera import Camera

n_cams = 4
n_points = 8
data = np.zeros((n_points, 3+n_cams*6), dtype=np.float64)
# 8 points at (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)
pos_3d = np.array([[1, 1, 1],[1, 1, -1],[1, -1, 1],[1, -1, -1], [-1, 1, 1],[-1, 1, -1],[-1, -1, 1],[-1, -1, -1]], dtype=np.float64)
data[:, :3] = pos_3d
# cam1  
cam = Camera()
cam.setRvecTvecByPosAim([3, 0, 0], [0, 0, 0])
cam.setCmatByImgsizeFovs([1920, 1080], 90)
cam.dvec = np.zeros((4, 1), dtype=np.float64)
pos_2d_result = cv2.projectPoints(pos_3d, cam.rvec, cam.tvec, cam.cmat, cam.dvec)
pos_2d = pos_2d_result[0].reshape(-1, 2)
data[:, 3:5] = pos_2d
tmplt_w, tmplt_h = 20, 20
data[:, 5] = np.round(pos_2d[:,0] - tmplt_w / 2., 0) 
data[:, 6] = np.round(pos_2d[:,1] - tmplt_h / 2., 0) 
data[:, 7] = tmplt_w 
data[:, 8] = tmplt_h 
cam1 = cam
# cam2
cam = Camera()
cam.setRvecTvecByPosAim([0, 3, 0], [0, 0, 0])
cam.setCmatByImgsizeFovs([1920, 1080], 90)
cam.dvec = np.zeros((4, 1), dtype=np.float64)
pos_2d_result = cv2.projectPoints(pos_3d, cam.rvec, cam.tvec, cam.cmat, cam.dvec)
pos_2d = pos_2d_result[0].reshape(-1, 2)
data[:, 9:11] = pos_2d
tmplt_w, tmplt_h = 20, 20
data[:, 11] = np.round(pos_2d[:,0] - tmplt_w / 2., 0) 
data[:, 12] = np.round(pos_2d[:,1] - tmplt_h / 2., 0) 
data[:, 13] = tmplt_w 
data[:, 14] = tmplt_h 
cam2 = cam
# cam3
cam = Camera()
cam.setRvecTvecByPosAim([-3, 0, 0], [0, 0, 0])
cam.setCmatByImgsizeFovs([1920, 1080], 90)
cam.dvec = np.zeros((4, 1), dtype=np.float64)
pos_2d_result = cv2.projectPoints(pos_3d, cam.rvec, cam.tvec, cam.cmat, cam.dvec)
pos_2d = pos_2d_result[0].reshape(-1, 2)
data[:, 15:17] = pos_2d
tmplt_w, tmplt_h = 20, 20
data[:, 17] = np.round(pos_2d[:,0] - tmplt_w / 2., 0) 
data[:, 18] = np.round(pos_2d[:,1] - tmplt_h / 2., 0) 
data[:, 19] = tmplt_w 
data[:, 20] = tmplt_h 
cam3 = cam
# cam4
cam = Camera()
cam.setRvecTvecByPosAim([0, -3, 0], [0, 0, 0])
cam.setCmatByImgsizeFovs([1920, 1080], 90)
cam.dvec = np.zeros((4, 1), dtype=np.float64)
pos_2d_result = cv2.projectPoints(pos_3d, cam.rvec, cam.tvec, cam.cmat, cam.dvec)
pos_2d = pos_2d_result[0].reshape(-1, 2)
data[:, 21:23] = pos_2d
tmplt_w, tmplt_h = 20, 20
data[:, 23] = np.round(pos_2d[:,0] - tmplt_w / 2., 0) 
data[:, 24] = np.round(pos_2d[:,1] - tmplt_h / 2., 0) 
data[:, 25] = tmplt_w 
data[:, 26] = tmplt_h 
cam4 = cam
pass
# print
c=cam1; print(("%d\t%d\t"+"%f\t"*19+"\n") % (c.imgSize[0], c.imgSize[1], c.rvec.flatten()[0], c.rvec.flatten()[1], c.rvec.flatten()[2], c.tvec.flatten()[0], c.tvec.flatten()[1], c.tvec.flatten()[2], c.cmat[0,0], c.cmat[0,1], c.cmat[0,2], c.cmat[1,0], c.cmat[1,1], c.cmat[1,2], c.cmat[2,0], c.cmat[2,1], c.cmat[2,2], c.dvec.flatten()[0], c.dvec.flatten()[1], c.dvec.flatten()[2], c.dvec.flatten()[3]))
c=cam2; print(("%d\t%d\t"+"%f\t"*19+"\n") % (c.imgSize[0], c.imgSize[1], c.rvec.flatten()[0], c.rvec.flatten()[1], c.rvec.flatten()[2], c.tvec.flatten()[0], c.tvec.flatten()[1], c.tvec.flatten()[2], c.cmat[0,0], c.cmat[0,1], c.cmat[0,2], c.cmat[1,0], c.cmat[1,1], c.cmat[1,2], c.cmat[2,0], c.cmat[2,1], c.cmat[2,2], c.dvec.flatten()[0], c.dvec.flatten()[1], c.dvec.flatten()[2], c.dvec.flatten()[3]))
c=cam3; print(("%d\t%d\t"+"%f\t"*19+"\n") % (c.imgSize[0], c.imgSize[1], c.rvec.flatten()[0], c.rvec.flatten()[1], c.rvec.flatten()[2], c.tvec.flatten()[0], c.tvec.flatten()[1], c.tvec.flatten()[2], c.cmat[0,0], c.cmat[0,1], c.cmat[0,2], c.cmat[1,0], c.cmat[1,1], c.cmat[1,2], c.cmat[2,0], c.cmat[2,1], c.cmat[2,2], c.dvec.flatten()[0], c.dvec.flatten()[1], c.dvec.flatten()[2], c.dvec.flatten()[3]))
c=cam4; print(("%d\t%d\t"+"%f\t"*19+"\n") % (c.imgSize[0], c.imgSize[1], c.rvec.flatten()[0], c.rvec.flatten()[1], c.rvec.flatten()[2], c.tvec.flatten()[0], c.tvec.flatten()[1], c.tvec.flatten()[2], c.cmat[0,0], c.cmat[0,1], c.cmat[0,2], c.cmat[1,0], c.cmat[1,1], c.cmat[1,2], c.cmat[2,0], c.cmat[2,1], c.cmat[2,2], c.dvec.flatten()[0], c.dvec.flatten()[1], c.dvec.flatten()[2], c.dvec.flatten()[3]))

pass
