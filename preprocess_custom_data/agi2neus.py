import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from glob import glob

def parse_args():
	parser = argparse.ArgumentParser(description="convert Agisoft XML to NeuS format poses.npy and cameras_sphere.npz")

	parser.add_argument("--file_in", default=r"E:/itest/", help="specify xml, ply file location")
	parser.add_argument("--out", default=r"E:/itest/", help="output path")
	parser.add_argument("--imgfolder", default=r"E:/data/barn-forNeuS/", help="location of folder with images")
	parser.add_argument("--imgtype", default="jpg", help="type of images (ex. jpg, png, ...)")
	args = parser.parse_args()
	return args


def get_calibration(root):
	for sensor in root[0][0]:
		for child in sensor:
			if child.tag == "calibration":
				return child
	print("No calibration found")	
	return None


if __name__ == "__main__":
	args = parse_args()
	IN_LOCATION = args.file_in
	IMGTYPE = args.imgtype
	IMGFOLDER = args.imgfolder
	OUTPATH = args.out

	with open(os.path.join(IN_LOCATION+'/'+'barn.xml'), "r") as f:
		root = ET.parse(f).getroot()

		w = float(root[0][0][0][0].get("width"))
		h = float(root[0][0][0][0].get("height"))
		calibration = get_calibration(root)
		f = float(calibration[1].text)

		components_matrix = np.ones([4, 4])
		components = root[0][1][0][0]
		rot_elements = [float(i) for i in components[0].text.split()]
		components_matrix[0][0] = rot_elements[0]
		components_matrix[0][1] = rot_elements[1]
		components_matrix[0][2] = rot_elements[2]
		components_matrix[1][0] = rot_elements[3]
		components_matrix[1][1] = rot_elements[4]
		components_matrix[1][2] = rot_elements[5]
		components_matrix[2][0] = rot_elements[6]
		components_matrix[2][1] = rot_elements[7]
		components_matrix[2][2] = rot_elements[8]
		shift_elements = [float(i) for i in components[1].text.split()]
		components_matrix[0][3] = shift_elements[0]
		components_matrix[1][3] = shift_elements[1]
		components_matrix[2][3] = shift_elements[2]

		components_matrix[3][0] = 0
		components_matrix[3][1] = 0
		components_matrix[3][2] = 0
  
		poses=[]
		c2w_mats=[]
  
		for frame in root[0][2]:
			current_frame = dict()
			if not len(frame):
				continue
			if(frame[0].tag != "transform"):
				continue

			matrix_elements = [float(i) for i in frame[0].text.split()]
   
			transform_matrix = np.array([[matrix_elements[0], matrix_elements[1], matrix_elements[2], matrix_elements[3]*float(components[2].text)], [matrix_elements[4], matrix_elements[5], matrix_elements[6], matrix_elements[7]*float(components[2].text)], [matrix_elements[8], matrix_elements[9], matrix_elements[10], matrix_elements[11]*float(components[2].text)], [matrix_elements[12], matrix_elements[13], matrix_elements[14], matrix_elements[15]]])
			transform_matrix =np.matmul(components_matrix,transform_matrix)

			c2w_mats.append(transform_matrix)
  
	poses=np.stack(c2w_mats,0)

	np.save(os.path.join(OUTPATH, 'poses.npy'), poses)
	cam_dict = dict()
	
	for i in range(poses.shape[0]):
		pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
		pose=poses[i]
		intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
		intrinsic[0, 2] = (w - 1) * 0.5
		intrinsic[1, 2] = (h - 1) * 0.5
		w2c = np.linalg.inv(pose)
		world_mat = intrinsic @ w2c
		world_mat = world_mat.astype(np.float32)
		cam_dict['camera_mat_{}'.format(i)] = intrinsic
		cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
		cam_dict['world_mat_{}'.format(i)] = world_mat
		cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)

 
	pcd = trimesh.load(os.path.join(IN_LOCATION, 'sparse_points_interest.ply'))
	vertices = pcd.vertices
	bbox_max = np.max(vertices, axis=0)
	bbox_min = np.min(vertices, axis=0)
	center = (bbox_max + bbox_min) * 0.5
	radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
	scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
	scale_mat[:3, 3] = center
 
	for i in range(poses.shape[0]):
		cam_dict['scale_mat_{}'.format(i)] = scale_mat
		cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)
  
	np.savez(os.path.join(OUTPATH, 'cameras_sphere.npz'), **cam_dict)
 
	print('Process done!')