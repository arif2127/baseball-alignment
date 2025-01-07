import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import torch
from unet import UNet
import torch.nn.functional as F
from itertools import combinations


class Canonical_view_points():
    def __init__(self, num_point=110, R=1):

        self.num_point = num_point
        self.t = np.linspace(0, 2 * np.pi, self.num_point)  # parameter t for generating 3D points
        self.R = R
        self.b = 0.28 * self.R
        self.a = self.R - self.b
        self.seampoints = self.calc_3d_point()

        self.triangle_nest = 0

    def calc_3d_point(self):
        x = self.a * np.sin(self.t) + self.b * np.sin(3 * self.t)
        y = self.a * np.cos(self.t) - self.b * np.cos(3 * self.t)
        z = np.sqrt(4 * self.a * self.b) * np.cos(2 * self.t)

        R_z_90 = np.array([[0, -1, 0],
                           [1, 0, 0],
                           [0, 0, 1]])

        return np.dot(R_z_90, np.array([x, y, z]))
        # return np.array([x, y, z])

    def get_three_triangle(self, num_point_edge=10):

        middle_point1 = 0
        middle_point2 = int(self.num_point * 1 / 4)
        middle_point3 = int(self.num_point / 2)
        middle_point4 = int(self.num_point * 3 / 4)
        triangles = []

        triannge = []

        triannge.append(self.seampoints[:, middle_point1])
        triannge.append(self.seampoints[:, middle_point1 + num_point_edge])
        triannge.append(self.seampoints[:, self.num_point - (num_point_edge + 1)])
        triangles.append(np.asarray(triannge))

        triannge = []
        triannge.append(self.seampoints[:, middle_point2])
        triannge.append(self.seampoints[:, middle_point2 + num_point_edge])
        triannge.append(self.seampoints[:, middle_point2 - num_point_edge])

        triangles.append(np.asarray(triannge))

        triannge = []
        triannge.append(self.seampoints[:, middle_point3])
        triannge.append(self.seampoints[:, middle_point3 + num_point_edge])
        triannge.append(self.seampoints[:, middle_point3 - num_point_edge])

        triangles.append(np.asarray(triannge))

        triannge = []
        triannge.append(self.seampoints[:, middle_point4])
        triannge.append(self.seampoints[:, middle_point4 + num_point_edge])
        triannge.append(self.seampoints[:, middle_point4 - num_point_edge])

        triangles.append(np.asarray(triannge))

        return triangles

    # Function to calculate the edge lengths of a triangle
    def triangle_edge_lengths(self, triangle):
        p1, p2, p3 = triangle
        edge1 = np.linalg.norm(p1 - p2)
        edge2 = np.linalg.norm(p2 - p3)
        edge3 = np.linalg.norm(p3 - p1)
        return np.array([edge1, edge2, edge3])

    # Function to calculate the angles of a triangle using the law of cosines
    def triangle_angles(self, triangle):
        p1, p2, p3 = triangle
        # Edge vectors
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p3 - p2

        # Normalize vectors
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        v3 /= np.linalg.norm(v3)

        # Calculate angles using dot products
        angle1 = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angle2 = np.arccos(np.clip(np.dot(-v1, v3), -1.0, 1.0))
        angle3 = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))

        return np.array([angle1, angle2, angle3])

    # Function to compute the similarity distance between two triangles based on edge lengths and angles
    def triangle_similarity_distance(self, edges1, angles1, edges2, angles2):
        edge_distance = np.sum((edges1 - edges2) ** 2)
        angle_distance = np.sum((angles1 - angles2) ** 2)
        return edge_distance + angle_distance

    # Function to find the rotation matrix using the Kabsch algorithm
    def find_rotation_matrix(self, A, B):
        H = A.T @ B
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        return R

    def find_bestrotation_matrix(self, A, B):
        B2 = np.zeros((3, 3))

        B2[0, :] = B[2, :]
        B2[1, :] = B[1, :]
        B2[2, :] = B[0, :]

        # B2[:, 0] = B[:, 2]
        # B2[:, 1] = B[:, 1]
        # B2[:, 2] = B[:, 0]

        R = self.find_rotation_matrix(A, B)
        R2 = self.find_rotation_matrix(A, B2)

        # Step 1: Rotate points B using rotation matrix R
        rotated_B = B @ R.T  # Multiply each point in B by the transpose of R
        rotated_B2 = B2 @ R2.T  # Multiply each point in B by the transpose of R

        # Step 2: Calculate Euclidean distance between A and rotated B
        distances = np.linalg.norm(A - rotated_B, axis=1)
        distances2 = np.linalg.norm(A - rotated_B2, axis=1)

        # # Step 3: Compute the average distance
        # average_distance = np.mean(distances)
        # average_distance2 = np.mean(distances2)
        #
        # print(distances[1] , " ", distances2[1])

        return R

    def get_rotation(self, triangle_original, rotated_point_t):

        # Calculate edge lengths and angles for the original triangle
        original_edges = self.triangle_edge_lengths(triangle_original)
        original_angles = self.triangle_angles(triangle_original)

        # Initialize variables to store the best matching triangle and minimum distance
        min_distance = float('inf')
        best_triangle = None

        # Iterate through all possible triangles in the point cloud
        for triangle_points in combinations(rotated_point_t, 3):
            triangle_points = np.array(triangle_points)
            current_edges = self.triangle_edge_lengths(triangle_points)
            current_angles = self.triangle_angles(triangle_points)

            # Compute the similarity distance
            current_distance = self.triangle_similarity_distance(original_edges, original_angles, current_edges,
                                                                 current_angles)

            # Update if a better triangle is found
            if current_distance < min_distance:
                min_distance = current_distance
                best_triangle = triangle_points
        self.triangle_nest = best_triangle

        # Use the points directly to calculate the rotation matrix between the two triangles
        return best_triangle

        # return self.find_rotation_matrix(triangle_original, best_triangle)

    def rotation_matrix_to_euler_zxy(self, R):
        # Ensure the rotation matrix is a numpy array
        R = np.array(R)

        # Extract the Euler angles
        theta_x = np.arcsin(R[2, 1])
        theta_y = np.arctan2(-R[2, 0], R[2, 2])
        theta_z = np.arctan2(-R[0, 1], R[1, 1])

        # Convert from radians to degrees if necessary
        theta_x_deg = np.degrees(theta_x)
        theta_y_deg = np.degrees(theta_y)
        theta_z_deg = np.degrees(theta_z)

        return theta_z_deg, theta_x_deg, theta_y_deg

    def rotation_matrix_to_euler_zyx(self, R):
        # Ensure the rotation matrix is a numpy array
        R = np.array(R)

        # Check for gimbal lock (singularity condition)
        if np.abs(R[2, 0]) != 1:
            # Extract the Euler angles (ZYX order)
            theta_y = np.arcsin(-R[2, 0])
            theta_x = np.arctan2(R[2, 1], R[2, 2])
            theta_z = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Handle the singularity (gimbal lock) case where cos(theta_y) is zero
            theta_z = 0
            if R[2, 0] == -1:
                theta_y = np.pi / 2
                theta_x = theta_z + np.arctan2(R[0, 1], R[0, 2])
            else:
                theta_y = -np.pi / 2
                theta_x = -theta_z + np.arctan2(-R[0, 1], -R[0, 2])

        # Convert from radians to degrees if necessary
        theta_x_deg = np.degrees(theta_x)
        theta_y_deg = np.degrees(theta_y)
        theta_z_deg = np.degrees(theta_z)

        return theta_z_deg, theta_y_deg, theta_x_deg

    def get_optimal_angle(self, rotation_matrix):

        angles = []

        angles.append(self.rotation_matrix_to_euler_zxy(rotation_matrix))

        # 180-degree rotation matrix around Z-axis
        R_z_180 = np.array([[-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]])

        rotation_matrix2 = np.dot(R_z_180, rotation_matrix)

        angles.append(self.rotation_matrix_to_euler_zxy(rotation_matrix2))

        # 180-degree rotation matrix around Y-axis
        R_y_180 = np.array([[-1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]])

        # Rotate the original matrix by multiplying with the Z-axis rotation matrix
        rotation_matrix3 = np.dot(R_y_180, rotation_matrix)

        # 90-degree rotation matrix around Z-axis
        R_z_90 = np.array([[0, -1, 0],
                           [1, 0, 0],
                           [0, 0, 1]])

        # Rotate the original matrix by multiplying with the Z-axis rotation matrix
        rotation_matrix3 = np.dot(R_z_90, rotation_matrix3)

        angles.append(self.rotation_matrix_to_euler_zxy(rotation_matrix3))

        # Rotate the original matrix by multiplying with the Z-axis rotation matrix
        rotation_matrix4 = np.dot(R_z_180, rotation_matrix3)

        angles.append(self.rotation_matrix_to_euler_zxy(rotation_matrix4))

        angle_y = 360.00
        index = 0
        for i in range(len(angles)):
            # print("angle", angles[i])
            if abs(angles[i][2]) < angle_y:
                angle_y = angles[i][2]

                index = i

        return angles[index]


class MultiCamera():
    def __init__(self, path):
        self.path = path
        self.cams = self.read_pose_json()

    def read_pose_json(self):
        # Load JSON data
        with open(self.path, 'r') as f:
            cams = json.load(f)

        # Extract camera parameters and 3D points
        return cams


class Baseball3D():
    def __init__(self, cams):

        self.cams = cams

    def get_ball_3D(self, points_2D, ball_center_2d, ball_radius_2d):

        ball_center_3D = self.calculate_3d_position_and_radius(ball_center_2d)

        ball_radius_3d = self.compute_3d_ball_radius(ball_radius_2d, ball_center_3D)

        points_3D = self.find_3d_points_from_2d(points_2D, ball_center_3D.flatten(), ball_radius_3d).T
        points_3D = points_3D - ball_center_3D
        points_3D = points_3D.T

        return points_3D, ball_radius_3d

    def pixel_ray(self, K, point):
        # Convert pixel coordinates to normalized camera coordinates
        inv_K = np.linalg.inv(K)
        normalized_point = inv_K @ np.array([point[0], point[1], 1.0])
        return normalized_point / np.linalg.norm(normalized_point)

    def intersect_ray_sphere(self, ray_origin, ray_direction, sphere_center, sphere_radius):
        # Quadratic formula coefficients for ray-sphere intersection
        oc = ray_origin - sphere_center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - sphere_radius * sphere_radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None
        else:
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            t = min(t1, t2)
            return ray_origin + t * ray_direction

    def get_ray_origin_and_direction(self, cam, point):
        K = np.array(cam['K'])
        R = np.array(cam['R'])
        t = np.array(cam['t'])
        point = np.array(point)

        ray_origin = -R.T @ t  # Camera center in world coordinates
        ray_direction = R.T @ self.pixel_ray(K, point)
        return ray_origin, ray_direction

    def find_3d_points_from_2d(self, points_2D, ball_center_3d, ball_radius_3d):
        """
        Find the 3D coordinates of points that intersect with a known 3D sphere from multiple 2D camera points.

        points_2D: dictionary with keys 'cam0', 'cam1', 'cam2' and their respective 2D ball seam points.
        Cam: dictionary with camera intrinsic and extrinsic parameters.
        ball_center_3d: 3D center of the ball.
        ball_radius_3d: radius of the 3D ball.

        Returns:
            - Dictionary of 3D points for each camera.
        """
        # points_3D = {}
        intersections = []
        for cam_id, points in points_2D.items():
            # R = np.array(self.cams[cam_id]['R'])
            # t = np.array(self.cams[cam_id]['t']).reshape(3, 1)
            K = np.array(self.cams[cam_id]['K'])
            cam = self.cams[cam_id]
            dist = np.array(self.cams[cam_id]['dist'])

            # Undistort 2D points
            undistorted_points = self.undistort_points(points, K, dist).reshape(-1, 2)

            for point in undistorted_points:
                ray_origin, ray_direction = self.get_ray_origin_and_direction(cam, point)

                intersection = self.intersect_ray_sphere(ray_origin, ray_direction, ball_center_3d, ball_radius_3d)

                if intersection is not None:
                    intersections.append(intersection)

            # points_3D[cam_id] = np.array(intersections)

        return np.array(intersections)

    def undistort_points(self, points, K, dist):
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        K = np.array(K, dtype=np.float32)
        dist = np.array(dist, dtype=np.float32)
        undistorted_points = cv2.undistortPoints(points, K, dist, P=K)
        return undistorted_points.reshape(-1, 2)

    def triangulate_points(self, proj_matrices, points_2d):
        points_4d = cv2.triangulatePoints(proj_matrices[0], proj_matrices[1], points_2d[0], points_2d[1])
        points_4d /= points_4d[3]  # Convert to non-homogeneous coordinates
        return points_4d[:3]

    def calculate_3d_position_and_radius(self, points_2d):
        # Undistort 2D points
        undistorted_points = {}
        for cam in points_2d:
            K = self.cams[cam]['K']
            dist = self.cams[cam]['dist']
            undistorted_points[cam] = self.undistort_points(points_2d[cam], K, dist)

        # Prepare for triangulation
        proj_matrices = []
        points_2d_list = []
        for cam in self.cams:
            R = np.array(self.cams[cam]['R'])
            t = np.array(self.cams[cam]['t']).reshape(3, 1)
            Rt = np.hstack((R, t))
            K = np.array(self.cams[cam]['K'])
            proj_matrix = K.dot(Rt)
            proj_matrices.append(proj_matrix)

            points_2d_list.append(np.array(undistorted_points[cam]).reshape(2, -1))

        # Triangulate the 3D position of the baseball center using pairs of cameras
        center_3d_list = []
        for i in range(len(proj_matrices)):
            for j in range(i + 1, len(proj_matrices)):
                point_3d = self.triangulate_points([proj_matrices[i], proj_matrices[j]],
                                                   [points_2d_list[i], points_2d_list[j]])
                center_3d_list.append(point_3d)

        center_3d = np.mean(center_3d_list, axis=0)

        return center_3d

    def compute_3d_ball_radius(self, ball_radius_2d, ball_center_3d):
        radii = []

        for cam_name, radius_2d in ball_radius_2d.items():
            # Get the camera intrinsic and extrinsic parameters
            K = np.array(self.cams[cam_name]['K'])
            R = np.array(self.cams[cam_name]['R'])
            t = np.array(self.cams[cam_name]['t']).reshape(3, 1)

            # Compute the distance from the camera to the ball center
            cam_to_ball_3d = R @ ball_center_3d + t
            distance_to_ball = np.linalg.norm(cam_to_ball_3d)

            # Use the 2D radius and the camera parameters to estimate the 3D radius
            # The 2D radius is related to the 3D radius by the formula:
            # radius_2d = (focal_length * radius_3d) / distance_to_ball
            focal_length = (K[0, 0] + K[1, 1]) / 2  # Average focal length from fx and fy
            radius_3d = (radius_2d * distance_to_ball) / focal_length

            radii.append(radius_3d)

        # Return the average radius from all cameras
        return np.mean(radii)


class Baseball2D():
    def __init__(self, model_path='checkpoint_epoch2000.pth'):
        self.device = 'cpu'
        self.net = UNet(n_channels=3, n_classes=2, bilinear=False)
        self.net.to(device=self.device)
        state_dict = torch.load('model.pth', map_location=self.device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def get_ball_center_2D(self, im, min_radius, max_radius):

        # # Convert to grayscale
        image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Apply a blur to reduce noise (optional but recommended)
        image = cv2.medianBlur(image, 5)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(image,
                                   cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=50, param2=30,
                                   minRadius=min_radius, maxRadius=max_radius)

        if circles is not None:
            return [circles[0][0][0], circles[0][0][1]], circles[0][0][2]
        else:
            return [0, 0], 0

    def preprocess(self, mask_values, img, scale, is_mask):

        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        w, h = img.shape[0], img.shape[1]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST)
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(img)

        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img

    def predict_img(self, full_img, scale_factor=0.4, out_threshold=0.2):

        img = torch.from_numpy(self.preprocess(None, full_img, scale_factor, is_mask=False))
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img).cpu()
            # print(output.shape)
            # output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
            output = F.interpolate(output, (full_img.shape[0], full_img.shape[1]), mode='bilinear')
            if self.net.n_classes > 1:
                mask = output.argmax(dim=1)
            else:
                mask = torch.sigmoid(output) > out_threshold

        return mask[0].long().squeeze().numpy()

    def get_2D_point(self, img, center, radius, scale_factor=0.4):

        min = [center[0] - radius, center[1] - radius]
        max = [center[0] + radius, center[1] + radius]
        crop_img = img[int(min[1]):int(max[1]), int(min[0]):int(max[0]), :]
        mask = self.predict_img(crop_img, scale_factor=0.4)

        mask = cv2.convertScaleAbs(mask * 255)

        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob_centers = [cv2.moments(cnt) for cnt in contours if cv2.contourArea(cnt) > 1]

        # Get centroid points
        centers = [(int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])) for m in blob_centers if m['m00'] != 0]

        centers = np.asarray(centers)

        centers = centers + (int(min[0]), int(min[1]))

        # Calculate distances from the center to each 2D point
        distances = np.linalg.norm(centers - center, axis=1)

        return centers[distances < (radius - 25)]
