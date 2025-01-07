from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGridLayout, QTextEdit, QSplitter
)
from PySide6.QtCore import Qt
import cv2
from PySide6.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt

from util import MultiCamera, Baseball3D, Baseball2D, Canonical_view_points
from io import BytesIO

import time
import open3d as o3d
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # index initialization
        self.ind = 0

        # Load camera parameters
        jeson_camera_path = 'output\\global_registration\\global_poses.json'
        cameras = MultiCamera(path=jeson_camera_path)
        self.cams = cameras.cams

        # Ball3D Setup with cams
        self.ball3d = Baseball3D(self.cams)

        # Ball2D Setup with cams
        self.Baseball2D = Baseball2D()

        self.setWindowTitle("Baseball Alignment System")

        # Fix the main window size to 800x600
        self.setFixedSize(1200, 920)

        # Main layout setup
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.is_connected = False

        # Splitter for left and right panels
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel Setup
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # Camera Connection Section
        # Camera Connection Section
        camera_connection_widget = QWidget()
        camera_layout = QVBoxLayout()
        self.connect_camera_button = QPushButton("Connect Camera")
        # Adjust the button size (width x height) and color
        self.connect_camera_button.setMinimumSize(150, 50)  # Example size: 150x50 pixels
        self.connect_camera_button.setStyleSheet("""
            QPushButton {
                background-color: lightblue;  /* Light blue color */
                border: 1px solid black;      /* Optional border */
                font-size: 18px;              /* Increase font size */
                font-weight: bold;            /* Make the text bold */
                padding: 5px;                 /* Padding inside the button */
            }
            QPushButton:hover {
                background-color: #ADD8E6;    /* Slightly darker on hover */
            }
        """)

        self.connect_camera_button.clicked.connect(self.connect_camera)

        # QLabel to display camera status
        self.camera_status_label = QLabel("Camera Status: Disconnected")
        camera_layout.addWidget(self.connect_camera_button)
        camera_layout.addWidget(self.camera_status_label)
        camera_connection_widget.setLayout(camera_layout)

        # Baseball Alignment Section
        baseball_alignment_widget = QWidget()
        baseball_layout = QVBoxLayout()
        self.run_alignment_button = QPushButton("Run Alignment")
        self.run_alignment_button.setMinimumSize(150, 50)  # Example size: 150x50 pixels
        self.run_alignment_button.setStyleSheet("""
                    QPushButton {
                        background-color: lightblue;  /* Light blue color */
                        border: 1px solid black;      /* Optional border */
                        font-size: 18px;              /* Increase font size */
                        font-weight: bold;            /* Make the text bold */
                        padding: 5px;                 /* Padding inside the button */
                    }
                    QPushButton:hover {
                        background-color: #ADD8E6;    /* Slightly darker on hover */
                    }
                """)

        self.run_alignment_button.clicked.connect(self.check_and_run_alignment)
        baseball_layout.addWidget(self.run_alignment_button)
        baseball_alignment_widget.setLayout(baseball_layout)

        # Add sections to the left panel
        left_layout.addWidget(camera_connection_widget)
        left_layout.addWidget(baseball_alignment_widget)

        # Right Panel Setup (2x2 Image Grid)
        # Right Panel Setup (2x2 Image Grid without using a loop)
        right_panel = QWidget()
        image_grid_layout = QGridLayout()

        # Create QLabel instances for each grid cell
        self.image_label_1 = QLabel("Image 1")
        self.image_label_1.setStyleSheet("border: 1px solid black;")
        self.image_label_2 = QLabel("Image 2")
        self.image_label_2.setStyleSheet("border: 1px solid black;")
        self.image_label_3 = QLabel("Image 3")
        self.image_label_3.setStyleSheet("border: 1px solid black;")

        # Centered text for image_label_4
        self.image_label_4 = QLabel("Centered Text Here")
        self.image_label_4.setStyleSheet("border: 1px solid black;font-size: 24px;font-weight: bold;")
        self.image_label_4.setAlignment(Qt.AlignCenter)  # Center the text

        # Add QLabel widgets to the grid layout manually
        image_grid_layout.addWidget(self.image_label_1, 0, 0)
        image_grid_layout.addWidget(self.image_label_2, 0, 1)
        image_grid_layout.addWidget(self.image_label_3, 1, 0)
        image_grid_layout.addWidget(self.image_label_4, 1, 1)

        right_panel.setLayout(image_grid_layout)

        # Log Panel
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setFixedHeight(120)

        # Add panels to the splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 3)

        # Add splitter and log panel to the main layout
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.log_panel)

    def update_image(self, opencv_image, index):
        """
        Updates the image placeholder with an actual OpenCV image.

        :param index: Index of the image placeholder (1 to 4).
        :param opencv_image: OpenCV image (numpy array) to display.
        """
        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width

        # Create a QImage from the numpy array
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(q_image)

        # Update the appropriate QLabel with the QPixmap
        if index == 1:
            self.image_label_1.setPixmap(pixmap.scaled(self.image_label_1.size(), Qt.KeepAspectRatio))
        elif index == 2:
            self.image_label_2.setPixmap(pixmap.scaled(self.image_label_2.size(), Qt.KeepAspectRatio))
        elif index == 3:
            self.image_label_3.setPixmap(pixmap.scaled(self.image_label_3.size(), Qt.KeepAspectRatio))
        elif index == 4:
            self.image_label_4.setPixmap(pixmap.scaled(self.image_label_4.size(), Qt.KeepAspectRatio))

    def connect_camera(self):
        # Stub function to handle camera connection

        ind = 4
        img1 = cv2.imread('cam1/' + str(ind) + '.png')
        img2 = cv2.imread('cam2/' + str(ind) + '.png')
        img3 = cv2.imread('cam3/' + str(ind) + '.png')
        self.update_image(img1, 1)
        self.update_image(img2, 2)
        self.update_image(img3, 3)

        self.is_connected = True  # Replace with actual connection check

        if self.is_connected:
            # Update the label to show a successful connection
            self.camera_status_label.setText("Camera Status: Connected")
            self.camera_status_label.setStyleSheet("color: green;")
            self.log_panel.append("Camera connected successfully.")
        else:
            # Update the label to show a failed connection
            self.camera_status_label.setText("Camera Status: Disconnected")
            self.camera_status_label.setStyleSheet("color: red;")
            self.log_panel.append("Failed to connect to the camera.")

    def get_ball_2d(self, img1, img2, img3):

        center2d1, radius2d1 = self.Baseball2D.get_ball_center_2D(img1, 285, 300)
        center2d2, radius2d2 = self.Baseball2D.get_ball_center_2D(img2, 285, 300)
        center2d3, radius2d3 = self.Baseball2D.get_ball_center_2D(img3, 285, 300)

        if radius2d1 > 290 or radius2d1 < 283:
            radius2d1 = 285.8
            center2d1 = [739.8, 396.6]

        if radius2d1 > 290 or radius2d1 < 283:
            radius2d2 = 287.1
            center2d2 = [678.600, 549.0]

        if radius2d1 > 291 or radius2d1 < 285:
            radius2d3 = 286.6
            center2d3 = [637.8, 523.8]

        ball_center_2d = {
            "cam0": center2d1,
            "cam1": center2d2,
            "cam2": center2d3
        }

        ball_radius_2d = {
            "cam0": radius2d1,
            "cam1": radius2d2,
            "cam2": radius2d3
        }

        points_2D = {
            "cam0": self.Baseball2D.get_2D_point(img1, ball_center_2d['cam0'], ball_radius_2d['cam0'],
                                                 scale_factor=0.25),
            "cam1": self.Baseball2D.get_2D_point(img2, ball_center_2d['cam1'], ball_radius_2d['cam1'],
                                                 scale_factor=0.25),
            "cam2": self.Baseball2D.get_2D_point(img3, ball_center_2d['cam2'], ball_radius_2d['cam2'],
                                                 scale_factor=0.25)
        }

        # print(points_2D["cam0"].shape)

        for i in range(points_2D["cam0"].shape[0]):
            cv2.circle(img1, points_2D["cam0"][i, :], 10, (255, 0, 0),
                       -1)  # Draw circle with center point, radius of 5, green color (0, 255, 0), and filled (-1)

        for i in range(points_2D["cam1"].shape[0]):
            cv2.circle(img2, points_2D["cam1"][i, :], 10, (255, 0, 0), -1)

        for i in range(points_2D["cam2"].shape[0]):
            cv2.circle(img3, points_2D["cam2"][i, :], 10, (255, 0, 0), -1)

        cv2.circle(img1, (int(center2d1[0]), int(center2d1[1])), int(radius2d1), (0, 255, 0), 2)
        self.update_image(img1, 1)
        cv2.circle(img2, (int(center2d2[0]), int(center2d2[1])), int(radius2d2), (0, 255, 0), 2)
        self.update_image(img2, 2)
        cv2.circle(img3, (int(center2d3[0]), int(center2d3[1])), int(radius2d3), (0, 255, 0), 2)
        self.update_image(img3, 3)

        return points_2D, ball_center_2d, ball_radius_2d

    def getrotationpcd(self, pcd1, pcd2):
        # Downsample point clouds (optional step to make processing faster)
        pcd1 = pcd1.voxel_down_sample(voxel_size=1)
        pcd2 = pcd2.voxel_down_sample(voxel_size=1)

        # Estimate normals for the source point cloud
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15.0, max_nn=30))
        pcd1.normalize_normals()

        # Estimate normals for the target point cloud
        pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15.0, max_nn=30))
        pcd2.normalize_normals()

        # Compute FPFH features for the source and target point clouds
        source_feature = o3d.pipelines.registration.compute_fpfh_feature(
            pcd1, o3d.geometry.KDTreeSearchParamHybrid(radius=15.0, max_nn=100)
        )
        target_feature = o3d.pipelines.registration.compute_fpfh_feature(
            pcd2, o3d.geometry.KDTreeSearchParamHybrid(radius=15.0, max_nn=100)
        )

        # Run RANSAC-based registration using feature matching
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd1,  # source point cloud
            pcd2,  # target point cloud
            source_feature,  # source feature
            target_feature,  # target feature
            mutual_filter=False,  # mutual filter set to False
            max_correspondence_distance=2,  # max correspondence distance
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # estimation method
            ransac_n=4,  # number of points for RANSAC
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(2.0)],  # checkers
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 1.0)  # RANSAC criteria
        )

        return result_ransac.transformation[:3, :3]

    def add_3d_plot_to_label(self, label, canonical_point, current_point, points_3D):
        # Create a 3D plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Rotated points
        ax.scatter(canonical_point[0], canonical_point[1], canonical_point[2], color='r', label="Canonical View Points")

        ax.scatter(current_point.T[0], current_point.T[1], current_point.T[2], color='b',
                   label="Rotated Current View Points")

        # ax.scatter(points_3D.T[0], points_3D.T[1], points_3D.T[2], color='g', label="Current View Points")

        # Render the plot to an image buffer
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Load the image into a QPixmap and set it to the QLabel
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), format='PNG')
        label.setPixmap(pixmap)

        # Close the plot to free memory
        plt.close(fig)

    # Rotate the point
    def rotate_point_cloud(self, point_cloud, rotation_matrix):
        # Apply rotation matrix to each point in the point cloud
        rotated_point_cloud = point_cloud @ rotation_matrix.T  # Transpose the matrix for correct multiplication
        return rotated_point_cloud

    # Example method to check the condition and run the alignment
    def check_and_run_alignment(self):
        if self.is_connected:
            self.run_alignment()
        else:
            # Optionally show a message or provide feedback
            self.log_panel.append("Warning: Please ensure the camera is connected before running alignment.")

    def run_alignment(self):
        # Stub function to run baseball alignment code
        start_time = time.time()

        # Load your point clouds (use arrays from your data)
        canO3d = o3d.geometry.PointCloud()
        ballO3d = o3d.geometry.PointCloud()

        # Load your point clouds (use arrays from your data)
        canO3d = o3d.geometry.PointCloud()
        ballO3d = o3d.geometry.PointCloud()

        # load the multi view images with index self.ind
        img1 = cv2.imread('cam1/' + str(self.ind) + '.png')
        img2 = cv2.imread('cam2/' + str(self.ind) + '.png')
        img3 = cv2.imread('cam3/' + str(self.ind) + '.png')
        self.ind = self.ind + 1

        # Collect all 2D seam points from 3 views
        points_2D, ball_center_2d, ball_radius_2d = self.get_ball_2d(img1, img2, img3)
        # Convert all 2D seam points into 3D point-cloud
        points_3D, ball_radius_3d = self.ball3d.get_ball_3D(points_2D, ball_center_2d, ball_radius_2d)


        # Run the alignment when the number of 3D seam point is greter than 15
        if points_3D.shape[0] > 15:

            ballO3d.points = o3d.utility.Vector3dVector(np.array(points_3D))

            # get all 3D information of the traget orientation of the ball
            canonical_3D = Canonical_view_points(num_point=108, R=ball_radius_3d)
            point = canonical_3D.seampoints
            canO3d.points = o3d.utility.Vector3dVector(np.array(canonical_3D.seampoints.T))

            rotation_matrix = self.getrotationpcd(ballO3d, canO3d)

            # angle = canonical_3D.rotation_matrix_to_euler_zxy(rotation_matrix)
            angle = canonical_3D.get_optimal_angle(rotation_matrix)
            output_signal = str(int(angle[0] * 100)) + "," + str(int(angle[1] * 100)) + "," + str(
                int(angle[2] * 100))  # The rotation will  be negative for all axis

            # conn.sendall(output_signal.encode())
            self.log_panel.append("Running baseball alignment...")
            self.image_label_4.setText(
                "Angle X : " + str(int(angle[1])) + ",  Angle Y : " + str(int(angle[2])) + ",  Ange Z : " + str(
                    int(angle[0])))
            self.add_3d_plot_to_label(self.image_label_4, point, self.rotate_point_cloud(points_3D, rotation_matrix),
                                      points_3D)

        else:

            self.image_label_4.setText("Number of seam point detected too low. Cannot get rotation")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
