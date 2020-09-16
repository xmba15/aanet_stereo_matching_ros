#!/usr/bin/env python
import cv2
import copy
import numpy as np
import os
import struct
import yaml
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import PointField, CameraInfo, Image


_FIELDS_XYZ = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]
_FIELDS_XYZRGB = _FIELDS_XYZ + [PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1)]


class Utility:
    @staticmethod
    def yaml_to_camera_info(yaml_file_path):
        if not os.path.isfile(yaml_file_path):
            raise Exception("invalid yaml file path: {}".format(yaml_file_path))

        with open(yaml_file_path, "r") as file_handle:
            calib_data = yaml.safe_load(file_handle)
            camera_info = CameraInfo()
            camera_info.width = calib_data["image_width"]
            camera_info.height = calib_data["image_height"]
            camera_info.K = calib_data["camera_matrix"]["data"]
            camera_info.D = calib_data["distortion_coefficients"]["data"]
            camera_info.R = calib_data["rectification_matrix"]["data"]
            camera_info.P = calib_data["projection_matrix"]["data"]
            camera_info.distortion_model = calib_data["distortion_model"]

            return camera_info

    @staticmethod
    def scale_camera_info(camera_info, scale_x, scale_y):
        output_camera_info = copy.deepcopy(camera_info)

        output_camera_info.width = int(camera_info.width * scale_x)
        output_camera_info.height = int(camera_info.width * scale_y)

        K = list(camera_info.K)
        K[0] *= scale_x
        K[2] *= scale_x
        K[4] *= scale_y
        K[5] *= scale_y
        output_camera_info.K = tuple(K)

        P = list(camera_info.P)
        P[0] *= scale_x
        P[2] *= scale_x
        P[3] *= scale_x
        P[5] *= scale_y
        P[6] *= scale_y
        output_camera_info.P = tuple(P)
        return output_camera_info

    @staticmethod
    def to_cv_image(image_msg):
        if image_msg is None:
            return None

        width = image_msg.width
        height = image_msg.height
        channels = int(len(image_msg.data) / (width * height))

        encoding = None
        if image_msg.encoding.lower() in ["rgb8", "bgr8"]:
            encoding = np.uint8
        elif image_msg.encoding.lower() == "mono8":
            encoding = np.uint8
        elif image_msg.encoding.lower() == "32fc1":
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels), dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == "mono8":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        elif image_msg.encoding.lower() == "rgb8":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        return cv_img

    @staticmethod
    def to_img_msg(disp_img, encoding, header=None):
        img_msg = Image()
        if header is not None:
            img_msg.header = header

        height, width = disp_img.shape[:2]
        img_msg.height = height
        img_msg.width = width

        if disp_img.dtype.byteorder == ">":
            img_msg.is_bigendian = True

        img_msg.encoding = encoding
        img_msg.data = disp_img.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height

        return img_msg

    @staticmethod
    def xyzrgb_array_to_pointcloud2(projected_points, colors, max_depth, header=None):
        points_3d = []

        if len(colors.shape) == 2:
            colors = np.dstack([colors, colors, colors])

        height, width = colors.shape[:2]
        for u in range(height):
            img_row = colors[u]
            projected_row = projected_points[u]
            for v in range(width):
                cur_projected_point = projected_row[v]
                x = cur_projected_point[2]
                y = -cur_projected_point[0]
                z = -cur_projected_point[1]

                if x < 0 or x > max_depth:
                    continue

                cur_color = img_row[v]
                b = cur_color[0]
                g = cur_color[1]
                r = cur_color[2]

                a = 255
                rgb = struct.unpack("I", struct.pack("BBBB", b, g, r, a))[0]
                pt = [x, y, z, rgb]
                points_3d.append(pt)

        if header is None:
            pointcloud_msg = pc2.create_cloud(std_msgs.msg.Header(), _FIELDS_XYZRGB, points_3d)
        else:
            pointcloud_msg = pc2.create_cloud(header, _FIELDS_XYZRGB, points_3d)

        return pointcloud_msg

    @staticmethod
    def remap(raw_img, camera_info: CameraInfo):
        cam = PinholeCameraModel()
        cam.fromCameraInfo(camera_info)
        map_x, map_y = cv2.initUndistortRectifyMap(
            cam.intrinsicMatrix(),
            cam.distortionCoeffs(),
            cam.R,
            cam.projectionMatrix(),
            (cam.width, cam.height),
            cv2.CV_32FC1,
        )
        rectified_image = cv2.remap(raw_img, map_x, map_y, cv2.INTER_CUBIC)

        return rectified_image

    @staticmethod
    def get_q_matrix(left_camera_info: CameraInfo, right_camera_info: CameraInfo):
        # https://github.com/ros-perception/vision_opencv/blob/noetic/image_geometry/src/stereo_camera_model.cpp#L86

        left_fx = left_camera_info.P[0]
        left_fy = left_camera_info.P[5]
        left_cx = left_camera_info.P[2]
        left_cy = left_camera_info.P[6]
        right_cx = right_camera_info.P[2]
        base_line = -right_camera_info.P[3] / right_camera_info.P[0]

        Tx = -base_line
        Q = np.zeros((4, 4), dtype=np.float32)
        Q[0, 0] = left_fy * Tx
        Q[0, 3] = -left_fy * left_cx * Tx
        Q[1, 1] = left_fx * Tx
        Q[1, 3] = -left_fx * left_cy * Tx
        Q[2, 3] = left_fx * left_fy * Tx
        Q[3, 2] = -left_fy
        Q[3, 3] = left_fy * (left_cx - right_cx)

        return Q

    @staticmethod
    def uint16_to_uint8(img_uint16):
        return cv2.convertScaleAbs(img_uint16, alpha=(255.0 / 65535.0))
