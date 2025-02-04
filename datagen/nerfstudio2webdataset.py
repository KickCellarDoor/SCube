# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import fvdb
import fvdb.nn as fvnn
import webdataset as wds
import json
import os
import numpy as np
import click
import sys
try:
    import open3d as o3d
except:
    import open3d_pycg as o3d
import traceback
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from loguru import logger
from tqdm import tqdm
from datagen.scube_data_utils import object_info_to_cuboid, object_info_to_object2world, get_points_in_cuboid, object_info_to_canonical_cuboid, \
    encode_dict_to_npz_bytes, imageencoder_imageio_png16, project_points_to_depth_image, align_depth_to_depth_batch, inference_metric3dv2, inference_mmseg

SAVE_RGB_WDS = True
SAVE_SEGMENTATION_WDS = True
SAVE_POSE_WDS = True
SAVE_DYNAMIC_OBJECT_BBOX_WDS = True
SAVE_ALL_OBJECT_BBOX_WDS = True 
SAVE_DEPTH_WDS = False

RECTIFY_DEPTH_AFFINE = True

def process_depth(filename_and_depth):
    filename, depth = filename_and_depth
    return filename, imageencoder_imageio_png16(depth)

def setup(local_rank, world_size):
    """
    bind the process to a GPU and initialize the process group
    But we do not need the communication among process, dist.init_process_group is unnecessary
    """
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    print(f"Process {local_rank} / {world_size} is using GPU {local_rank % torch.cuda.device_count()} in its node.")


def get_intr_extr(extrinsic_intrinsic_file, camera):
    """
    Returns:
        vehicle_to_camera_opencv: torch.Tensor, shape (4, 4), dtype=torch.float32
    """
    extrinsic_intrinsic = json.load(open(extrinsic_intrinsic_file, 'r'))

    camera_extrinsic = np.array(extrinsic_intrinsic['sensor_params'][camera]['extrinsic']) # OpenGL convention. camera to vehicle
    camera_to_vehicle_opencv = np.concatenate([camera_extrinsic[:,0:1],
                                                -camera_extrinsic[:,1:2],
                                                -camera_extrinsic[:,2:3],
                                                camera_extrinsic[:,3:4]], axis=-1)
    vehicle_to_camera_opencv = np.linalg.inv(camera_to_vehicle_opencv)

    camera_intrinsic = np.array(extrinsic_intrinsic['sensor_params'][camera]['camera_intrinsic'])

    width = extrinsic_intrinsic['sensor_params'][camera]['width']
    height = extrinsic_intrinsic['sensor_params'][camera]['height']

    return vehicle_to_camera_opencv, camera_intrinsic, height, width


def write_to_tar(sample, output_file):
    # prepare output file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # write tar file
    sink = wds.TarWriter(str(output_file))
    sink.write(sample)
    sink.close()
    print(f"Saved {output_file}")


def generate_shards(clip_id,
                    inferenecer,
                    ns_extraction_root,
                    output_root):
    
    ns_extraction_image_root_p = Path(ns_extraction_root)
    ns_extraction_lidar_root_p = Path(ns_extraction_root)
    output_root_p = Path(output_root)

    camera_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']

    logger.info(f'We use {camera_names} cameras')

    image_folder_p = ns_extraction_image_root_p / clip_id / 'images'

    print(f"Processing {clip_id}...")

    if SAVE_RGB_WDS:
        for camera_name in camera_names:
            sample = {}
            sample['__key__'] = f'{clip_id}'
            camera_folder_p = image_folder_p / camera_name
            camera_files = list(camera_folder_p.glob('*.jpg'))

            camera_files.sort()
            for idx, camera_file in enumerate(camera_files):
                with open(camera_file, 'rb') as stream:
                    sample[f'{idx:06d}.image.{camera_name.lower()}.jpg'] = stream.read()
            
            # write to tar file
            output_file = output_root_p / f"image_{camera_name.lower()}" / f"{clip_id}.tar"
            write_to_tar(sample, output_file)

    if SAVE_SEGMENTATION_WDS:
        for camera_name in camera_names:
            image_tar_file = output_root_p / f"image_{camera_name.lower()}" / f"{clip_id}.tar"
            if (output_root_p / f"skymask_{camera_name.lower()}" / f"{clip_id}.tar").exists():
                print(f"Skip {clip_id} for segmentation, already exists")
                continue

            dataset = wds.WebDataset(image_tar_file.as_posix(), nodesplitter=wds.non_empty).decode('npraw') 
            images = next(iter(dataset))
            video_numpy_list = [image for name, image in images.items() if name.endswith('.jpg')]
            segmentation_numpy_list = inference_mmseg(video_numpy_list, inferenecer)
            sample = {}
            sample['__key__'] = f'{clip_id}'
            for idx, segmentation_numpy in enumerate(segmentation_numpy_list):
                sky_mask = segmentation_numpy.astype(np.uint8) == 10
                sample[f'{idx:06d}.skymask.{camera_name.lower()}.png'] = sky_mask # do not use .tobytes(), there is no compression.

            # write to tar file
            output_file = output_root_p / f"skymask_{camera_name.lower()}" / f"{clip_id}.tar"
            write_to_tar(sample, output_file)

    if SAVE_POSE_WDS:
        extrinsic_intrinsic_file = ns_extraction_image_root_p / clip_id / 'transforms.json'
        extrinsic_intrinsic = json.load(open(extrinsic_intrinsic_file, 'r'))
        sample = {}
        sample['__key__'] = f'{clip_id}'

        # 1) save pose
        pose_dict = {}
        for camera_name in camera_names:
            pose_dict[camera_name] = {}
            for cam_info in extrinsic_intrinsic['frames']:
                if camera_name == cam_info['camera']:
                    pose = np.array(cam_info['transform_matrix'])
                    timestamp = str(cam_info['timestamp'])
                    pose_dict[camera_name][timestamp] = pose

            # sort the pose_dict by timestamp, stack the pose to a big ndarray
            timestamps = list(pose_dict[camera_name].keys())
            timestamps.sort()
            pose_list = [pose_dict[camera_name][timestamp] for timestamp in timestamps]

            poses_opengl = np.stack(pose_list, axis=0) # ns_extract follows OpenGL convention. We change it to OpenCV
            poses_opencv = \
                np.concatenate([poses_opengl[:,:,0:1], -poses_opengl[:,:,1:2], -poses_opengl[:,:,2:3], poses_opengl[:,:,3:4]], axis=-1)

            for idx, pose_opencv in enumerate(poses_opencv):
                sample[f'{idx:06d}.pose.{camera_name.lower()}.npy'] = pose_opencv

        # write to tar file
        output_file = output_root_p / f"pose" / f"{clip_id}.tar"
        write_to_tar(sample, output_file)

        # 2) save intrinsic along with pose, but in a separate tar file
        sample = {}
        sample['__key__'] = f'{clip_id}'
        for camera_name in camera_names:
            intrinsic_matrix = extrinsic_intrinsic['sensor_params'][camera_name]['camera_intrinsic']
            fx, fy, cx ,cy = intrinsic_matrix[0][0], intrinsic_matrix[1][1], intrinsic_matrix[0][2], intrinsic_matrix[1][2]
            w = extrinsic_intrinsic['sensor_params'][camera_name]['width']
            h = extrinsic_intrinsic['sensor_params'][camera_name]['height']
            intrisinc = np.array([fx, fy, cx, cy, w, h])
            sample[f'intrinsic.{camera_name.lower()}.npy'] = intrisinc
        
        # write to tar file
        output_file = output_root_p / f"intrinsic" / f"{clip_id}.tar"
        write_to_tar(sample, output_file)

    if SAVE_DYNAMIC_OBJECT_BBOX_WDS or SAVE_ALL_OBJECT_BBOX_WDS:
        annotation_file = ns_extraction_image_root_p / clip_id / 'annotation.json'
        objects = json.load(open(annotation_file, 'r'))['frames'] # "timestamp", "objects"

        extrinsic_intrinsic_file = ns_extraction_image_root_p / clip_id / 'transforms.json'
        extrinsic_intrinsic = json.load(open(extrinsic_intrinsic_file, 'r'))

        all_timestamps = set([str(cam_info['timestamp']) for cam_info in extrinsic_intrinsic['frames']])
        all_timestamps = list(all_timestamps)
        all_timestamps.sort()

        ### dynamic object ###
        if SAVE_DYNAMIC_OBJECT_BBOX_WDS:
            timestamp_to_objects = {}
            
            sample_corners = {}
            sample_corners['__key__'] = f'{clip_id}'

            sample_points_canonical = {}
            sample_points_canonical['__key__'] = f'{clip_id}'
            sample_points_canonical_data = {}

            sample_transformation = {}
            sample_transformation['__key__'] = f'{clip_id}'

            WAYMO_CATEGORY_NAMES = [
                "UNDEFINED", "CAR", "TRUCK", "BUS", "OTHER_VEHICLE", "MOTORCYCLIST", "BICYCLIST", "PEDESTRIAN",
                "SIGN", "TRAFFIC_LIGHT", "POLE", "CONSTRUCTION_CONE", "BICYCLE", "MOTORCYCLE", "BUILDING",
                "VEGETATION", "TREE_TRUNK", "CURB", "ROAD", "LANE_MARKER", "OTHER_GROUND", "WALKABLE", "SIDEWALK"
            ]

            lidar_anno_mapping = {
                'car': WAYMO_CATEGORY_NAMES.index('CAR'),
                'pedestrian': WAYMO_CATEGORY_NAMES.index('PEDESTRIAN'),
                'cyclist': WAYMO_CATEGORY_NAMES.index('BICYCLIST'),
                'sign': WAYMO_CATEGORY_NAMES.index('SIGN'),
                'unknown': WAYMO_CATEGORY_NAMES.index('UNDEFINED')
            }

            for per_frame_info in objects:
                timestamp = str(per_frame_info['timestamp'])
                timestamp_to_objects[timestamp] = per_frame_info['objects']

            for i, timestamp in enumerate(all_timestamps):
                sample_transformation_current_timestamp = {}

                objects_info = timestamp_to_objects[timestamp]
                dynamic_bboxes = np.zeros((0, 8, 3), dtype=np.float32)

                lidar_frame_info = [x for x in extrinsic_intrinsic['lidar_frames'] if str(x['timestamp']) == timestamp]
                assert len(lidar_frame_info) == 1
                lidar_frame_info = lidar_frame_info[0]
                lidar_to_world = np.array(lidar_frame_info['transform_matrix']) # FLU.

                lidar_points_path = ns_extraction_lidar_root_p / clip_id / 'lidars' / 'lidar_TOP' / f'{int(eval(timestamp) * 1e6)}.pcd'
                lidar_points = np.array(
                    o3d.io.read_point_cloud(lidar_points_path.as_posix()).points
                )
                for object_info in objects_info:
                    if object_info['is_moving']:
                        bbox = object_info_to_cuboid(object_info)
                        dynamic_bboxes = np.concatenate([dynamic_bboxes, bbox[np.newaxis]], axis=0)
                        sample_transformation_current_timestamp[object_info['gid']] = object_info_to_object2world(object_info) # current timestamp
                        points_in_bbox = get_points_in_cuboid(lidar_points, lidar_to_world, object_info)

                        obj_xyz_name = object_info['gid']+"_xyz"
                        obj_semantic_name = object_info['gid']+"_semantic_waymo"
                        obj_corner_name = object_info['gid']+"_corner"

                        if obj_xyz_name not in sample_points_canonical_data:
                            sample_points_canonical_data[obj_xyz_name] = points_in_bbox
                            sample_points_canonical_data[obj_semantic_name] = lidar_anno_mapping[object_info['type']]
                            sample_points_canonical_data[obj_corner_name] = object_info_to_canonical_cuboid(object_info)
                        else:
                            sample_points_canonical_data[obj_xyz_name] = np.concatenate([sample_points_canonical_data[obj_xyz_name], points_in_bbox], axis=0)

                sample_corners[f'{i:06d}.dynamic_object.npy'] = dynamic_bboxes
                sample_transformation[f'{i:06d}.dynamic_object_transformation.npz'] = encode_dict_to_npz_bytes(sample_transformation_current_timestamp)

            # write to tar file
            output_file = output_root_p / f"dynamic_object" / f"{clip_id}.tar"
            write_to_tar(sample_corners, output_file)

            output_file = output_root_p / f"dynamic_object_transformation" / f"{clip_id}.tar"
            write_to_tar(sample_transformation, output_file)

            sample_points_canonical['dynamic_object_points_canonical.npz'] = encode_dict_to_npz_bytes(sample_points_canonical_data)
            output_file = output_root_p / f"dynamic_object_points_canonical" / f"{clip_id}.tar"
            write_to_tar(sample_points_canonical, output_file)


        if SAVE_ALL_OBJECT_BBOX_WDS:
            timestamp_to_objects = {}
            sample = {}
            sample['__key__'] = f'{clip_id}'

            for per_frame_info in objects:
                timestamp = str(per_frame_info['timestamp'])
                timestamp_to_objects[timestamp] = per_frame_info['objects']
            
            for i, timestamp in enumerate(all_timestamps):
                objects_info = timestamp_to_objects[timestamp] # dict, with the following keys: translation, size, rotation
                current_frame_dict = {}
                for object_info in objects_info:
                    object_gid = object_info['gid']  # str
                    object_to_world = object_info_to_object2world(object_info) # [4, 4], np.float32
                    object_lwh = object_info['size']  # list
                    object_is_moving = object_info['is_moving'] # bool
                    current_frame_dict[object_gid] = {
                        'object_to_world': object_to_world.tolist(),
                        'object_lwh': object_lwh,
                        'object_is_moving': object_is_moving,
                        'object_type': object_info['type']
                    }
                sample[f'{i:06d}.all_object_info.json'] = json.dumps(current_frame_dict)
            
            # write to tar file
            output_file = output_root_p / f"all_object_info" / f"{clip_id}.tar"
            write_to_tar(sample, output_file)

    if SAVE_DEPTH_WDS:
        depth_model_name = 'metric3d'
        output_files = [output_root_p / f"rectified_{depth_model_name}_depth{'_affine' if RECTIFY_DEPTH_AFFINE else '' }_100_{camera_name.lower()}" / f"{clip_id}.tar"
                        for camera_name in camera_names]
        
        if np.all([output_file.exists() for output_file in output_files]):
            print(f"Skip {clip_id} for depth, already exists")
            return
        
        for camera in camera_names:
            if (output_root_p / f"rectified_{depth_model_name}_depth{'_affine' if RECTIFY_DEPTH_AFFINE else '' }_100_{camera.lower()}" / f"{clip_id}.tar").exists():
                print(f"Skip {clip_id} {camera} for depth, already exists")
                continue

            image_tar_file = output_root_p / f"image_{camera.lower()}" / f"{clip_id}.tar"
            dataset = wds.WebDataset(image_tar_file.as_posix(), nodesplitter=wds.non_empty).decode('npraw') 
            images = next(iter(dataset))
            video_numpy_list = [image for name, image in images.items() if name.endswith('.jpg')]
            video_tensor = torch.from_numpy(np.stack(video_numpy_list, axis=0)).permute(0, 3, 1, 2).float().cuda() / 255.0

            if depth_model_name == 'metric3d':
                infered_depth, _ = inference_metric3dv2(video_tensor)
                infered_depth = infered_depth.squeeze(1) # cuda tensor, [T,H,W]
            else:
                raise NotImplementedError('unknown depth model')

            # lidar projection
            single_lidar_scan_folder = ns_extraction_lidar_root_p / clip_id / 'lidars' / 'lidar_TOP'
            single_lidar_scan_files = list(single_lidar_scan_folder.glob('*.pcd'))
            single_lidar_scan_files.sort()

            # get the extrinsic and intrinsic matrix, project lidar for depth
            extrinsic_intrinsic_file = ns_extraction_image_root_p / clip_id / 'transforms.json'
            vehicle_to_camera_opencv, camera_intrinsic, height, width = get_intr_extr(extrinsic_intrinsic_file, camera)
            # to cuda
            vehicle_to_camera_opencv = torch.from_numpy(vehicle_to_camera_opencv).cuda().float()
            camera_intrinsic = torch.from_numpy(camera_intrinsic).cuda().float()

            # project lidar to depth
            lidar_depths = []
            for idx in range(len(video_tensor)):
                points = torch.from_numpy(np.array(o3d.io.read_point_cloud(single_lidar_scan_files[idx].as_posix()).points)).cuda().float()
                lidar_depth, lidar_depth_mask = project_points_to_depth_image(points, vehicle_to_camera_opencv, camera_intrinsic, width, height)
                lidar_depths.append(lidar_depth)
            lidar_depth = torch.stack(lidar_depths, dim=0)
            lidar_depth_mask = lidar_depth > 0
            
            sample_rectified_depth = {}
            sample_rectified_depth['__key__'] = f'{clip_id}'

            sample_lidar_projection = {}
            sample_lidar_projection['__key__'] = f'{clip_id}'

            sample_infered_depth = {}
            sample_infered_depth['__key__'] = f'{clip_id}'

            if RECTIFY_DEPTH_AFFINE:
                rectified_depth = align_depth_to_depth_batch(infered_depth, lidar_depth, lidar_depth_mask)
            else:
                raise NotImplementedError('unknown rectification method')

            for idx in range(len(video_tensor)):
                lidar_depth_np = lidar_depth[idx].cpu().numpy()
                rectified_depth_np = rectified_depth[idx].cpu().numpy()

                sample_rectified_depth[f"{idx:06d}.rectified_{depth_model_name}_depth{'_affine' if RECTIFY_DEPTH_AFFINE else '' }.{camera.lower()}.png"] = \
                    imageencoder_imageio_png16((rectified_depth_np * 100).astype(np.uint16))
                
                sample_lidar_projection[f'{idx:06d}.lidar_depth.{camera.lower()}.png'] = \
                    imageencoder_imageio_png16((lidar_depth_np * 100).astype(np.uint16))

            try:
                # write to tar file
                output_file = output_root_p / f"rectified_{depth_model_name}_depth{'_affine' if RECTIFY_DEPTH_AFFINE else '' }_100_{camera.lower()}" / f"{clip_id}.tar"
                write_to_tar(sample_rectified_depth, output_file)

                output_file = output_root_p / f"lidar_depth_100_{camera.lower()}" / f"{clip_id}.tar"
                write_to_tar(sample_lidar_projection, output_file)

                output_file = output_root_p / f"{depth_model_name}_depth_100_{camera.lower()}" / f"{clip_id}.tar"
                write_to_tar(sample_infered_depth, output_file)

            except Exception as e:
                # print error tracing
                print(f"Error in writing depth to tar file for {clip_id} {camera}, ", traceback.format_exc())
                from datetime import datetime
                now = datetime.now()
                # dd/mm/YY H:M:S
                current_time = now.strftime("%d/%m/%Y %H:%M:%S")
                # write down to debug.txt
                with open('debug.txt', 'a') as f:
                    f.write(f"{current_time} Error in writing depth to tar file for {clip_id} {camera}: {e}\n")


@click.command()
@click.option('--clip_list', type=str,
                default='datagen/waymo_all.json',
                help='The json file that contains the clip ids')
@click.option('--ns_extraction_root', '-i', type=str, 
               default='../waymo_ns',
               help='The root folder of the extracted image data')
@click.option('--output_root', '-o', type=str,
                default='../waymo_webdataset',
                help='The root folder of the output webdataset')
@click.option('--num_nodes', '-n', default=1, type=int, help='Number of nodes')
@click.option('--manual_split', '-m', default=-1, type=int, help='Manually split the clips for multi node running')
def main(clip_list, 
         ns_extraction_root, 
         output_root, 
         num_nodes, 
         manual_split):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    setup(local_rank, world_size)

    if SAVE_SEGMENTATION_WDS:
        from mmseg.apis import MMSegInferencer
        inferenecer = MMSegInferencer(model='segformer_mit-b5_8xb1-160k_cityscapes-1024x1024', device='cuda')
    else:
        inferenecer = None

    if clip_list.endswith('.json'):
        clip_list = json.load(open(clip_list, 'r'))
    else:
        clip_list = [clip_list]

    if manual_split != -1:
        print(f"Now we suppose {num_nodes} nodes in total, set -m to {list(range(num_nodes))}")
        clip_list = clip_list[manual_split::num_nodes]

    clip_list = clip_list[local_rank::world_size]

    for clip_id in tqdm(clip_list):
        generate_shards(clip_id, inferenecer, ns_extraction_root, output_root)

if __name__ == '__main__':
    main()
