import time
import pytorch3d

import fvdb
import numpy as np
import point_cloud_utils as pcu
import torch
import torch_scatter
from fvdb import GridBatch, JaggedTensor
from icecream import ic
from pycg.isometry import Isometry
from scube.utils.color_util import semantic_from_points
from scube.utils.render_util import create_rays_from_intrinsic_torch_batch
from scube.utils.vis_util import WAYMO_CATEGORY_NAMES


def single_semantic_voxel_to_mesh(voxel_ijk, voxel_size = 0.1, voxel_origin = [0, 0, 0]):
    cube_v, cube_f = pcu.voxel_grid_geometry(voxel_ijk, gap_fraction=0.02, voxel_size=voxel_size, voxel_origin=voxel_origin)
    
    return cube_v, cube_f

def get_distance_from_voxel(camera_poses, intrinsics, grid, return_depth=False):
    """
    Args:
        camera_poses: torch.tensor
            camera poses, shape [N, 4, 4]
        intrinsics: torch.tensor
            camera intrinsics, shape [N, 6]
        grid: GridBatch
            grid to render, shape [1, ]

    Returns:
        distance: torch.Tensor, 
            distance from the camera origin to the voxel, shape [N, H, W, 1]
    """
    N = camera_poses.shape[0]

    # N, H, W, 3
    rays_o, rays_d = create_rays_from_intrinsic_torch_batch(camera_poses, intrinsics)
    H, W = rays_d.shape[1], rays_d.shape[2]
    rays_o = rays_o.reshape(N, 1, 1, 3).repeat(1, H, W, 1)

    assert isinstance(grid, GridBatch)
    segment = grid.segments_along_rays(rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), 1, eps=1e-3) 
    pixel_hit = segment.joffsets[1:] - segment.joffsets[:-1]
    pixel_hit = pixel_hit.view(N, H, W).float()
    distance = segment.jdata[:, 0] # [N_hit,]

    distance_map = torch.zeros((N, H, W)).to(distance.device)
    distance_map[pixel_hit > 0] = distance

    if return_depth:
        local_forward = torch.tensor([0, 0, 1]).to(camera_poses).reshape((1, 1, 1, 3, 1))

        # Multiply the rotation matrix by the local forward vector.
        # This transforms the vector into world space.
        world_forward = camera_poses[:, None, None, :3, :3] @ local_forward
        world_forward = world_forward.squeeze(-1)
        # Normalize the result to create a unit vector.
        norm = torch.linalg.norm(world_forward, dim=-1, keepdim=True)
        world_forward /= norm

        cos_d = torch.nn.functional.cosine_similarity(world_forward, rays_d, dim=-1)

        return distance_map.unsqueeze(-1), (distance_map * cos_d).unsqueeze(-1)
    else:
        return distance_map.unsqueeze(-1)


def get_mask_as_alpha_gt(camera_poses, intrinsics, grid):
    """
    Args:
        camera_poses: torch.tensor
            camera poses, shape [N, 4, 4]
        intrinsics: torch.tensor
            camera intrinsics, shape [N, 6]
        grid: GridBatch
            grid to render, shape [1, ]

    Returns:
        masks: torch.tensor
            masks for each camera pose, shape [N, H, W, 1]
    """
    N = camera_poses.shape[0]

    # N, H, W, 3
    rays_o, rays_d = create_rays_from_intrinsic_torch_batch(camera_poses, intrinsics)
    H, W = rays_d.shape[1], rays_d.shape[2]
    rays_o = rays_o.reshape(N, 1, 1, 3).repeat(1, H, W, 1)

    assert isinstance(grid, GridBatch)
    if fvdb.__version__ == '0.0.0':
        pack_info_segment, _, _ = grid.segments_along_rays(rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), 1, eps=1e-3)
        mask = pack_info_segment.jdata[:, 1] > 0
        mask = mask.view(N, H, W).bool()
    else:
        segment = grid.segments_along_rays(rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), 1, eps=1e-3) 
        mask = segment.joffsets[1:] - segment.joffsets[:-1]
        mask = mask.view(N, H, W).bool()

    return mask.unsqueeze(-1)


def generate_grid_mask_for_batch_data(batch, use_high_res_grid_for_alpha_mask):
    from scube.data.base import DatasetSpec as DS

    grid_to_create_mask = batch[DS.INPUT_PC] if not use_high_res_grid_for_alpha_mask \
                          else batch[DS.INPUT_PC_HIGHRES]
    
    if DS.IMAGES_INPUT_MASK in batch:
        B = len(batch[DS.IMAGES_INPUT_MASK])
        for i in range(B):
            grid_foreground_mask = get_mask_as_alpha_gt(batch[DS.IMAGES_INPUT_POSE][i], 
                                                    batch[DS.IMAGES_INPUT_INTRINSIC][i], 
                                                    grid_to_create_mask[i]) 
            batch[DS.IMAGES_INPUT_MASK][i][..., 3:4] = grid_foreground_mask  

    if DS.IMAGES_MASK in batch:
        B = len(batch[DS.IMAGES_MASK])
        for i in range(B):
            grid_foreground_mask = get_mask_as_alpha_gt(batch[DS.IMAGES_POSE][i], 
                                                    batch[DS.IMAGES_INTRINSIC][i], 
                                                    grid_to_create_mask[i])
            batch[DS.IMAGES_MASK][i][..., 3:4] = grid_foreground_mask


def get_rays_torch(pose_matrix, fx: float, fy: float, cx: float, cy: float, w: int, h: int):
    # torch version
    x, y = torch.meshgrid(torch.arange(w).to(pose_matrix.device), 
                          torch.arange(h).to(pose_matrix.device), 
                          indexing='xy')
    x = x.flatten()
    y = y.flatten()

    ray_dir = torch.stack([
        (x - cx) / fx,
        (y - cy) / fy,
        torch.ones_like(x)
    ]) # [3, H*W]
    ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=0)

    ray_dir = pose_matrix[:3, :3] @ ray_dir # [3, H*W]
    ray_dir = ray_dir.transpose(1, 0) # [H*W, 3]
    ray_orig = pose_matrix[:3, 3:4].repeat(1, int(h*w)).transpose(1, 0) # [H*W, 3]

    ray_dir = ray_dir.to(ray_orig) # fix for float16

    return ray_orig, ray_dir

def get_occ_front_voxel(grid: GridBatch, camera_pose, intrinsics, max_height=2400, max_voxels=3, return_per_cam_occ=False):
    """
    Get the front voxel of the grid, which is the voxel that first hit by the camera ray.
    Use higher resolution to do the ray casting.

    Args:
        grid: GridBatch
            grid to render, shape [1, ]
        camera_pose: torch.tensor
            camera poses, shape [B, N, 4, 4]
        intrinsics: torch.tensor
            camera intrinsics, shape [B, N, 6]

    Returns:
        occ_voxel_mask: JaggedTensor, 
            lshape [num_voxel1, num_voxel2, ...], eshape [1]
        occ_front_per_camera: JaggedTensor
            indicating this voxel is visible by which camera, 
            lshape [num_voxel1, num_voxel2, ...]
            eshape [N] (camera number)
    """
    out_voxel_masks = []
    occ_front_per_cameras = []
    B, N = camera_pose.shape[:2]

    for bidx in range(grid.grid_count):
        cur_grid = grid[bidx]
        cur_pose = camera_pose[bidx]
        cur_intrinsics = intrinsics[bidx]

        # make sure the resolution is high enough
        if cur_intrinsics[0, 5] < max_height:
            pseduo_cur_intrinsics = cur_intrinsics * (max_height / cur_intrinsics[0, 5])
        else:
            pseduo_cur_intrinsics = cur_intrinsics

        # [N, 3], [N, H, W, 3] -> [N * H * W, 3]
        nimg_origins, nimg_directions = create_rays_from_intrinsic_torch_batch(cur_pose, pseduo_cur_intrinsics)
        H, W = nimg_directions.shape[1], nimg_directions.shape[2]
        nimg_directions = nimg_directions.reshape(-1, 3)
        nimg_origins = nimg_origins.view(nimg_origins.size(0), 1, 1, 3).expand(-1, H, W, -1).reshape(-1, 3)
        
        H_time_W = H * W
        
        cur_grid_multi_view = fvdb.jcat([cur_grid] * N)
        out_voxel_ids, ray_start_end =  cur_grid_multi_view.voxels_along_rays(
                                            JaggedTensor(nimg_origins.split(H_time_W, dim=0)), 
                                            JaggedTensor(nimg_directions.split(H_time_W, dim=0)), 
                                            max_voxels=max_voxels, return_ijk=False
                                        )

        out_voxel_mask = torch.zeros((cur_grid.total_voxels, 1), device=cur_pose.device, dtype=torch.bool)
        uni_index = torch.unique(out_voxel_ids.jdata)
        out_voxel_mask[uni_index] = True
        out_voxel_masks.append(out_voxel_mask)

        if return_per_cam_occ:
            occ_front_per_camera = torch.zeros((cur_grid.total_voxels, N), device=cur_pose.device, dtype=torch.bool)
            for i in range(N):
                uni_index = torch.unique(out_voxel_ids[i].jdata)
                occ_front_per_camera[uni_index, i] = True
            occ_front_per_cameras.append(occ_front_per_camera)

    if return_per_cam_occ:
        return JaggedTensor(out_voxel_masks), JaggedTensor(occ_front_per_cameras)

    return JaggedTensor(out_voxel_masks)


def project_points(xyzs, proj_matrix):
    """
    Args:
        xyzs: jagged tensor, lshape [grid_num1, grid_num2], eshape [3]
        proj_matrix: [B, num_views, 3, 4], already normalized with h and w

    Returns:
        reference_points_cam: Jagged tensor, lshape [grid_num1, grid_num2], eshape [num_views, 1, 2]
        per_image_visibility_mask: Jagged tensor, lshape [grid_num1, grid_num2], eshape [num_views]
    """
    B = len(xyzs)
    V = proj_matrix.shape[1]
    reference_points_cam = []
    per_image_visibility_mask = []
    
    for b, xyz_jagged in enumerate(xyzs):
        xyz = xyz_jagged.jdata
        # Project 3D points to image space
        pts3d = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1) # grid_num_i, 4
        world_to_image = proj_matrix[b] # num_views, 3, 4
        pts2d = torch.einsum('vij,gj->gvi', world_to_image, pts3d) # grid_num_i, num_views, 3, fall in range [0,1]
        depth = pts2d[..., 2:]
        depth_valid_mask = depth > 0 # grid_num_i, num_views, 1
        uvs = pts2d[..., :2] / depth  # [0, 1]
        uv_valid_mask = (uvs >= 0) & (uvs <= 1) # grid_num_i, num_views, 2
        mask = depth_valid_mask[..., 0] & uv_valid_mask[..., 0] & uv_valid_mask[..., 1] # grid_num_i, num_views

        per_image_visibility_mask.append(mask)
        reference_points_cam.append(uvs.unsqueeze(2)) # grid_num_i, num_views, 1, 2

    reference_points_cam = JaggedTensor(reference_points_cam)
    per_image_visibility_mask = JaggedTensor(per_image_visibility_mask)


    return reference_points_cam, per_image_visibility_mask


def compatible_fvdb_cat(*args, **kwargs):
    if hasattr(fvdb, 'jcat'): # 0.0.1 version
        return fvdb.jcat(*args, **kwargs)
    else:
        return fvdb.cat(*args, **kwargs)


def compatible_jaggedtensor_elements(jagged_tensor):
    if len(jagged_tensor.joffsets.shape) == 1: # 0.0.1 version
        # also return len(jagged_tensor.lshape)
        return jagged_tensor.joffsets.size(0) - 1 
    else:
        return jagged_tensor.joffsets.size(0)


def crop_scene_level_grid(grid: fvdb.GridBatch, cam2grid: torch.Tensor, 
                          semantic: torch.Tensor = None,
                          grid_crop_bbox_min_in_meter=[-10.24, -51.2, -12.8], 
                          grid_crop_bbox_max_in_meter=[92.16, 51.2, 38.4]):
    """
    Args:
        grid: fvdb.GridBatch
            grid to crop, it is scene level point cloud so need to be cropped
        cam2grid: torch.Tensor
            input front camera to grid transformation, shape [B, 4, 4].
            Note that it is OpenCV convention (RDF), change to FLU
        semantic: list of torch.Tensor
            semantic for each voxel
    """
    raise NotImplementedError("This function is not implemented yet")


def keep_surface_voxels(batch):
    """
    only keep a thin shell of the grid, eliminate the inside voxels
    """
    from scube.data.base import DatasetSpec as DS

    grids = batch[DS.INPUT_PC]
    new_grids = []
    surface_masks = []
    for i, grid in enumerate(grids):
        neighbors = grid.neighbor_indexes(grid.ijk, 1)
        inner_mask = torch.all(neighbors.jdata[:, 1:, 1:, 1:].reshape(-1, 8) != -1, dim=-1) # shape (N, )
        surface_mask = ~inner_mask
        surface_ijk = grid.ijk.rmask(surface_mask)
        surface_masks.append(surface_mask)

        new_grid = fvdb.gridbatch_from_ijk(surface_ijk, voxel_sizes=grid.voxel_sizes, origins=grid.origins)
        new_grids.append(new_grid)

    batch[DS.INPUT_PC] = fvdb.jcat(new_grids)

    if DS.GT_SEMANTIC in batch:
        batch[DS.GT_SEMANTIC] = batch[DS.GT_SEMANTIC].rmask(torch.cat(surface_masks))

def prepare_semantic_jagged_tensor(batch):
    from scube.data.base import DatasetSpec as DS

    if DS.GT_SEMANTIC in batch:
        batch[DS.GT_SEMANTIC] = JaggedTensor(batch[DS.GT_SEMANTIC]).to(batch[DS.INPUT_PC].device)


def clip_batch_grid(batch, ijk_min, ijk_max):
    from scube.data.base import DatasetSpec as DS

    if DS.GT_SEMANTIC in batch:
        clipped_semantic, clipped_grid = \
            batch[DS.INPUT_PC].clip(batch[DS.GT_SEMANTIC], ijk_min, ijk_max)
        batch[DS.INPUT_PC] = clipped_grid
        batch[DS.GT_SEMANTIC] = clipped_semantic
    else:
        clipped_grid = batch[DS.INPUT_PC].clipped_grid(ijk_min, ijk_max)
        batch[DS.INPUT_PC] = clipped_grid

    if DS.INPUT_PC_HIGHRES in batch:
        resolution_ratio = batch[DS.INPUT_PC].voxel_sizes[0][0] / batch[DS.INPUT_PC_HIGHRES].voxel_sizes[0][0]
        ijk_min_highres = [int(i * resolution_ratio) for i in ijk_min]
        ijk_max_highres = [int(i * resolution_ratio) for i in ijk_max]
        batch[DS.INPUT_PC_HIGHRES] = batch[DS.INPUT_PC_HIGHRES].clipped_grid(ijk_min_highres, ijk_max_highres)


def coarsen_batch_grid(batch, coarsen_factor):
    from scube.data.base import DatasetSpec as DS

    if DS.GT_SEMANTIC in batch:
        vdb_tensor = fvdb.nn.VDBTensor(batch[DS.INPUT_PC], batch[DS.GT_SEMANTIC].float())
        vdb_tensor_downsample = fvdb.nn.MaxPool(coarsen_factor)(vdb_tensor)
        batch[DS.INPUT_PC] = vdb_tensor_downsample.grid
        batch[DS.GT_SEMANTIC] = vdb_tensor_downsample.data.int()
    else:
        batch[DS.INPUT_PC] = batch[DS.INPUT_PC].coarsened_grid(coarsen_factor)


def get_semantic_visual_data(grid, semantics):
    """
    Args:
        res_feature_sets
    """
    from scube.utils.vis_util import get_waymo_palette
    waymo_palette, waymo_mapping = get_waymo_palette()

    vox_ijk = grid.ijk.jdata.cpu().numpy()
    visualization_color_category = waymo_mapping[semantics.tolist()].tolist()
    visualizaiton_color = waymo_palette[visualization_color_category].astype(np.float32)

    
    return vox_ijk, visualization_color_category, visualizaiton_color

def offscreen_mesh_render_for_vae_decoded_list(vae_decoded_list, backend='filament', extend_direction='y',
                                               default_camera_kwargs={"pitch_angle": 80.0, "fill_percent": 0.8, "fov": 80.0, 'plane_angle': 90}):
    """
    Args:
        vae_decoded_list: list of tuple
            [(res_feature_set, out_vdb_tensor), ...], res_feature_set is the FeatureSet defined in VAE

    Returns:
        np.array, shape [H, W, 3]
    """
    assert len(vae_decoded_list) > 0, "latents_list should have at least 1 elements"

    vox_ijk_collection = []
    visualization_color_category_collection = []
    visualization_color_collection = []

    max_x_interval = 0
    max_y_interval = 0

    for i, vae_decoded in enumerate(vae_decoded_list):
        res_feature_set, out_vdb_tensor = vae_decoded
        grid = out_vdb_tensor.grid
        semantic_prob = res_feature_set.semantic_features[-1].jdata # [n_voxel, 23]
        semantics = semantic_prob.argmax(dim=-1) # [n_voxel, ]
        semantics = np.array(semantics.cpu().numpy()).astype(np.uint8)

        vox_ijk, visualization_color_category, visualization_color = get_semantic_visual_data(grid, semantics) 

        max_x_interval = max(max_x_interval, int((np.max(vox_ijk[:,0]) - np.min(vox_ijk[:,0])) * 1.2))
        max_y_interval = max(max_y_interval, int((np.max(vox_ijk[:,1]) - np.min(vox_ijk[:,1])) * 1.2))
        
        if i > 0:
            if extend_direction == 'x':
                vox_ijk[:, 0] += i * max_x_interval
            elif extend_direction == 'y':
                vox_ijk[:, 1] += i * max_y_interval

        vox_ijk_collection.append(vox_ijk)
        visualization_color_category_collection.append(visualization_color_category)
        visualization_color_collection.append(visualization_color)

    vox_ijk = np.concatenate(vox_ijk_collection, axis=0)
    vox_ijk = (vox_ijk - np.mean(vox_ijk, axis=0)).astype(np.int32)
    visualization_color_category = np.concatenate(visualization_color_category_collection, axis=0)
    visualization_color = np.concatenate(visualization_color_collection, axis=0)

    return offscreen_mesh_render(vox_ijk, visualization_color_category, visualization_color, backend=backend, default_camera_kwargs=default_camera_kwargs)

def offsreen_mesh_renderer_for_vae(grid_semantic_pairs, backend='filament', extend_direction='y',
                                   default_camera_kwargs={"pitch_angle": 80.0, "fill_percent": 0.8, "fov": 80.0, 'plane_angle': 90}):
    """
    Args:
        grid_semantic_pairs: list of tuple
            [(grid, semantics), ...], grid is fvdb.GridBatch, semantics is torch.Tensor shape [n_voxel, ]

    Returns:
        np.array, shape [H, W, 3]
    """
    vox_ijk_collection = []
    visualization_color_category_collection = []
    visualization_color_collection = []

    max_x_interval = 0
    max_y_interval = 0


    for i, (grid, semantics) in enumerate(grid_semantic_pairs):
        vox_ijk, visualization_color_category, visualization_color = get_semantic_visual_data(grid, semantics)
        
        max_x_interval = max(max_x_interval, int((np.max(vox_ijk[:,0]) - np.min(vox_ijk[:,0])) * 1.2))
        max_y_interval = max(max_y_interval, int((np.max(vox_ijk[:,1]) - np.min(vox_ijk[:,1])) * 1.2))


        if i > 0:
            if extend_direction == 'x':
                vox_ijk[:, 0] += i * max_x_interval
            elif extend_direction == 'y':
                vox_ijk[:, 1] += i * max_y_interval

        vox_ijk_collection.append(vox_ijk)
        visualization_color_category_collection.append(visualization_color_category)
        visualization_color_collection.append(visualization_color)

    vox_ijk = np.concatenate(vox_ijk_collection, axis=0)
    vox_ijk = (vox_ijk - np.mean(vox_ijk, axis=0)).astype(np.int32)
    visualization_color_category = np.concatenate(visualization_color_category_collection, axis=0)
    visualization_color = np.concatenate(visualization_color_collection, axis=0)

    return offscreen_mesh_render(vox_ijk, visualization_color_category, visualization_color, backend=backend, default_camera_kwargs=default_camera_kwargs)


def offscreen_mesh_render(vox_ijk, visualization_color_category, visualization_color, backend='pyrender', 
                          default_camera_kwargs={"pitch_angle": 80.0, "fill_percent": 0.8, "fov": 80.0, 'plane_angle': 90}):
    """
    backend: pyrender or filament
    """
    from scube.utils.voxel_util import single_semantic_voxel_to_mesh
    from pycg import image, render, vis

    cube_v_list = []
    cube_f_list = []
    cube_color_list = []
    geometry_list = []

    visualization_types = np.unique(visualization_color_category)
    for visualization_type in visualization_types:
        mask = visualization_color_category == visualization_type
        cube_v_i, cube_f_i = single_semantic_voxel_to_mesh(vox_ijk[mask])
        color_i = visualization_color[mask][0] 

        cube_v_list.append(cube_v_i)
        cube_f_list.append(cube_f_i)
        cube_color_list.append(color_i)
        
        geometry = vis.mesh(cube_v_i, cube_f_i, np.array(color_i).reshape(1,3).repeat(cube_v_i.shape[0], axis=0))
        geometry_list.append(geometry)

    scene: render.Scene = vis.show_3d(geometry_list, show=False, up_axis='+Z', default_camera_kwargs=default_camera_kwargs)

    if backend == 'pyrender':
        img = scene.render_pyrender()
    elif backend == 'filament':
        img = scene.render_filament()
    else:
        raise NotImplementedError
    img_np = np.array(img)

    del scene
    del geometry_list

    return img_np


def offscreen_map_voxel_render(maps_3d, grid_crop_bbox_min=None, grid_crop_bbox_max=None):
    """
    Args:
        maps_3d: dict
            keys are the map type name, values are the 3D maps points that is embraced in a list. 
            points already in grid coordinate!

        grid_crop_bbox_min & grid_crop_bbox_max: list
            if not provided, do not crop the map, or the map is already cropped
            note that they are saying to ego, align with dataset's setting

    Returns:
        np.array, shape [H, W, 3]
    """
    # render map 
    from matplotlib.colors import LinearSegmentedColormap
    from pycg import render, vis

    colors = ["orange", "cyan", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    pc_list = []
    rendered_images = []

    if grid_crop_bbox_min is not None and grid_crop_bbox_max is not None:
        # this is saying to ego, not grid coordinate
        grid_crop_bbox_min = torch.tensor(grid_crop_bbox_min)
        grid_crop_bbox_max = torch.tensor(grid_crop_bbox_max)

        # convert to grid coordinate
        grid_crop_bbox_max = (grid_crop_bbox_max - grid_crop_bbox_min) / 2
        grid_crop_bbox_min = - grid_crop_bbox_max
        
    # crop the points 
    for map_type, map_points in maps_3d.items():
        if isinstance(map_points, list) and len(map_points) == 1:
            map_points = map_points[0] # extract the first sample
        map_points = map_points.to(torch.int32) # convert to int

        # discard points outside the grid
        if grid_crop_bbox_min is not None and grid_crop_bbox_max is not None:
            map_points = map_points[(map_points >= grid_crop_bbox_min).all(dim=1) & (map_points < grid_crop_bbox_max).all(dim=1)]
            
        # only keep unique points
        map_points = torch.unique(map_points, dim=0)

        maps_3d[map_type] = map_points.float()

    # prepare pc_list
    for idx, map_type in enumerate(maps_3d):
        if maps_3d[map_type].shape[0] == 0:
            continue
        map_color = np.array(cmap(idx / len(maps_3d)))[:3].reshape(1, 3).repeat(maps_3d[map_type].shape[0], axis=0)
        map_pc = vis.pointcloud(pc=maps_3d[map_type].to('cpu').numpy(), color=map_color)
        pc_list.append(map_pc)

    # render the map
    for plane_angle in [90, 180, 270, 0]:
        scene: render.Scene = vis.show_3d(pc_list, show=False, up_axis='+Z', default_camera_kwargs={"pitch_angle": 45.0, "fill_percent": 0.7, "fov": 40.0, 'plane_angle': plane_angle})
        img = scene.render_filament()
        rendered_images.append(img)

    rendered_images = np.concatenate(rendered_images, axis=1)

    return rendered_images


def cc_removal_cpu(fvdb_grid, min_connected_voxels, grid_batch_kwargs, semantics=None):
    import scipy.sparse as sp

    # Cross indices on 3x3x3 voxels.
    cc_inds = [4, 10, 12, 13, 14, 16, 22]

    # torch implementation
    # [N_ijk, 7], if neighbor voxel exists, store their unique index, else -1
    nn_inds = fvdb_grid.neighbor_indexes(fvdb_grid.ijk, 1).jdata.view(-1, 27)[:, cc_inds]
    nn_mask = nn_inds > -1 # valid mask
    col_ind = nn_inds[nn_mask] 
    row_ptr = torch.cumsum(torch.sum(nn_mask, dim=1), 0)
    row_ptr = torch.cat([torch.zeros(1).long().to(row_ptr.device), row_ptr])

    torch.cuda.synchronize()
    current_time = time.time()

    # if considered dense matrix, it would be [N_ijk, N_ijk] 
    sp_mat = sp.csr_matrix((np.ones(col_ind.size(0)), 
                            col_ind.cpu().numpy().astype(int), 
                            row_ptr.cpu().numpy().astype(int)))
    _, component = sp.csgraph.connected_components(sp_mat, directed=False) # belong to which component

    component = torch.from_numpy(component).to(fvdb_grid.device)
    _, count = torch.unique(component, return_counts=True) # and how many voxels in each component
    cc_mask = (count > min_connected_voxels)[component]

    torch.cuda.synchronize()
    print(f"Time for computing connected component: {time.time() - current_time}")
    current_time = time.time()

    fvdb_grid.set_from_ijk(fvdb_grid.ijk.jdata[cc_mask], **grid_batch_kwargs)
    if semantics is not None:
        semantics = semantics[cc_mask]
        return fvdb_grid, semantics

    return fvdb_grid


def create_fvdb_grid_w_semantic_from_points(cropped_xyz, semantics, grid_batch_kwargs_target, grid_batch_kwargs_finest, 
                                            extra_meshes=None, remove_cc=False):
    """
    Important to note that if we want to get [0.4m, 0.4m, 0.4m] voxel size (target), while our finest 
    voxe_size goal is [0.1m, 0.1m, 0.1m], the best choice is voxelize at [0.1m, 0.1m, 0.1m] and
    coarsen the grid to [0.4m, 0.4m, 0.4m]!
    """
    if extra_meshes is not None:
        # {'vertices': [tensor1, tensor2], 'faces': [tensor1, tensor2]}
        B = len(extra_meshes['vertices'])
        car_label = WAYMO_CATEGORY_NAMES.index('CAR')

        extra_mesh_xyzs = []
        extra_mesh_semantics = []

        for i in range(B):
            if len(extra_meshes['vertices'][i]) == 0:
                extra_mesh_xyz = torch.zeros((0, 3), device=cropped_xyz[i].device, dtype=torch.int32)
                extra_mesh_semantic = torch.zeros((0,), device=semantics[i].device, dtype=torch.int32)
            else:
                extra_mesh_grid = fvdb.gridbatch_from_mesh(extra_meshes['vertices'][i], extra_meshes['faces'][i], **grid_batch_kwargs_finest)
                extra_mesh_xyz = extra_mesh_grid.grid_to_world(extra_mesh_grid.ijk.float()).jdata
                extra_mesh_semantic = torch.ones(extra_mesh_xyz.shape[0], device=semantics[i].device, dtype=torch.int32) * car_label

            cropped_xyz[i] = torch.cat([cropped_xyz[i], extra_mesh_xyz], dim=0)
            semantics[i] = torch.cat([semantics[i], extra_mesh_semantic], dim=0)

    fvdb_grids = fvdb.gridbatch_from_points(JaggedTensor(cropped_xyz), **grid_batch_kwargs_finest)

    if remove_cc:
        voxel_sizes = grid_batch_kwargs_finest['voxel_sizes']
        min_connected_voxels = min(int(1 / voxel_sizes[0] / voxel_sizes[0]), 35)
        fvdb_grids_cc_removal = []
        for grid in fvdb_grids:
            fvdb_grids_cc_removal.append(cc_removal_cpu(grid, min_connected_voxels, grid_batch_kwargs_finest))
        
        fvdb_grids = fvdb.jcat(fvdb_grids_cc_removal)

    if semantics is None:
        return fvdb_grids, None

    fvdb_grids_finest = []
    semantics_finest = []

    for fvdb_grid, point_xyz, point_semantic in zip(fvdb_grids, cropped_xyz, semantics):
        assert point_xyz.shape[0] == point_semantic.shape[0], "The number of points and semantics should be the same"
        # now some points fall out of the grid, we need to filter them out. The remaining are valid voxels
        # world_to_grid returns a float tensor coordinate respecting to origins and voxel_sizes, when given any input point cloud coordinate (can be outside the grid)
        pts_vox_idx = fvdb_grid.ijk_to_index(fvdb_grid.world_to_grid(point_xyz).jdata.round().long()).jdata 
        pts_valid_mask = pts_vox_idx >= 0 # -1 means out of voxels

        # filter out invalid points (but usually there is no invalid points, since we have already filtered them out in the mask above)
        valid_pts_vox_idx = pts_vox_idx[pts_valid_mask]
        valid_semantics = point_semantic[pts_valid_mask]

        unique_categories = torch.unique(valid_semantics)
        category_counts = []
        for category in unique_categories:
            category_count = torch_scatter.scatter_sum(
                (valid_semantics == category).float(),
                valid_pts_vox_idx,
                dim=0, dim_size=fvdb_grid.total_voxels
            )
            category_counts.append(category_count)
        
        if len(category_counts) == 0:
            print('No valid points in the grid')
            print("unique_categories: ", unique_categories)
            print(category_counts)
            
        voxel_categories = torch.stack(category_counts, dim=1).argmax(dim=1)
        voxel_categories = unique_categories[voxel_categories]

        fvdb_grids_finest.append(fvdb_grid)
        semantics_finest.append(voxel_categories)

    # batched version
    fvdb_grids_finest = fvdb.jcat(fvdb_grids_finest)
    semantics_finest = semantics_finest # tensor

    # if the finest grid is not the target grid, we need to coarsen it
    if torch.any(grid_batch_kwargs_target['voxel_sizes'] != grid_batch_kwargs_finest['voxel_sizes']):
        semantics_target = []
        downsample_ratio = int(grid_batch_kwargs_target['voxel_sizes'][0] / grid_batch_kwargs_finest['voxel_sizes'][0])
        fvdb_grids_target = fvdb_grids_finest.coarsened_grid(downsample_ratio)
        fvdb_grids_target_xyz = fvdb_grids_target.grid_to_world(fvdb_grids_target.ijk.float()) # jagged tensor
        fvdb_grids_finest_xyz = fvdb_grids_finest.grid_to_world(fvdb_grids_finest.ijk.float()) # jagged tensor

        for fvdb_grid_target_xyz_i, fvdb_grid_finest_xyz_i, semantics_finest_i in zip(fvdb_grids_target_xyz, fvdb_grids_finest_xyz, semantics_finest):
            semantics_target.append(semantic_from_points(fvdb_grid_target_xyz_i.jdata, fvdb_grid_finest_xyz_i.jdata, semantics_finest_i))
    else:
        fvdb_grids_target = fvdb_grids_finest
        semantics_target = semantics_finest

    return fvdb_grids_target, semantics_target
        

def fill_fvdb_grid_w_semantic_from_points(given_fvdb_grid, points, semantics):
    """
    Args:
        given_fvdb_grid: fvdb.GridBatch
            grid to fill, grid_count = 1
        points: torch.Tensor
            point cloud, shape [N, 3]
        semantics: torch.Tensor
            semantics for each point, shape [N, ]
        grid_batch_kwargs: dict
            voxel_sizes, origins, etc.

    Returns:
        filled_semantics: torch.Tensor
            filled semantics, shape [N_voxel, ] for the given grid
    """

    # now some points fall out of the grid, we need to filter them out. The remaining are valid voxels
    # world_to_grid returns a float tensor coordinate respecting to origins and voxel_sizes, when given any input point cloud coordinate (can be outside the grid)
    pts_vox_idx = given_fvdb_grid.ijk_to_index(given_fvdb_grid.world_to_grid(points).jdata.round().long()).jdata 
    pts_valid_mask = pts_vox_idx >= 0 # -1 means out of voxels

    # filter out invalid points (but usually there is no invalid points, since we have already filtered them out in the mask above)
    valid_pts_vox_idx = pts_vox_idx[pts_valid_mask]
    valid_xyz = points[pts_valid_mask]
    valid_semantics = semantics[pts_valid_mask]

    unique_categories = torch.unique(valid_semantics)
    category_counts = []
    for category in unique_categories:
        category_count = torch_scatter.scatter_sum(
            (valid_semantics == category).float(),
            valid_pts_vox_idx,
            dim=0, dim_size=given_fvdb_grid.total_voxels
        )
        category_counts.append(category_count)

    voxel_categories = torch.stack(category_counts, dim=1).argmax(dim=1)
    voxel_categories = unique_categories[voxel_categories]

    filled_semantics = voxel_categories

    return filled_semantics


if __name__ == '__main__':
    grid = fvdb.GridBatch()
    ijk = torch.tensor([[0, 0, 0]])
    grid.set_from_ijk(ijk, voxel_sizes=[1, 1, 1], origins=[0, 0, 0])
    camera_pose = torch.Tensor(
        [[ 1, 0, 0, 0],
          [0, 0, 1, -5],
          [0,-1, 0, 0],
          [0, 0, 0, 1]]
    ) # [4, 4]
    camera_intrinsic = torch.tensor([30, 20, 15, 10, 30, 20]).float() # [6]

    camera_pose = camera_pose.unsqueeze(0)
    camera_intrinsic = camera_intrinsic.unsqueeze(0)

    depth = get_distance_from_voxel(camera_pose, camera_intrinsic, grid)