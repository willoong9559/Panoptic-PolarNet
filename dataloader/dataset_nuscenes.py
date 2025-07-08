import os
import numpy as np
import yaml
from pathlib import Path
from torch.utils import data

from nuscenes import NuScenes
from nuscenes.utils import splits
# from nuscenes.lidarseg.lidarseg import NuScenesLidarSeg as NuScenes

map_name_from_general_to_segmentation_class = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
    'flat.driveable_surface': 'driveable_surface',
    'flat.other': 'other_flat',
    'flat.sidewalk': 'sidewalk',
    'flat.terrain': 'terrain',
    'static.manmade': 'manmade',
    'static.vegetation': 'vegetation',
    'noise': 'ignore',
    'static.other': 'ignore',
    'vehicle.ego': 'ignore'
}

map_name_from_segmentation_class_to_segmentation_index = {
    'ignore': 0,
    'barrier': 1,
    'bicycle': 2,
    'bus': 3,
    'car': 4,
    'construction_vehicle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'trailer': 9,
    'truck': 10,
    'driveable_surface': 11,
    'other_flat': 12,
    'sidewalk': 13,
    'terrain': 14,
    'manmade': 15,
    'vegetation': 16
}

map_name_from_segmentation_class_to_thing_class = {
    0: False, # ignore
    1: True, # pedestrian
    2: True, # barrier
    3: True, # bicycle
    4: True, # bus
    5: True, # car
    6: True, # construction_vehicle
    7: True, # motorcycle
    8: True, # trailer
    9: True, # truck
    10: False, # road
    11: False, # other_ground
    12: False, # sidewalk
    13: False, # terrain
    14: False, # manmade
    15: False # vegetation
}

# class NuScenesNoAttr(NuScenes):
#     def __init__(self, version, dataroot, verbose=True):
#         self.table_names = [
#             'category', 'attribute', 'visibility', 'instance', 'sensor',
#             'calibrated_sensor', 'ego_pose', 'log', 'scene', 'sample',
#             'sample_data', 'sample_annotation', 'map'
#         ]
#         self.table_names.remove('attribute')  # 移除 attribute
#         super().__init__(version, dataroot, verbose)

#     def __load_table__(self, table_name: str):
#         if table_name == 'attribute':
#             setattr(self, table_name, [])  # 给个空列表
#         else:
#             super().__load_table__(table_name)

class Nuscenes(data.Dataset):
    def __init__(self, data_path, version = 'v1.0-trainval', split = 'train', return_ref = False):
        self.split = split
        self.data_path = data_path
        self.return_ref = return_ref
        
        # 获取点云文件列表
        self.lidar_path = os.path.join(data_path, 'velodyne')
        self.panoptic_path = os.path.join(data_path, 'panoptic')
        
        # 获取所有点云文件
        self.lidar_files = sorted([f for f in os.listdir(self.lidar_path) if f.endswith('.bin')])
        
        # 根据split划分数据集
        if split == 'train':
            self.lidar_files = self.lidar_files[:int(len(self.lidar_files)*0.8)]
        elif split == 'val':
            self.lidar_files = self.lidar_files[int(len(self.lidar_files)*0.8):]
            
        print(f'{split}: {len(self.lidar_files)} files')
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.lidar_files)
    
    def __getitem__(self, index):
        # 加载点云数据
        lidar_file = self.lidar_files[index]
        lidar_path = os.path.join(self.lidar_path, lidar_file)
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))
        
        # 加载全景分割标签
        panoptic_file = lidar_file.replace('velodyne', 'panoptic')
        panoptic_path = os.path.join(self.panoptic_path, panoptic_file)
        if os.path.exists(panoptic_path):
            annotated_data = np.fromfile(panoptic_path, dtype=np.uint8).reshape((-1, 1))
        else:
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:,0], dtype=int), axis=1)
            
        data_tuple = (raw_data[:,:3], annotated_data)
        if self.return_ref:
            data_tuple += (raw_data[:,3],)
        return data_tuple

    def change_split(self, s):
        assert s in ['train', 'val']
        self.split = s

def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break

        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes

def get_path_infos(nusc,train_scenes,val_scenes):
    train_token_list = []
    val_token_list = []
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        data_token = sample['data']['LIDAR_TOP']
        if scene_token in train_scenes:
            train_token_list.append(data_token)
        else:
            val_token_list.append(data_token)
    return train_token_list, val_token_list

if __name__ == '__main__':
    data_path = '/workspace/Panoptic-PolarNet-main/data/nuScenes-panoptic-v1.0-all'
    # dataset = Nuscenes(data_path+"/v1.0-mini", version = 'v1.0-mini', return_ref = False)
    train_pt_dataset = Nuscenes(data_path, version = 'v1.0-trainval', split = 'train', return_ref = True)
    val_pt_dataset = Nuscenes(data_path, version = 'v1.0-trainval', split = 'val', return_ref = True)
    data = train_pt_dataset[0]
