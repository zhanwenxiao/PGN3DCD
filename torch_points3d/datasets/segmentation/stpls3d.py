import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
import csv
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil

from torch_points3d.core.data_transform.feature_augment import ChromaticJitter, ChromaticTranslation, ChromaticAutoContrast
from torch_points3d.core.data_transform import GridSampling3D, CylinderSampling, SphereSampling
from torch_points3d.datasets.change_detection.base_siamese_dataset import BaseSiameseDataset
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_points3d.metrics.hkCD_tracker import HKCDTracker
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

IGNORE_LABEL: int = -1
STPLS3D_NUM_CLASSES = 15

INV_OBJECT_LABEL = {
    0: "Ground",
    1: "Build",
    2: "LowVeg",
    3: "MediumVeg",
    4: "HiVeg",
    5: "Vehicle",
    6: "Truck",
    7: "Aircraft",
    8: "MilitaryVec",
    9: "Bike",
    10: "Motorcycle",
    11: "LightPole",
    12: "StreetSign",
    13: "Clutter",
    14: "Fence"
}

OBJECT_COLOR = np.asarray(
    [
        [0, 255, 0],  # Ground
        [0, 0, 255],  # Build
        [0, 255, 255],  # LowVeg
        [255, 255, 0],  # MediumVeg
        [255, 0, 255],  # HiVeg
        [100, 100, 255],  # Vehicle
        [200, 200, 100],  # Truck
        [170, 120, 200],  # Aircraft
        [255, 0, 0],  # MilitaryVec
        [200, 100, 100],  # Bike
        [10, 200, 100],  # Motorcycle
        [200, 200, 200],  # LightPole
        [50, 50, 50],  # StreetSign
        [60, 130, 60],  # Clutter
        [130, 30, 60], # Fence
    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

################################### UTILS #######################################

def to_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    colors = OBJECT_COLOR[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, "S3DIS")
    PlyData([el], byte_order=">").write(file)

def data_augment(pos, rgb, angle=2 * np.pi, paramGaussian=[0.01, 0.05], color_aug=False):
    """
    Random data augmentation
    """
    # random rotation around the Z axis
    angle = (np.random.random()-0.5) * angle
    M = torch.from_numpy(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])).type(
        torch.float32)
    pos[:, :2] = torch.matmul(pos[:, :2], M)  # perform the rotation efficiently

    # random gaussian noise
    sigma, clip = paramGaussian
    # Hint: use torch.clip to clip and np.random.randn to generate gaussian noise
    pos = pos + torch.clip(torch.randn(pos.shape) * sigma, -clip, clip).type(torch.float32)
    # data color augmentation
    if color_aug:
        rgb = _color_jitter(rgb)
        return pos, rgb
    return pos

def _color_jitter(rgb):
    rgb_data = Data(rgb=rgb)
    if random.random() < 0.5:
        chromaticJitter = ChromaticJitter()
        rgb_data = chromaticJitter(rgb_data)

    if random.random() < 0.2:
        chromaticAutoContrast = ChromaticAutoContrast()
        rgb_data = chromaticAutoContrast(rgb_data)

    if random.random() < 0.5:
        chromaticTranslation = ChromaticTranslation()
        rgb_data = chromaticTranslation(rgb_data)
    return rgb_data.rgb



class STPLS3D(Dataset):
    """
    Definition of STPLS3D Dataset
    """

    def __init__(self, sample_per_epoch=6000, filePaths="", split="train", DA=False, pre_transform=None, transform=None, preprocessed_dir="",
                 reload_preproc=False, reload_trees=False, nameInPly="params", comp_norm = False ):
        super(STPLS3D, self).__init__(None, None, pre_transform)
        self.class_labels = OBJECT_LABEL
        self._ignore_label = IGNORE_LABEL
        self.preprocessed_dir = preprocessed_dir
        if not osp.isdir(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
        self.filePaths = filePaths
        self.nameInPly = nameInPly
        self._get_paths()
        self.split = split
        self.DA = DA
        self.pre_transform = pre_transform
        self.transform = None
        self.manual_transform = transform
        self.reload_preproc = reload_preproc
        self.reload_trees = reload_trees
        self.num_classes = STPLS3D_NUM_CLASSES
        self.nb_elt_class = torch.zeros(self.num_classes)
        self.filesPC_prepoc = [None] * len(self.filesPC)
        self.sample_per_epoch = sample_per_epoch

        self.process_files = []

        AVAILABLE_SPLITS = ["Train", "Val", "Test"]

        self.process(comp_normal=comp_norm)
        if self.nb_elt_class.sum() == 0:
            self.get_nb_elt_class()
        self.weight_classes = 1 - self.nb_elt_class / self.nb_elt_class.sum()

    @property
    def raw_file_names(self):
        return ["STPLS3D"]

    @property
    def processed_file_names(self):
        return [s for s in self.AVAILABLE_SPLITS[:-1]]

    def _get_paths(self):
        self.filesPC = []
        globPath = os.scandir(self.filePaths)
        for file in globPath:
            if file.is_file():
                self.filesPC.append(file.path)
        globPath.close()

    def size(self):
        return len(self.filesPC)

    def len(self):
        if self.sample_per_epoch > 0:
            return self.sample_per_epoch

    # def len(self):
    #     return len(self.process_files)

    def get_nb_elt_class(self):
        self.nb_elt_class = torch.zeros(self.num_classes)
        for idx in range(len(self.filesPC)):
            pc = torch.load(osp.join(self.preprocessed_dir, 'pc_{}.pt'.format(idx)))
            cpt = torch.bincount(pc.y)
            for c in range(cpt.shape[0]):
                self.nb_elt_class[c] += cpt[c]

    def hand_craft_process(self, comp_normal=False):
        existfile = True
        for idx in range(len(self.filesPC)):
            exist_file = existfile and osp.isfile(osp.join(self.preprocessed_dir, 'pc_{}.pt'.format(idx)))
            self.process_files.append(osp.join(self.preprocessed_dir, 'pc_{}.pt'.format(idx)))

        if not self.reload_preproc or not exist_file:
            for idx in range(len(self.filesPC)):
                pos, color, label, inst = self.clouds_loader(idx)
                color_norm = color.float() / 255.0
                pc = Data(pos=pos, f=color_norm, y=label, inst_y=inst)
                if comp_normal:
                    normal = getFeaturesfromPDAL(pc.pos.numpy())
                    pc.norm = torch.from_numpy(normal)
                if self.pre_transform is not None:
                    pc = self.pre_transform(pc)
                cpt = torch.bincount(pc.y)
                for c in range(cpt.shape[0]):
                    self.nb_elt_class[c] += cpt[c]
                torch.save(pc, osp.join(self.preprocessed_dir, 'pc_{}.pt'.format(idx)))
                self.process_files.append(osp.join(self.preprocessed_dir, 'pc_{}.pt'.format(idx)))

    def process(self, comp_normal = False):
        self.hand_craft_process(comp_normal)

    def get(self, idx):
        if self.pre_transform is not None:
            pos, color, label, inst = self._preproc_clouds_loader(idx)
        else:
            pos, color, label, inst = self.clouds_loader(idx, nameInPly=self.nameInPly)

        color_norm = color.float() / 255.0
        data = Data(pos=pos, f=color_norm, y=label, inst_y=inst)
        return data.contiguous()

    def clouds_loader(self, area):
        print("Loading " + self.filesPC[area])
        points = self.cloud_loader(self.filesPC[area])
        coordinates, color, labels, instances = (
            np.vstack((points['x'] - min(points['x']), points['y'] - min(points['y']), points['z'] - min(points['z']))).T,
            np.vstack((points['r'], points['g'], points['b'])).T,
            points['label'].values,
            points['inst'].values
        )
        return torch.from_numpy(coordinates).type(torch.float), torch.from_numpy(color).type(torch.int), torch.from_numpy(labels).type(torch.int64), torch.from_numpy(instances).type(torch.int64)

    def _preproc_clouds_loader(self, area):
        data_pc = torch.load(osp.join(self.preprocessed_dir, 'pc_{}.pt'.format(area)))
        return data_pc.pos, data_pc.f, data_pc.y, data_pc.inst_y

    def cloud_loader(self, pathPC, cuda=False):
        """
        load a tile and returns points features (normalized xyz + intensity) and
        ground truth
        INPUT:
        pathPC = string, path to the tile of PC
        OUTPUT
        pc_data, [n x 3] float array containing points coordinates and intensity
        lbs, [n] long int array, containing the points semantic labels
        """
        assert os.path.isfile(pathPC)
        pc_data = pd.read_table(pathPC, sep=',', header=None, names=['x', 'y', 'z', 'r', 'g', 'b', 'label', 'inst'])

        if cuda:  # put the cloud data on the GPU memory
            pc_data = pc_data.cuda()
        return pc_data

    @property
    def num_features(self):
        return 3


class STPLS3DSphere(STPLS3D):
    """ Small variation of Urb3DCD that allows random sampling of spheres
    within an Area during training and validation. Spheres have a radius of 2m. If sample_per_epoch is not specified, spheres
    are taken on a 2m grid.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, sample_per_epoch=100, radius=2, fix_cyl=False, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = GridSampling3D(size=radius / 2.0) # GridSampling3D(size=radius / 10.0)
        self.fix_cyl = fix_cyl
        super().__init__(*args, **kwargs)
        self._prepare_centers()
        # Trees are built in case it needs, now don't need to compute anymore trees
        self.reload_trees = True

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return self.grid_regular_centers.shape[0]

    def get(self, idx, dc = False):
        if self._sample_per_epoch > 0:
            if self.fix_cyl:
                centre = self._centres_for_sampling_fixed[idx, :3]
                area_sel = self._centres_for_sampling_fixed[idx, 3].int()
                data = self._load_save(area_sel)
                sphere_sampler = SphereSampling(self._radius, centre, align_origin=False)
                dataPC = Data(pos=data.pos)
                setattr(dataPC, SphereSampling.KDTREE_KEY, data.KDTREE_KEY_PC)
                dataPC_sphere = sphere_sampler(dataPC)
                pair_spheres = Data(pos=dataPC_sphere.pos, y=dataPC_sphere.y)
                return pair_spheres
            else:
                return self._get_random()
        else:
            centre = self.grid_regular_centers[idx, :3]
            area_sel = self.grid_regular_centers[idx, 3].int()
            data = self._load_save(area_sel)
            sphere_sampler = SphereSampling(self._radius, centre, align_origin=False)
            dataPC = Data(pos=data.pos)
            setattr(dataPC, SphereSampling.KDTREE_KEY, data.KDTREE_KEY_PC)
            dataPC_sphere = sphere_sampler(dataPC)
            if self.manual_transform is not None:
                dataPC_sphere = self.manual_transform(dataPC_sphere)
            pair_spheres = Data(pos=dataPC_sphere.pos, y=dataPC_sphere.y)
            return pair_spheres.contiguous()


    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        #  choice of the corresponding PC if several PCs are loaded
        area_sel = centre[3].int()
        data = self._load_save(area_sel)
        sphere_sampler = SphereSampling(self._radius, centre[:3], align_origin=False)
        dataPC = Data(pos=data.pos, f=data.f, y=data.y, inst_y=data.inst_y)
        setattr(dataPC, SphereSampling.KDTREE_KEY, data.KDTREE_KEY_PC)
        dataPC_sphere = sphere_sampler(dataPC)
        if self.manual_transform is not None:
            dataPC_sphere = self.manual_transform(dataPC_sphere)
        data_sphere = Data(pos=dataPC_sphere.pos, f=dataPC_sphere.f, y=dataPC_sphere.y, inst_y=dataPC_sphere.inst_y)
        return data_sphere

    def _prepare_centers(self):
        self._centres_for_sampling = []
        grid_sampling = GridSampling3D(size=self._radius / 2)
        self.grid_regular_centers = []
        for i in range(len(self.filesPC)):
            data = self._load_save(i)
            if self._sample_per_epoch > 0:
                dataPC = Data(pos=data.pos, f=data.f, y=data.y, inst_y=data.inst_y)
                low_res = self._grid_sphere_sampling(dataPC)
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
            else:
                # Get regular center on PC1, PC0 will be sampled using the same center
                dataPC = Data(pos=data.pos, f=data.f, y=data.y, inst_y=data.inst_y)
                grid_sample_centers = grid_sampling(dataPC.clone())
                centres = torch.empty((grid_sample_centers.pos.shape[0], 4), dtype=torch.float)
                centres[:, :3] = grid_sample_centers.pos
                centres[:, 3] = i
                self.grid_regular_centers.append(centres)

        if self._sample_per_epoch > 0:
            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            print(uni_counts)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            print(self._label_counts)
            self._labels = uni
            self.weight_classes = torch.from_numpy(self._label_counts).type(torch.float)
            if self.fix_cyl:
                self._centres_for_sampling_fixed = []
                # choice of cylinders for all the training
                np.random.seed(1)
                chosen_labels = np.random.choice(self._labels, p=self._label_counts, size=(self._sample_per_epoch, 1))
                uni, uni_counts = np.unique(chosen_labels, return_counts=True)
                print("fixed cylinder", uni, uni_counts)
                for c in range(uni.shape[0]):
                    valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, -1] == uni[c]]
                    centres_idx = np.random.randint(low = 0, high=valid_centres.shape[0], size=(uni_counts[c],1))
                    self._centres_for_sampling_fixed.append(np.squeeze(valid_centres[centres_idx,:], axis=1))
                self._centres_for_sampling_fixed = torch.cat(self._centres_for_sampling_fixed, 0)
        else:
            self.grid_regular_centers = torch.cat(self.grid_regular_centers, 0)

    def _load_save(self, i):
        if self.pre_transform is not None:
            pc, rgb, label, inst = self._preproc_clouds_loader(i)
        else:
            pc, rgb, label, inst = self.clouds_loader(i)
        data = Data(pos=pc, f=rgb, y=label, inst_y=inst)
        path = self.filesPC[i]
        name_tree = os.path.basename(path).split(".")[0] + "_radius" + str(int(self._radius)) + "_" + str(i) + ".p"
        path_treesPC = os.path.join(self.preprocessed_dir, "tp3DTree", name_tree)  # osp.dirname(path)
        if self.reload_trees and osp.isfile(path_treesPC):
            file = open(path_treesPC, "rb")
            tree = pickle.load(file)
            file.close()
            data.KDTREE_KEY_PC = tree
        else:
            # tree not existing yet should be saved
            # test if tp3D directory exists
            if not osp.isdir(os.path.join(self.preprocessed_dir, "tp3DTree")):
                os.makedirs(osp.join(self.preprocessed_dir, "tp3DTree"))
            tree = KDTree(np.asarray(pc), leaf_size=10)
            file = open(path_treesPC, "wb")
            pickle.dump(tree, file)
            file.close()
            data.KDTREE_KEY_PC = tree
        return data


class STPLS3DCylinder(STPLS3DSphere):
    def get(self, idx):
        if self._sample_per_epoch > 0:
            if self.fix_cyl:
                pair_correct = False
                while not pair_correct and idx < self._centres_for_sampling_fixed.shape[0]:
                    centre = self._centres_for_sampling_fixed[idx, :3]
                    area_sel = self._centres_for_sampling_fixed[
                        idx, 3].int()  # ---> ici choix du pc correspondant si pls pc chargés
                    data = self._load_save(area_sel)
                    cylinder_sampler = CylinderSampling(self._radius, centre, align_origin=False)
                    dataPC = Data(pos=data.pos, f=data.f, y=data.y, inst_y=data.inst_y, idx=torch.arange(data.pos.shape[0]).reshape(-1))
                    setattr(dataPC, CylinderSampling.KDTREE_KEY, dataPC.KDTREE_KEY_PC)
                    dataPC_cyl = cylinder_sampler(dataPC)
                    data_cylinders = Data(pos=dataPC_cyl.pos, x=dataPC_cyl.f, y=dataPC_cyl.y, inst_y=dataPC_cyl.inst_y, idx=dataPC_cyl.idx, area=area_sel)
                    try:
                        pair_correct = True
                    except:
                        print(data_cylinders.pos.shape)
                        print(data_cylinders.pos_target.shape)
                        idx += 1
                return data_cylinders
            else:
                return self._get_random()
        else:
            pair_correct = False
            while not pair_correct and idx < self.grid_regular_centers.shape[0]:
                centre = self.grid_regular_centers[idx, :3]
                area_sel = self.grid_regular_centers[idx, 3].int()
                data = self._load_save(area_sel)
                cylinder_sampler = CylinderSampling(self._radius, centre, align_origin=False)
                # dataPC0 = Data(pos=pair.pos, idx=torch.arange(pair.pos.shape[0]).reshape(-1))
                dataPC = Data(pos=data.pos, f=data.f, y=data.y, inst_y=data.inst_y, idx=torch.arange(data.pos.shape[0]).reshape(-1))
                setattr(dataPC, CylinderSampling.KDTREE_KEY, dataPC.KDTREE_KEY_PC)
                dataPC_cyl = cylinder_sampler(dataPC)
                try:
                    if self.manual_transform is not None:
                        dataPC_cyl = self.manual_transform(dataPC_cyl)
                    data_cylinders = Data(pos=dataPC_cyl.pos, x=dataPC_cyl.f, y=dataPC_cyl.y, inst_y=dataPC_cyl.inst_y, idx=dataPC_cyl.idx, area=area_sel)
                    if self.DA:
                        data_cylinders.pos = data_augment(data_cylinders.pos, data_cylinders.rgb)
                    pair_correct = True
                except:
                    print('pair not correct')
                    idx += 1
            return data_cylinders

    def _get_random(self):
        # Random cylinder biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        #  choice of the corresponding PC if several PCs are loaded
        area_sel = centre[3].int()
        data = self._load_save(area_sel)
        cylinder_sampler = CylinderSampling(self._radius, centre[:3], align_origin=False)
        dataPC = Data(pos=data.pos, f=data.f, y=data.y, inst_y=data.inst_y)
        setattr(dataPC, CylinderSampling.KDTREE_KEY, data.KDTREE_KEY_PC)
        dataPC_cyl = cylinder_sampler(dataPC)
        if self.manual_transform is not None:
            dataPC_cyl = self.manual_transform(dataPC_cyl)
        data_cyl = Data(pos=dataPC_cyl.pos, x=dataPC_cyl.f, y=dataPC_cyl.y, inst_y=dataPC_cyl.inst_y)
        if self.DA:
            data_cyl.pos = data_augment(data_cyl.pos, data_cyl.rgb)
        return data_cyl

    def _load_save(self, i):
        if self.pre_transform is not None:
            pos, color, label, inst = self._preproc_clouds_loader(i)
        else:
            pos, color, label, inst = self.clouds_loader(i)

        data = Data(pos=pos, f=color, y=label, inst_y=inst)
        data = self._get_tree(data, i)
        return data.contiguous()

    def _get_tree(self, data, i):
        path = self.filesPC[i]
        name_tree = os.path.basename(path).split(".")[0] + "_2D_radius" + str(int(self._radius)) + "_" + str(i) + ".p"
        path_treesPC = os.path.join(self.preprocessed_dir, "tp3DTree", name_tree)
        if self.reload_trees and osp.isfile(path_treesPC):
            try:
                file = open(path_treesPC, "rb")
                tree = pickle.load(file)
                file.close()
                data.KDTREE_KEY_PC = tree
            except:
                print('not able to load tree')
                print(file)
                print(data)
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                data.KDTREE_KEY_PC = tree
        else:
            # tree not existing yet should be saved
            # test if tp3D directory is existing
            if not os.path.isdir(os.path.join(self.preprocessed_dir, "tp3DTree")):
                os.makedirs(os.path.join(self.preprocessed_dir, "tp3DTree"))
            tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
            file = open(path_treesPC, "wb")
            pickle.dump(tree, file)
            file.close()
            data.KDTREE_KEY_PC = tree
        return data


class STPLS3DDataset(BaseDataset):
    """ Wrapper around S3DISSphere that creates train and test datasets.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get("sampling_format", "sphere")
        dataset_cls = STPLS3DCylinder if sampling_format == "cylinder" else STPLS3DSphere

        self.pre_transform = dataset_opt.get("pre_transforms", None)
        super().__init__(dataset_opt)
        self.radius = float(self.dataset_opt.radius)
        self.sample_per_epoch = int(self.dataset_opt.sample_per_epoch)
        self.DA = self.dataset_opt.DA
        self.TTA = False
        self.preprocessed_dir = self.dataset_opt.preprocessed_dir
        self.train_dataset = dataset_cls(
            filePaths=self.dataset_opt.dataTrainFile,
            split="train",
            radius=self.radius,
            sample_per_epoch=self.sample_per_epoch,
            DA=self.DA,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Train"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nameInPly=self.dataset_opt.nameInPly,
            fix_cyl=self.dataset_opt.fix_cyl,
        )
        self.val_dataset = dataset_cls(
            filePaths=self.dataset_opt.dataValFile,
            split="val",
            radius=self.radius,
            sample_per_epoch=int(self.sample_per_epoch / 2),
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Val"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nameInPly=self.dataset_opt.nameInPly,
            fix_cyl=self.dataset_opt.fix_cyl,
        )
        # self.test_dataset = dataset_cls(
        #     filePaths=self.dataset_opt.dataTestFile,
        #     split="test",
        #     radius=self.radius,
        #     sample_per_epoch=-1,
        #     pre_transform=self.pre_transform,
        #     preprocessed_dir=osp.join(self.preprocessed_dir, "Test"),
        #     reload_preproc=self.dataset_opt.load_preprocessed,
        #     reload_trees=self.dataset_opt.load_trees,
        #     nameInPly=self.dataset_opt.nameInPly,
        # )

    @property
    def train_data(self):
        if type(self.train_dataset) == list:
            return self.train_dataset[0]
        else:
            return self.train_dataset

    @property
    def val_data(self):
        if type(self.val_dataset) == list:
            return self.val_dataset[0]
        else:
            return self.val_dataset

    @property
    def test_data(self):
        if type(self.test_dataset) == list:
            return self.test_dataset[0]
        else:
            return self.test_dataset

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save s3dis predictions to disk using s3dis color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool, full_pc=False, full_res=False):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.stpls3d_tracker import STPLS3DTracker

        return STPLS3DTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log,
                                 full_pc=full_pc, full_res=full_res, ignore_label=IGNORE_LABEL)