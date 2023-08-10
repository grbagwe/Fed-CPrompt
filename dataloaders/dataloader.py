from __future__ import print_function
import os
import numpy as np
import sys
import pickle
import torch
import torch.utils.data as data
import pickle
import random
import yaml
import torchvision.datasets as datasets
from PIL import Image
from .utils import download_url, check_integrity


class iDataset_Fed(data.Dataset):

    def __init__(self, root,
                train=True,
                transform=None,
                download_Flag=True,
                lab=True,
                swap_dset=None,
                tasks=None,
                seed=-1,
                rand_split=False,
                validation=False,
                k_folds=5,
                num_clients=None,
                iid=0):
        self.root = os.path.expanduser(root)  # root dir to save the data
        self.transform = transform
        self.train = train  # training or testing phase
        self.seed = seed
        self.validation = validation
        self.tasks = tasks
        self.download_flag = download_Flag
        self.num_clients = num_clients

        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap class labels
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # total data
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        if self.validation:
            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)

        # use 90% of the data for training and 10% for validation from all classes
        if  self.train:
            self.data = self.data[:int(0.9 * len(self.data))]
            self.targets = self.targets[:int(0.9 * len(self.targets))]
        if self.validation:
            print("Validation is true")

            self.data = self.data[int(0.9 * len(self.data)):]
            self.targets = self.targets[int(0.9 * len(self.targets)):]
            # print("len of validation data ", len(self.targets))


        if self.train:  # in trainign phase
            self.archive = []

            for task in self.tasks:  # cycle through the tasks
                locs = np.isin(self.targets, task).nonzero()[0]  # locations of the current task
                if iid == 0:  # uniform distribution
                    # print("len iid ", len(locs))
                    client_locs = np.array_split(locs, self.num_clients)
                    # client_locs = np.split(locs, self.num_clients)
                if iid > 0:  # if using non-iid distribution (dircichlet distribution)

                    # non iid distribution following the implementation in
                    # https://github.com/Xtra-Computing/NIID-Bench/blob/main/partition.py
                    beta = iid  # the beta parameter
                    N = len(locs) # number of instances in the current task
                    min_size = 0  # the min size of each dataset
                    min_require_size = 10  # how much min data each client should get
                    while min_size < min_require_size:
                        idx_batch = [[] for _ in range(num_clients)]
                        for k in task:
                            idx_k = np.where(self.targets== k)[0] # locations of the current task
                        
                            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                            proportions = np.array([p * (len(idx_j) < N / num_clients)
                                                    for p, idx_j in zip(proportions, idx_batch)])
                            proportions = proportions / proportions.sum()
                            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                            idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                                        zip(idx_batch, np.split(idx_k, proportions))]
                            min_size = min([len(idx_j) for idx_j in idx_batch])
                    client_locs = idx_batch
                # if iid > 0:  # if using non-iid distribution (dircichlet distribution)

                #     # non iid distribution following the implementation in
                #     # https://github.com/Xtra-Computing/NIID-Bench/blob/main/partition.py
                #     beta = iid  # the beta parameter
                #     N = len(locs)
                #     min_size = 0  # the min size of each dataset
                #     min_require_size = 32  # how much min data each client should get
                #     while min_size < min_require_size:
                #         idx_batch = [[] for _ in range(num_clients)]
                #         proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                #         proportions = np.array([p * (len(idx_j) < N / num_clients)
                #                                 for p, idx_j in zip(proportions, idx_batch)])
                #         proportions = proportions / proportions.sum()
                #         proportions = (np.cumsum(proportions) * len(locs)).astype(int)[:-1]
                #         idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                #                      zip(idx_batch, np.split(locs, proportions))]
                #         min_size = min([len(idx_j) for idx_j in idx_batch])
                #     client_locs = idx_batch

                client_data = []
                for client in range(self.num_clients):
                    client_data.append((self.data[client_locs[client]].copy(),
                                        self.targets[client_locs[client]].copy()))
                self.archive.append(client_data)
        
        else:  # in test set
            self.archive = []
            for task in self.tasks:
                locs = np.isin(self.targets, task).nonzero()[0]  # locations of the current task
                self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))
        if self.validation:
            self.archive = []
            for task in self.tasks:
                locs = np.isin(self.targets, task).nonzero()[0]
                self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))
            

    def __getitem__(self, index, simple=False):
        """
                Args:
                    index (int): Index
                Returns:
                    tuple: (image, target) where target is index of the target class.
        """
        # code from torchvision
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target]

    def load_dataset(self, t, train=True, client=-1):
        if train:  # in the training model
            if client >= 0:
                self.data, self.targets = self.archive[t][client]
            else:
                self.data, self.targets = self.archive[t]

        else:
            self.data = np.concatenate([self.archive[s][0] for s in range(t + 1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t + 1)], axis=0)
        print("targets in dataloader", np.unique(self.targets))

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def load(self):
        pass


class iCIFAR10_Fed(iDataset_Fed):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iDataset Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    im_size = 32
    nch = 3

    def load(self):

        # download dataset
        if self.download_flag:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train or self.validation:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.course_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                if 'coarse_labels' in entry:
                    self.course_targets.extend(entry['coarse_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


class iCIFAR100_Fed(iCIFAR10_Fed):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iCIFAR10 Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    im_size = 32
    nch = 3


class iIMAGENET_R_fed(iDataset_Fed):
    
    base_folder = 'imagenet-r'
    im_size=224
    nch=3
    def load(self):
        # # HISTORY - HOW WE SAMPLED THE SPLITS THE FIRST TIME (DO NOT CHANGE!!!!!!)
        # # a - load all data
        # self.data, self.targets = [], []
        # images_path = os.path.join(self.root, self.base_folder)
        # data_dict = get_data(images_path)
        # y = 0
        # for key in data_dict.keys():
        #     num_y = len(data_dict[key])
        #     self.data.extend([data_dict[key][i] for i in np.arange(0,num_y)])
        #     self.targets.extend([y for i in np.arange(0,num_y)])
        #     y += 1
        # # b - sample all data
        # n_data = len(self.targets)
        # index_sample = [i for i in range(n_data)]
        # import random
        # random.seed(0)
        # random.shuffle(index_sample)
        # if self.train or self.validation:
        #     index_sample = index_sample[:int(0.8*n_data)]
        # else:
        #     index_sample = index_sample[int(0.8*n_data):]
        # self.data = [self.data[i] for i in index_sample]
        # self.targets = [self.targets[i] for i in index_sample]
        # data_config = {'data': self.data, 'targets': self.targets}
        # # c - save for future loading
        # if self.train or self.validation:
        #     with open('dataloaders/splits/imagenet-r_train.yaml', 'w') as f:
        #         yaml.dump(data_config, f, default_flow_style=False)
        # else:
        #     with open('dataloaders/splits/imagenet-r_test.yaml', 'w') as f:
        #         yaml.dump(data_config, f, default_flow_style=False)

        # load splits from config file
        if self.train or self.validation:
            data_config = yaml.load(open('dataloaders/splits/imagenet-r_train.yaml', 'r'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open('dataloaders/splits/imagenet-r_test.yaml', 'r'), Loader=yaml.Loader)
        self.data = data_config['data']
        self.targets = data_config['targets']

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        img = jpg_image_to_array(img_path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target], self.t

    
    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
    

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:      
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr