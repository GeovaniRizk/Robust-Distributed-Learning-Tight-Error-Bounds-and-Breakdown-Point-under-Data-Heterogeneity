import os, sys
from shutil import move
import fnmatch
import json
import numpy as np
import torch
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.folder import default_loader
from PIL import Image
import warnings




# def key_sort(element):
#     print(element)
#     print(element[9:-27])
#     return  element[9:-27]

class FEMNIST(VisionDataset):
    """
    classes: 10 digits, 26 lower cases, 26 upper cases.
    We use torch.save, torch.load in this dataset
    """

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, user_list: list = None):
        super(FEMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        """
        0 <= any user in user_list < total_users
        """
        self.train = train
        self.user_list = user_list

        if download:
            self.download()

        if not self._check_exists():
            raise FileNotFoundError("Dataset not found. You can use download=True to download it")

        self.total_num_users = torch.load(os.path.join(self.processed_folder, "num_users.pt"))

        if self.user_list is not None:
            self.num_users = len(self.user_list)
        else:
            self.user_list = [i for i in range(self.total_num_users)]
            self.num_users = self.total_num_users

        if self.train:
            self.data, self.targets = self.load(train=True)
        else:
            self.data, self.targets = self.load(train=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # Needs 0~255, uint8 scale
        # img = Image.fromarray(np.uint8(255 * (1 - img.numpy())), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def all_data_folder(self):
        return os.path.join(self.root, "femnist", "data", "raw_data")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.raw_folder, "train")) and
                os.path.exists(os.path.join(self.raw_folder, "test")) and
                os.path.exists(os.path.join(self.processed_folder, "num_users.pt")))

    def download(self):     
        if self._check_exists():
            print("Data already downloaded.")
            return

        if os.path.isdir(self.raw_folder) and len(os.listdir(self.raw_folder)) != 0:
            self.process()
            return

        root = self.root
        if not os.path.isdir(root):
            os.mkdir(root)

        if not os.path.exists(self.all_data_folder):
            # download from https://github.com/TalwalkarLab/leaf/tree/master/data/femnist
            input_str = input("Downloading and processing data will take "
                              "approximately 10 to 30 minutes, and it consumes about 15GB of storage. Continue? [y/n]")
            if input_str.lower() in ["y", "yes"]:
                os.system(rf"git clone https://github.com/TalwalkarLab/leaf.git {root}/github_repo"
                          rf"&& cd {root}/github_repo/data/femnist"
                          r"&& ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample --smplseed 1549786595 --spltseed 1549786796"
                          r"&& cd ../../.."
                          r"&& mv github_repo/data/utils utils"
                          r"&& mv github_repo/data/femnist femnist"
                          r"&& rm -rf github_repo")
                os.makedirs(self.raw_folder, exist_ok=False)
                os.system(rf"cd {root}"
                          r"&& rm -r femnist/data/rem_user_data femnist/data/sampled_data"
                          r"&& mv femnist/data/test raw/ && mv femnist/data/train raw/")
            else:
                print("Exiting...")
                exit()
        else:
            if os.path.exists(os.path.join(root, "data", "rem_user_data")):
                os.system(rf"rm -r {root}/data/rem_user_data")
            if os.path.exists(os.path.join(root, "data", "sampled_data")):
                os.system(rf"rm -r {root}/data/sampled_data")
            if os.path.exists(os.path.join(root, "data", "train")):
                os.system(rf"rm -r {root}/data/train")
            if os.path.exists(os.path.join(root, "data", "test")):
                os.system(rf"rm -r {root}/data/test")
            if os.path.exists(os.path.join(root, "raw")):
                os.system(rf"rm -r {root}/raw")
            if os.path.exists(os.path.join(root, "processed")):
                os.system(rf"rm -r {root}/processed")

            os.makedirs(self.raw_folder, exist_ok=False)
            os.system(rf"cd {root}/femnist"
                      r"&& ./preprocess.sh -s niid --sf 0.10 -k 0 -t sample"
                      r"&& cd .."
                      r"&& rm -r femnist/data/rem_user_data femnist/data/sampled_data"
                      r"&& mv femnist/data/test raw/ && mv femnist/data/train raw/")

        self.process()

    def process(self):
        print("Processing data...")

        if not os.path.isdir(self.processed_folder):
            os.makedirs(self.processed_folder)

        total_users_train = 0
        list_train_f = [f for f in os.listdir(os.path.join(self.raw_folder, "train")) if
                        fnmatch.fnmatch(f, "*.json")]
        # print('hello', list_train_f)

        list_train_f.sort(key=lambda fname: int(fname[9:-27]))

        # sys.exit()

        print(list_train_f)

        for filename in list_train_f:
            with open(os.path.join(self.raw_folder, "train", filename)) as file:
                data = json.load(file)
                for user_name, val in data["user_data"].items():
                    # key: user name
                    # val: dict {x: x_data, y: y_data}
                    x = torch.tensor(val["x"]).reshape((-1, 1,28,28))
                    y = torch.tensor(val["y"])

                    torch.save((x, y), os.path.join(self.processed_folder, "train_{}.pt".format(total_users_train)))
                    total_users_train += 1

        total_users_test = 0
        list_test_f = [f for f in os.listdir(os.path.join(self.raw_folder, "test")) if fnmatch.fnmatch(f, "*.json")]
        list_test_f.sort(key=lambda fname: int(fname[9:-26]))

        for filename in list_test_f:
            with open(os.path.join(self.raw_folder, "test", filename)) as file:
                data = json.load(file)
                for user_name, val in data["user_data"].items():
                    # key: user name
                    # val: dict {x: x_data, y: y_data}
                    x = torch.tensor(val["x"]).reshape((-1, 1,28,28))
                    y = torch.tensor(val["y"])

                    torch.save((x, y), os.path.join(self.processed_folder, "test_{}.pt").format(total_users_test))
                    total_users_test += 1

        assert total_users_train == total_users_test
        torch.save(total_users_train, os.path.join(self.processed_folder, "num_users.pt"))
        print("Done. {} users processed.".format(total_users_train))

    def load(self, train):
        if train:
            prf = "train"
        else:
            prf = "test"

        data_list, label_list = [], []
        for user_id in self.user_list:
            x, y = torch.load(os.path.join(self.processed_folder, "{}_{}.pt".format(prf, user_id)))
            data_list.append(x)
            label_list.append(y)
        return torch.cat(data_list, dim=0), torch.cat(label_list, dim=0)
    
    def load_client_dataset(self, train, client_id):
        if train:
            prf = "train"
        else:
            prf = "test"

        # data_list, label_list = [], []
        # print(self.user_list)

        x, y = torch.load(os.path.join(self.processed_folder, "{}_{}.pt".format(prf, client_id)))
        # print(x.shape, y.shape)
        # sys.exit()
        # for user_id in self.user_list:
        #     x, y = torch.load(os.path.join(self.processed_folder, "{}_{}.pt".format(prf, user_id)))
        #     data_list.append(x)
        #     label_list.append(y)
        return x, y



# import os, sys
# from shutil import move
# import fnmatch
# import json
# import numpy as np
# import torch
# from torchvision.datasets import VisionDataset, ImageFolder
# from torchvision.datasets.folder import default_loader
# from PIL import Image
# import warnings




# # def key_sort(element):
# #     print(element)
# #     print(element[9:-27])
# #     return  element[9:-27]

class workerFEMNIST(VisionDataset):
    """
    classes: 10 digits, 26 lower cases, 26 upper cases.
    We use torch.save, torch.load in this dataset
    """

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, user_list: list = None, client_id=0):
        super(workerFEMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        """
        0 <= any user in user_list < total_users
        """
        self.train = train
        self.user_list = user_list
        self.client_id = client_id

        if download:
            self.download()

        if not self._check_exists():
            raise FileNotFoundError("Dataset not found. You can use download=True to download it")

        self.total_num_users = torch.load(os.path.join(self.processed_folder, "num_users.pt"))

        if self.user_list is not None:
            self.num_users = len(self.user_list)
        else:
            self.user_list = [i for i in range(self.total_num_users)]
            self.num_users = self.total_num_users

        if self.train:
            self.data, self.targets = self.load(train=True)
        else:
            self.data, self.targets = self.load(train=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # Needs 0~255, uint8 scale
        # img = Image.fromarray(np.uint8(255 * (1 - img.numpy())), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def all_data_folder(self):
        return os.path.join(self.root, "femnist", "data", "raw_data")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.raw_folder, "train")) and
                os.path.exists(os.path.join(self.raw_folder, "test")) and
                os.path.exists(os.path.join(self.processed_folder, "num_users.pt")))

    def download(self):     
        if self._check_exists():
            print("Data already downloaded.")
            return

        if os.path.isdir(self.raw_folder) and len(os.listdir(self.raw_folder)) != 0:
            self.process()
            return

        root = self.root
        if not os.path.isdir(root):
            os.mkdir(root)

        if not os.path.exists(self.all_data_folder):
            # download from https://github.com/TalwalkarLab/leaf/tree/master/data/femnist
            input_str = input("Downloading and processing data will take "
                              "approximately 10 to 30 minutes, and it consumes about 15GB of storage. Continue? [y/n]")
            if input_str.lower() in ["y", "yes"]:
                os.system(rf"git clone https://github.com/TalwalkarLab/leaf.git {root}/github_repo"
                          rf"&& cd {root}/github_repo/data/femnist"
                          r"&& ./preprocess.sh -s niid --sf 0.10 -k 0 -t sample"
                          r"&& cd ../../.."
                          r"&& mv github_repo/data/utils utils"
                          r"&& mv github_repo/data/femnist femnist"
                          r"&& rm -rf github_repo")
                os.makedirs(self.raw_folder, exist_ok=False)
                os.system(rf"cd {root}"
                          r"&& rm -r femnist/data/rem_user_data femnist/data/sampled_data"
                          r"&& mv femnist/data/test raw/ && mv femnist/data/train raw/")
            else:
                print("Exiting...")
                exit()
        else:
            if os.path.exists(os.path.join(root, "data", "rem_user_data")):
                os.system(rf"rm -r {root}/data/rem_user_data")
            if os.path.exists(os.path.join(root, "data", "sampled_data")):
                os.system(rf"rm -r {root}/data/sampled_data")
            if os.path.exists(os.path.join(root, "data", "train")):
                os.system(rf"rm -r {root}/data/train")
            if os.path.exists(os.path.join(root, "data", "test")):
                os.system(rf"rm -r {root}/data/test")
            if os.path.exists(os.path.join(root, "raw")):
                os.system(rf"rm -r {root}/raw")
            if os.path.exists(os.path.join(root, "processed")):
                os.system(rf"rm -r {root}/processed")

            os.makedirs(self.raw_folder, exist_ok=False)
            os.system(rf"cd {root}/femnist"
                      r"&& ./preprocess.sh -s niid --sf 0.10 -k 0 -t sample"
                      r"&& cd .."
                      r"&& rm -r femnist/data/rem_user_data femnist/data/sampled_data"
                      r"&& mv femnist/data/test raw/ && mv femnist/data/train raw/")

        self.process()

    def process(self):
        print("Processing data...")

        if not os.path.isdir(self.processed_folder):
            os.makedirs(self.processed_folder)

        total_users_train = 0
        list_train_f = [f for f in os.listdir(os.path.join(self.raw_folder, "train")) if
                        fnmatch.fnmatch(f, "*.json")]
        # print('hello', list_train_f)

        list_train_f.sort(key=lambda fname: int(fname[9:-27]))

        # sys.exit()

        print(list_train_f)

        for filename in list_train_f:
            with open(os.path.join(self.raw_folder, "train", filename)) as file:
                data = json.load(file)
                for user_name, val in data["user_data"].items():
                    # key: user name
                    # val: dict {x: x_data, y: y_data}
                    x = torch.tensor(val["x"]).reshape((-1, 1,28,28))
                    y = torch.tensor(val["y"])

                    torch.save((x, y), os.path.join(self.processed_folder, "train_{}.pt".format(total_users_train)))
                    total_users_train += 1

        total_users_test = 0
        list_test_f = [f for f in os.listdir(os.path.join(self.raw_folder, "test")) if fnmatch.fnmatch(f, "*.json")]
        list_test_f.sort(key=lambda fname: int(fname[9:-26]))

        for filename in list_test_f:
            with open(os.path.join(self.raw_folder, "test", filename)) as file:
                data = json.load(file)
                for user_name, val in data["user_data"].items():
                    # key: user name
                    # val: dict {x: x_data, y: y_data}
                    x = torch.tensor(val["x"]).reshape((-1, 1,28,28))
                    y = torch.tensor(val["y"])

                    torch.save((x, y), os.path.join(self.processed_folder, "test_{}.pt").format(total_users_test))
                    total_users_test += 1

        assert total_users_train == total_users_test
        torch.save(total_users_train, os.path.join(self.processed_folder, "num_users.pt"))
        print("Done. {} users processed.".format(total_users_train))

    # def load(self, train):
    #     if train:
    #         prf = "train"
    #     else:
    #         prf = "test"

    #     data_list, label_list = [], []
    #     for user_id in self.user_list:
    #         x, y = torch.load(os.path.join(self.processed_folder, "{}_{}.pt".format(prf, user_id)))
    #         data_list.append(x)
    #         label_list.append(y)
    #     return torch.cat(data_list, dim=0), torch.cat(label_list, dim=0)
    
    def load(self, train):
        if train:
            prf = "train"
        else:
            prf = "test"
        x, y = torch.load(os.path.join(self.processed_folder, "{}_{}.pt".format(prf, self.client_id)))
        return x, y















class Gaussians(VisionDataset):
    """
    classes: 10 digits, 26 lower cases, 26 upper cases.
    We use torch.save, torch.load in this dataset
    """

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data
    def __init__(self, root, train=True, transform=None, nb_train = 300, nb_test = 50, nb_classes=2, ecart = 10):
        super(Gaussians, self).__init__(root, transform=transform)

        self.train = train

        self.ecart = ecart
        pi = np.pi

        if self.train:
            data = []
            targets = []
            for i in range(nb_classes):
                x = torch.randn(nb_train,2) 
                x[:,0] = x[:,0] + np.cos((2*pi/nb_classes)*i)*ecart
                x[:,1] = x[:,1] + np.sin((2*pi/nb_classes)*i)*ecart
                y = torch.zeros(nb_train) + i
                data.append(x)
                targets.append(y)
            self.data = torch.cat(data)
            self.targets = torch.cat(targets)
        else:
            data = []
            targets = []
            for i in range(nb_classes):
                x = torch.randn(nb_test,2) 
                x[:,0] = x[:,0] + np.cos((2*pi/nb_classes)*i)*ecart
                x[:,1] = x[:,1] + np.sin((2*pi/nb_classes)*i)*ecart
                y = torch.zeros(nb_test) + i
                data.append(x)
                targets.append(y)
            self.data = torch.cat(data)
            self.targets = torch.cat(targets)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # Needs 0~255, uint8 scale
        # img = Image.fromarray(np.uint8(255 * (1 - img.numpy())), mode='L')

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

