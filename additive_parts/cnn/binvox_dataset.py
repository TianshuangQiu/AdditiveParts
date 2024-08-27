import os
import pandas as pd
from torch.utils.data import Dataset
import multiprocess as multiprocessing
import json
import re
import os
import binvox


def get_binvox_name_256(file_name):
    return (
        file_name.replace("rotated_files", "Binvox_files_default_res")[:-3] + "binvox"
    )


def get_binvox_name_64(file_name):
    file_name = file_name.replace("rotated_files", "Binvox_files_64_res_compressed")
    name = re.search(".{8}-.{4}-.{4}-.{4}-.{12}", file_name)[0]
    file_name = file_name.replace(name, name + "_compressed")
    file_name = file_name[:-3] + "binvox"
    return file_name


def load_sample_from_disk(file_path):
    return (file_path, binvox.read_as_3d_array(open(file_path, "rb")).data)


def load_sample_from_savio(file_path):
    binvox_name = (
        file_path.replace("rotated_files", "Binvox_files_default_res")[:-4] + ".binvox"
    )
    sample = binvox.read_as_3d_array(open(binvox_name, "rb")).data
    return (file_path, sample)


class BinvoxDataset(Dataset):
    def __init__(
        self, data_path, label_file_path, transform=None, ram_limit=1000, resolution=256
    ):
        self.data_path = data_path
        self.label_file_path = label_file_path
        self.transform = transform
        self.ram_limit = ram_limit
        self.resolution = resolution
        self.labels = self.load_labels()

    def load_labels(self):
        j = json.load(open(os.path.join(self.data_path, self.label_file_path)))
        return pd.Series(data=j) / 100

    def __len__(self):
        return self.labels.size

    def load_sample_into_ram(self, idx):
        num_cores = multiprocessing.cpu_count()
        samples_in_ram = {}
        # need to load ram_limit # of samples, starting from idx
        print("Loading samples", idx)
        labels_to_load = self.labels[
            idx : min(idx + self.ram_limit, self.__len__())
        ].index.to_list()
        paths_to_load = [
            os.path.join(self.data_path, label) for label in labels_to_load
        ]
        pool = multiprocessing.Pool(processes=num_cores)
        tuples = pool.map(load_sample_from_savio, paths_to_load)
        pool.close()
        pool.join()
        for sample in tuples:
            samples_in_ram[sample[0]] = sample[1]
        return samples_in_ram

    def load_sample_from_ram(self, file_path):
        return self.samples_in_ram[file_path]

    def __getitem__(self, idx):
        # Multi core
        """
        if idx % self.ram_limit == 0:
            # clear existing samples in ram
            self.samples_in_ram = None
            # need to load new samples into ram
            self.samples_in_ram = self.load_sample_into_ram(idx)
        sample = self.load_sample_from_ram(os.path.join(self.data_path, self.labels.index[idx]))
        if self.transform:
            sample = self.transform(sample)
        if idx % 1000 == 0:
            print('Processing sample number', idx)
        return sample, self.labels.iloc[idx]
        """
        # Single core
        if self.resolution == 256:
            binvox_name = get_binvox_name_256(self.labels.index[idx])
        if self.resolution == 64:
            binvox_name = get_binvox_name_64(self.labels.index[idx])
        sample_path = os.path.join(self.data_path, binvox_name)
        sample = binvox.read_as_3d_array(open(sample_path, "rb")).data
        if self.transform:
            sample = self.transform(sample)
        # if idx % 1000 == 0:
        #     print('Processing sample number', idx)
        return sample, self.labels.iloc[idx]
