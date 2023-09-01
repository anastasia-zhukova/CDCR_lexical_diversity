import random
import numpy as np

from torch.utils.data import BatchSampler,Subset

def create_bins(bin_size, maxlen):
    return [min(i + bin_size, maxlen) for i in range(0, maxlen, bin_size)]


def search_bin(bins, size):
    idx = len(bins) - 1
    for i, bin in enumerate(bins):
        if size <= bin:
            idx = i
            break
    return idx


class MultiTaskBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        mix_opt=1,
        extra_task_ratio=1,
        current_epoch=1,
        total_epochs=5,
        bin_size=64,
        bin_on=False,
        bin_grow_ratio=0.5,
        sampling = "sequential"
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        self.bin_size = bin_size
        self.bin_on = bin_on
        self.bin_grow_ratio = bin_grow_ratio
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        self.sampling = sampling
        self.datasets = self.create_subsets()


        self._train_data_list = self.create_train_data_list()

    def create_subsets(self):

        dataset_names = []

        for s in self._dataset.instance_ids:
            parts = s.split("[SEP]", 1)
            sub_parts = parts[0].split("/")
            dataset_names.append(sub_parts[0])

        unique_datasets = set(dataset_names)

        subsets_list = []

        for value in unique_datasets:
            indices = [idx for idx, instance in enumerate(dataset_names) if instance == value]
            subsets_list.append( Subset(self._dataset, indices))

        return subsets_list

    def create_train_data_list(self):
        train_data_list = []

        if self.sampling=="simulteneous":
            #List of batches randomly sampled from all datasets
            train_data_list.append(self._get_shuffled_index_batches(len(self._dataset), self._batch_size))
        else:
            for dataset in self.datasets:
                # if bin_on:
                #     train_data_list.append(
                #         self._get_shuffled_index_batches_bin(
                #             dataset,
                #             batch_size,
                #             bin_size=bin_size,
                #             bin_grow_ratio=bin_grow_ratio,
                #         )
                #     )
                # else:
                #Batches of one dataset only
                train_data_list.append(
                        self._get_shuffled_index_batches(len(dataset), self._batch_size))
                
        return train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        random.shuffle(index_batches)
        return index_batches

    @staticmethod
    def _get_shuffled_index_batches_bin(dataset, batch_size, bin_size, bin_grow_ratio):
        maxlen = dataset.maxlen
        bins = create_bins(bin_size, maxlen)
        data = [[] for i in range(0, len(bins))]

        for idx, sample in enumerate(dataset):
            bin_idx = search_bin(bins, len(sample["sample"]["token_id"]))
            data[bin_idx].append(idx)
        index_batches = []

        for idx, sub_data in enumerate(data):
            if len(sub_data) < 1:
                continue
            batch_size = 1 if batch_size < 1 else batch_size
            sub_dataset_len = len(sub_data)
            sub_batches = [
                list(range(i, min(i + batch_size, sub_dataset_len)))
                for i in range(0, sub_dataset_len, batch_size)
            ]
            index_batches.extend(sub_batches)
            batch_size = int(batch_size * bin_grow_ratio)
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        #Conditional to sampling type
        # all_indices = self._gen_task_indices(
        #     self._train_data_list, self._mix_opt, self._extra_task_ratio
        # )

        if self.sampling in ['sequential',"simultaneous"]:
            all_indices = self._gen_task_indices(
                self._train_data_list
            )
        elif self.sampling=='annealed':
            all_indices = self._gen_task_indices_annealed(
                self._train_data_list,self.current_epoch,
            self.total_epochs
            )
        else:
            raise ValueError("Invalid value for 'sampling' parameter. "
                         "Supported values are 'sequential', 'simultaneous' and 'annealed'.")
    
        # Loop until there are no more batches to generate
        while all_indices:
            local_task_idx = all_indices.pop(0)
            
            try:
                batch = next(all_iters[local_task_idx])
                yield_list = []

                for sample_id in batch:
                    instance, label, instance_id, task = self._dataset[sample_id]
                    yield_list.append((instance, label, instance_id, task))

                yield yield_list
            except StopIteration:
                # The current task has no more batches,stop the whole training
                # Could result in lower performance, we must see.
                break

    @staticmethod
    def _gen_task_indices_when_main_task(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(
                min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices))
            )
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices

    @staticmethod
    def _gen_task_indices(train_data_list):
        num_tasks = len(train_data_list)
        all_indices = []
        
        # Create a list of task indices
        for i in range(num_tasks):
            all_indices += [i] * len(train_data_list[i])
        
        # Shuffle the list of task indices
        random.shuffle(all_indices)
        
        return all_indices
    
    def _gen_task_indices_annealed(self, train_data_list, current_epoch, total_epochs):
        num_tasks = len(train_data_list)
        
        # Calculate alpha based on the provided formula
        alpha = 1 - 0.8 * ((current_epoch - 1) / (total_epochs - 1))
        
        # Calculate the normalized probabilities based on dataset sizes raised to the power of alpha
        dataset_sizes = [len(dataset) for dataset in train_data_list]
        probs = np.array(dataset_sizes) ** alpha
        probs /= np.sum(probs)
                # Generate task indices based on probabilities
        task_indices = np.random.choice(range(num_tasks), size=self.__len__(), p=probs)
        
        return task_indices.tolist()
    