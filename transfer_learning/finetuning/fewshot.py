# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utility to find k-shot dataset indices, outputs the indices on stdout."""
#!/usr/bin/env python3
# coding: utf-8

from collections import *
from functools import *
import random
import sys
import numpy as np
import torch
import torchvision as tv


class AddIndexIter(torch.utils.data.dataloader._SingleProcessDataLoaderIter):
  def _next_data(self):
    index = self._next_index()  # may raise StopIteration
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    if self._pin_memory:
      data = torch.utils.data._utils.pin_memory.pin_memory(data)
    return index, data


def find_indices_loader(loader, n_shots, n_classes):
  per_label_indices = defaultdict(partial(deque, maxlen=n_shots))

  for ibatch, (indices, (images, labels)) in enumerate(AddIndexIter(loader)):
    for idx, lbl in zip(indices, labels):
      per_label_indices[lbl.item()].append(idx)
  
      findings = sum(map(len, per_label_indices.values()))
      if findings == n_shots * n_classes:
        return per_label_indices
  raise RuntimeError("Unable to find enough examples!")


class ShuffledDataset:

    def __init__(self, dataset, random_state=None):
        self.dataset = dataset
        rng = np.random.RandomState(random_state)
        inds = np.arange(len(dataset))
        rng.shuffle(inds)
        self.inds = inds

    def __getitem__(self, i):
        return self.dataset[self.inds[i]]
    
    def __len__(self):
        return len(self.inds)

def find_fewshot_indices(dataset, n_shots, random_state=None):
  n_classes = len(dataset.classes)

  orig_transform = dataset.transform
  dataset.transform = tv.transforms.Compose([
      tv.transforms.CenterCrop(1),
      tv.transforms.ToTensor()
  ])

  # TODO(lbeyer): if dataset isinstance DatasetFolder, we can (maybe?) do much better!
  if random_state is not None:
      rng = torch.Generator()
      rng.manual_seed(random_state)
  else:
      rng = None
  loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0, generator=rng)
  per_label_indices = find_indices_loader(loader, n_shots, n_classes)
  all_indices = [i for indices in per_label_indices.values() for i in indices]
  if random_state is None:
      random.shuffle(all_indices)
  else:
      rng = np.random.RandomState(random_state)
      rng.shuffle(all_indices)
  dataset.transform = orig_transform
  print(all_indices)
  return all_indices

if __name__ == "__main__":
  dataset = tv.datasets.ImageFolder(sys.argv[2], preprocess)
  all_indices = find_fewshot_indices(dataset, int(sys.argv[1]))
  for i in all_indices:
      print(i)
