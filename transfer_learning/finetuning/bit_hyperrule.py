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

def get_resolution(original_resolution):
  """Takes (H,W) and returns (precrop, crop)."""
  area = original_resolution[0] * original_resolution[1]
  return (160, 128) if area < 96*96 else (512, 480)


known_dataset_sizes = {
  'cifar10': (32, 32),
  'cifar100': (32, 32),
  'oxford_iiit_pet': (224, 224),
  'oxford_flowers102': (224, 224),
  'imagenet2012': (224, 224),
}


def get_resolution_from_dataset(dataset):
  # if dataset not in known_dataset_sizes:
    # raise ValueError(f"Unsupported dataset {dataset}. Add your own here :)")
  return get_resolution(known_dataset_sizes.get(dataset, (224, 224)))


def get_mixup(dataset_size):
  return 0.0 if dataset_size < 20_000 else 0.1


def get_max_train_set(steps):
    assert steps in (500, 10000, 20000)
    if steps == 500:
        return 19999
    elif steps == 10000:
        return 499999
    elif steps == 20000:
        return float('inf')

def get_schedule(dataset_size):
  if dataset_size < 20_000:
    return [100, 200, 300, 400, 500]
  elif dataset_size < 500_000:
    return [500, 3000, 6000, 9000, 10_000]
  else:
    return [500, 6000, 12_000, 18_000, 20_000]


def get_lr(step, dataset_size, base_lr=0.003):
  """Returns learning-rate for `step` or None at the end."""
  supports = get_schedule(dataset_size)
  # Linear warmup
  if step < supports[0]:
    return base_lr * step / supports[0]
  # End of training
  elif step >= supports[-1]:
    return None
  # Staircase decays by factor of 10
  else:
    for s in supports[1:]:
      if s < step:
        base_lr /= 10
    return base_lr

def get_final_lr(dataset_size, base_lr=0.003):
  supports = get_schedule(dataset_size)
  last_step = supports[-1] - 1
  return get_lr(last_step, dataset_size, base_lr=base_lr)
