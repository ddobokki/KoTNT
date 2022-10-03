import os
from typing import List
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm


def get_dataset_paths(dataset_dir: os.PathLike, train_type: str) -> List[os.PathLike]:
    """_summary_
    arrow파일이 저장된 datasets의 dir에 0 ~ n번 까지의 arrow데이터를 불러오는 함수 입니다.

    _dir_structure_
    .
    └── Arrow_data_dir/
        ├── sharded_train/
        │   ├── arrow_01/
        │   │   └── arrow_file
        │   ├── arrow_02/
        │   │   └── arrow_file
        │   └── ...
        ├── sharded_eval
        └── sharded_dev

    Args:
        dataset_dir (os.PathLike): arrow데이터가 저장되어 있는 경로를 전달받습니다.
        train_type (str): 불러올 데이터의 폴더 이름을 지정합니다. 이때 폴더 이름은 train, dev, eval과 같은 이름을 사용합니다.

    Returns:
        List[os.PathLike]: train_type아래에 저장되어 있는 arrow파일을 list로 반환합니다.
        ex: [kspon/train/0, kspon/train/1 ...]
    """

    dataset_dir = os.path.join(dataset_dir, train_type)
    sharding_dataset_paths = [os.path.join(dataset_dir, shard_num) for shard_num in os.listdir(dataset_dir)]
    sharding_dataset_paths.sort()
    return sharding_dataset_paths


def get_concat_dataset(dataset_dirs: List[os.PathLike], train_type: str) -> Dataset:
    """_summary_
    dir에 있는 분할된 arrow파일을 불러온 다음 하나로 합친뒤 반환하는 함수 입니다.

    _dir_structure_
    .
    └── Arrow_data_dir/
        ├── sharded_train/
        │   ├── arrow_01/
        │   │   └── arrow_file
        │   ├── arrow_02/
        │   │   └── arrow_file
        │   └── ...
        ├── sharded_eval
        └── sharded_dev

    Args:
        dataset_dir (os.PathLike): arrow데이터가 저장되어 있는 경로를 전달받습니다.
        train_type (str): 불러올 데이터의 폴더 이름을 지정합니다. 이때 폴더 이름은 train, dev, eval과 같은 이름을 사용합니다.

    Returns:
        Dataset: HuggingFace Dataset를 반환합니다.
    """
    dataset_lists = []
    for dataset_dir in dataset_dirs:
        sharding_dataset_paths = get_dataset_paths(dataset_dir, train_type)
        sharding_datasets = [load_from_disk(p) for p in tqdm(sharding_dataset_paths)]
        dataset_lists.extend(sharding_datasets)
    concat_dataset = concatenate_datasets(dataset_lists)
    concat_dataset.set_format("torch")
    return concat_dataset
