import os
from torch.utils.data import DataLoader
from data.dataset import SkeletonDataset
from config.config import config

def prepare_dataset():
    files = [f for f in os.listdir(config['DATA_DIR']) if f.endswith('.mat')]

    person_files = {}
    for file in files:
        try:
            person_num = int(file.split('P')[1].split('_')[0])
            if person_num not in person_files:
                person_files[person_num] = []
            person_files[person_num].append(file)
        except Exception as e:
            print(f"Lỗi xử lý file {file}: {e}")
            continue

    train_files = []
    test_files = []

    for person, person_data in person_files.items():
        if person % 3 == 0:
            test_files.extend(person_data)
        else:
            train_files.extend(person_data)

    print(f"Dataset split - Train: {len(train_files)}, Test: {len(test_files)}")

    train_dataset = SkeletonDataset(train_files, config['DATA_DIR'], config['NUM_FRAMES'], mode='train')
    test_dataset = SkeletonDataset(test_files, config['DATA_DIR'], config['NUM_FRAMES'], mode='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader