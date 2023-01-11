import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler


class HARDataset(Dataset):
    def __init__(self, data_dir: str, rec_names: list[str], window_size: int = 60):
        super(HARDataset, self).__init__()
        
        df_list = []
        
        # since all data fit into the RAM, we can load them to create global indexes
        for rec_name in rec_names:
            loader = np.load(os.path.join(data_dir, rec_name), allow_pickle=True)
            X = loader['X']
            index = loader['index']
            self.columns = loader['columns']
            y = loader['y']
            df = pd.DataFrame(data=X, index=index, columns=self.columns)
            df['target'] = y
            df = df.iloc[:(df.shape[0]//window_size)*window_size]
            df_list.append(df)

        self.joint_df = pd.concat(df_list)
        self.joint_df = self.joint_df.reset_index(drop=False).reset_index(drop=False)\
                                     .rename(columns={'index': 'time', 'level_0': 'sample_id'})
        self.joint_df['window_id'] = self.joint_df['sample_id'].apply(lambda x: x//window_size)
        
        # window label - last element label
        self.window_target = []
        for i in self.joint_df['window_id'].unique():
            slice_df = self.joint_df[self.joint_df['window_id']==i]
            self.window_target.append(slice_df.iloc[-1]['target'])
        
    def __len__(self) -> int:
        return int(self.joint_df['window_id'].max()) + 1

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.joint_df[self.joint_df['window_id']==idx][self.columns].values)
        y = torch.tensor(self.joint_df[self.joint_df['window_id']==idx]['target'].values[-1])
        return x.float(), y.float().reshape(-1)
    
    def get_labels(self):
        """For imbalance sampler."""
        return self.window_target
    
    def get_origin_df(self):
        """For visualization"""
        return self.joint_df
    

def get_loaders(data_dir: str, rec_train: list[str], rec_val: list[str], rec_test: list[str], 
                batch_size: int, window_size: int):
    train_dataset = HARDataset(data_dir=data_dir, rec_names=rec_train, window_size=window_size)
    train_loader = DataLoader(train_dataset, batch_size, drop_last=True, 
                              sampler=ImbalancedDatasetSampler(train_dataset))

    val_dataset = HARDataset(data_dir=data_dir, rec_names=rec_val, window_size=window_size)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, drop_last=True)

    test_dataset = HARDataset(data_dir=data_dir, rec_names=rec_test, window_size=window_size)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader