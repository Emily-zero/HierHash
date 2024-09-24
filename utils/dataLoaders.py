import pandas as pd
from torch.utils.data import DataLoader
from text_Data.dataset import  dataset_NYT, dataset_BERT


class Data():
    def __init__(self, hparams):
        self.hparams = hparams
        self.file_path = hparams.data_path
        self.dataset_Name = hparams.dataset
        self.max_length = hparams.max_length
        self.dataset_List=['WOS','20News','nyt','yelp','arxiv','BERT']
        self.batch_size=hparams.batch_size
        self.num_workers=hparams.workers
        if self.dataset_Name not in self.dataset_List:
            raise ValueError


    def load_datasets(self):
        raise NotImplementedError

    def get_loaders(self,get_test=True):

        if self.dataset_Name == 'BERT':
            # print("{self.file_path}/train.csv")
            train_loader = get_dataloader_bert(f"{self.file_path}/trainbertData.csv", self.batch_size, num_workers=self.num_workers,
                                                       shuffle=True, get_one_bert=False)
            database_loader = get_dataloader_bert(f"{self.file_path}/trainbertData.csv", self.batch_size,
                                              num_workers=self.num_workers,
                                              shuffle=False)
            val_loader = get_dataloader_bert(f"{self.file_path}/valbertData.csv", self.batch_size,
                                              num_workers=self.num_workers,
                                              shuffle=True)
            test_loader = get_dataloader_bert(f"{self.file_path}/testbertData.csv", self.batch_size,
                                              num_workers=self.num_workers,
                                              shuffle=True)
        else :
            train_loader = get_dataloader_NYT(f"{self.file_path}/train.csv", self.batch_size, num_workers=self.num_workers,
                                                       shuffle=True, max_length=self.max_length)
            database_loader = get_dataloader_NYT(f"{self.file_path}/train.csv", self.batch_size,
                                              num_workers=self.num_workers,
                                              shuffle=True, max_length=self.max_length)
            val_loader = get_dataloader_NYT(f"{self.file_path}/val.csv", self.batch_size,
                                              num_workers=self.num_workers,
                                              shuffle=True, max_length=self.max_length)
            test_loader = get_dataloader_NYT(f"{self.file_path}/test.csv", self.batch_size,
                                              num_workers=self.num_workers,
                                              shuffle=True, max_length=self.max_length)

        return train_loader, database_loader, val_loader, test_loader




def get_dataloader_NYT(DataDir='../baselines/data/nyt_c2f',batch_size=16, num_workers=2, shuffle=True,max_length=200):
    nyt = dataset_NYT(DATA_DIR=DataDir,max_length=max_length)
    #print(type(nyt))
    loader = DataLoader(nyt,shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,drop_last=False)
    return loader

def get_dataloader_bert(DataDir='../baselines/data/nyt_c2f/val.csvbertData.csv',batch_size=16, num_workers=2, shuffle=True,get_one_bert = True):
    nyt = dataset_BERT(DATA_DIR=DataDir, get_one_bert=get_one_bert)
    # print(type(nyt))
    loader = DataLoader(nyt, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=False)
    return loader

