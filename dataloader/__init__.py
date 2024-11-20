from .temp_dataloader import TempDataset

def load_dataset(dataset:str="temp",
                 data_dir:str='data/AIS.v1i.yolov8',
                 mode:str='train',
                 encoder:str='ResNet'):
    
    if dataset == 'temp':
        return TempDataset(data_dir=data_dir,
                           mode=mode,
                           encoder=encoder)