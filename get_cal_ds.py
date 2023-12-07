from torch.utils.data import DataLoader
from utils import *

filename = './export_quantization/data/cal_ds.npy'
sample_name = './export_quantization/data/input_sample.npy'

if __name__ == '__main__':
    cal_data = [x.numpy() for x, _ in DataLoader(get_dataset())]
    sample_data = cal_data[0]
    cal_data = np.concatenate(cal_data)
    
    np.save(filename, np.array(cal_data))
    np.save(sample_name, np.array(sample_data))
    
    x = np.load(filename, allow_pickle=True)
    print(x.shape)
    x = np.load(sample_name, allow_pickle=True)
    print(x.shape)