import glob
import shutil
import os
import random
import math

if __name__ == '__main__':

    img_dir        = 'C://My_WorkDir//002//AE//img//'
    dataset_dir    = 'C://My_WorkDir//002//AE//data//'
    train_data_dir = dataset_dir + 'train//'
    val_data_dir   = dataset_dir + 'val//'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    if not os.path.isdir(train_data_dir):
        os.makedirs(train_data_dir, exist_ok=True)
        for ii in range(2,13):
            os.makedirs(train_data_dir+'%02d'%ii, exist_ok=True)
        files = glob.glob(img_dir + '*.png')
        random_sample_file = random.sample(files,math.ceil(len(files)*0.3))
        for ff in random_sample_file:
            for ii in range(2,13):
                if 'img_%02d_'%ii in ff:        
                    shutil.copy2(ff,train_data_dir+'%02d//'%ii)
            
        os.makedirs(val_data_dir, exist_ok=True)
        for ii in range(2,13):
            os.makedirs(val_data_dir+'%02d'%ii, exist_ok=True)
        cc = 1 
        for ff in files:
            if ff in random_sample_file:
                continue
            for ii in range(2,13):
                if 'img_%02d_'%ii in ff:        
                    cc += 1 
                    shutil.copy2(ff,val_data_dir+'%02d//'%ii)
                    if cc > math.ceil(len(files)*0.3):
                        break