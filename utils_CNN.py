import os
from PIL import Image
import random
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

imagepath='/home/hoku/Desktop/time_series_to_image/images'
cnntrainpath='/home/hoku/Desktop/time_series_to_image/images_cnn'
train_val_test='/home/hoku/Desktop/time_series_to_image/train_val_test'


def resize_recurse(path=''):
    ''' recursively looks through all files and resizes images
    :param (str): root path with images in subdirectories
    '''

    for file in sorted(os.listdir(path)):
        if file.endswith('.jpeg'):
            img=Image.open(os.path.join(path,file))
            img_new=img.resize((224,224))
            img_new.save(os.path.join(path, file))
        else:
            resize_recurse(path=os.path.join(path,file))


def balance_classes(p2img='',cnnpath=''):
    ''' with the understanding that for some dates there is significant class imbalance,
    this function resamples the minority class and copies the images to new directory
    *** note - this step is inefficient and should be replaced by resampling ***
    :p2img (str) - path to data saved as dates/class folder structure
    :cnnpath (str) - path to save the balanced data dates/class folder structure
    '''

    for date in sorted(os.listdir(p2img)):
        print(date)
        if len(os.listdir(os.path.join(p2img,date,'0')))>len(os.listdir(os.path.join(p2img,date,'1'))):
            paths_min=[os.path.join(p2img,date,'1',file) for file in os.listdir(os.path.join(p2img,date,'1'))]
            paths_maj = [os.path.join(p2img, date,'0',file) for file in os.listdir(os.path.join(p2img, date, '0'))]
            num_maj=len(paths_min)
            paths_maj_resampled=random.sample(paths_maj,num_maj)
            paths_dict={'0':paths_min,'1':paths_maj_resampled}

            #make saving structure
            if not os.path.exists(os.path.join(cnnpath,date)):
                os.mkdir(os.path.join(cnnpath,date))
            if not os.path.exists(os.path.join(cnnpath,date,'1')):
                os.mkdir(os.path.join(cnnpath,date,'1'))
            if not os.path.exists(os.path.join(cnnpath,date,'0')):
                os.mkdir(os.path.join(cnnpath,date,'0'))

            for outcome in paths_dict.keys():
                paths=paths_dict[outcome]
                for file in paths:
                    filename=file.split('/')[-1]
                    img=Image.open(file)
                    img.save(os.path.join(cnnpath,date,outcome,filename))


def make_train_val_test_ds(inpath='',tvt_path=''):
    ''' takes data from a balanced data and transferrs them into standard train/val/test folders structure
     :param inpath (str) - path to the data in balanced folders with date/class
     :param tvt_path (str) - path to new data to save in train/val/test folder structure

     '''

    date_list=sorted(os.listdir(inpath))
    test_list=date_list[-10:]
    date_list=date_list[:-10]
    len_data=len(date_list)
    train_list=random.sample(date_list,int(len_data*0.7))
    val_list=list(set(date_list).symmetric_difference(set(train_list)))
    val_dict={'train':train_list,'val':val_list,'test':test_list}

    for val in ['train','val','test']:
        if not os.path.exists(os.path.join(tvt_path,val)):
            os.mkdir(os.path.join(tvt_path,val))
            os.mkdir(os.path.join(tvt_path,val,'1'))
            os.mkdir(os.path.join(tvt_path, val,'0'))

    for item in val_dict.keys():
        print(item)
        dlist=val_dict[item]
        for date in dlist:
            intial_paths_pos=[os.path.join(inpath,date,'1',file) for file in os.listdir(os.path.join(inpath,date,'1'))]
            initial_paths_neg=[os.path.join(inpath,date,'0',file) for file in os.listdir(os.path.join(inpath,date,'0'))]

            [shutil.copy2(os.path.join(inpath,date,'1',filename.split('/')[-1]),os.path.join(tvt_path,item,'1',date+'_'+filename.split('/')[-1])) for filename in intial_paths_pos]
            [shutil.copy2(os.path.join(inpath,date,'0',filename.split('/')[-1]),os.path.join(tvt_path,item,'0',date+'_'+filename.split('/')[-1])) for filename in initial_paths_neg]



if __name__=='__main__':
    #balance_classes(p2img=imagepath,cnnpath=cnntrainpath)
    #make_train_val_test_ds(inpath=cnntrainpath,tvt_path=train_val_test)
    resize_recurse(path=train_val_test)