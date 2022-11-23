import os


import fastai
from fastai.vision.all import *



class TrainMultidayCNN:

    def __init__(self):
        self.basepath='/home/hoku/Desktop/time_series_to_image/train_val_test'
        self.modelpath='/home/hoku/Desktop/time_series_to_image/train_val_test/model'


    def train_CNN(self):
        dls=ImageDataLoaders.from_folder(self.basepath, train='train', valid='val',item_tfms=Resize(224))
        learn = vision_learner(dls, resnet34, metrics=error_rate)
        learn.fine_tune(15)
        learn.export(os.path.join(self.basepath,'model','model.pkl'))

    def predict(self):
        pass





if __name__=='__main__':
    c=TrainMultidayCNN()
    c.train_CNN()

