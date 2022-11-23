import os
import fastai
from fastai.vision.all import *



class PredictCNN:

    def __init__(self):
        self.basepath='/home/hoku/Desktop/time_series_to_image/'
        self.modelpath='/home/hoku/Desktop/time_series_to_image/train_val_test/model'


    def predict_CNN_pos(self):
        learn = load_learner(os.path.join(self.basepath,'model','model.pkl'), cpu=False)
        pos_files= get_image_files(os.path.join(self.basepath,'train_val_test','test','1'))

        correct=0; correct_high_prob=0; total=0; total_high_prob=0; correct_percent=0; correct_percent_high_prob=0
        for file in pos_files:
            total+=1
            output=learn.predict(file)
            class_prediction=int(output[0])
            prob_1=output[2][1].numpy()

            if class_prediction==1:
                correct+=1
                correct_percent=correct/total

            if prob_1 > 0.95:
                print(output)
                total_high_prob += 1
                if class_prediction == 1:
                    correct_high_prob += 1
                    correct_percent_high_prob = correct_high_prob / total_high_prob

            print('correct percent after {} is {}'.format(total,correct_percent))
            print('correct percent after {} high probability is {}'.format(total_high_prob,correct_percent_high_prob))


    def predict_CNN_neg(self):
        learn = load_learner(os.path.join(self.basepath,'model','model.pkl'), cpu=False)
        neg_files=get_image_files(os.path.join(self.basepath,'train_val_test','test','0'))

        correct=0; correct_high_prob=0; total=0; total_high_prob=0; correct_percent=0; correct_percent_high_prob=0
        for file in neg_files:
            total+=1
            output=learn.predict(file)
            class_prediction=int(output[0])
            prob_1=output[2][1].numpy()

            if class_prediction==0:
                correct+=1
                correct_percent=correct/total

            if prob_1 < 0.05:
                print(output)
                total_high_prob += 1
                if class_prediction == 0:
                    correct_high_prob += 1
                    correct_percent_high_prob = correct_high_prob / total_high_prob


            print('correct percent after {} is {}'.format(total,correct_percent))
            print('correct percent after {} high probability is {}'.format(total_high_prob,correct_percent_high_prob))




if __name__=='__main__':
    c=PredictCNN()
    c.predict_CNN_neg()
