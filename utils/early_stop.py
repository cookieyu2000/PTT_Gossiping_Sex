# utils.early_stop.early_stop
import torch
import numpy as np
import os,shutil

class early_stop:
    def __init__(self,
                 save_path,
                 weight_name,
                 monitor='train_loss',
                 mode='min',
                 patience=10,
                 verbose=False,
                 delta=0):
        '''
        @Input\n
        `monitor`:監控的指標 ex: val_loss 、 val_acc 、 val_iou ...
        `mode`:監控指標min or max
        `save_path`:model儲存路徑
        `patience`:容忍多少次epoch的validation loss持續上升
        `verbose`:印出每個validdation的loss
        `delta`:
        '''
        self.monitor = monitor
        self.save_path = save_path
        self.weight_name = weight_name
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.monitor_tmp = np.inf if self.mode == 'min' else 0
        
        # try:
        #     shutil.rmtree(self.save_path)
        # except:
        #     pass
        # os.makedirs(self.save_path)
        
    def __call__(self,
                 perform_matrix,
                 model,
                 performance_value):
        #判斷 傳入的評估指標 是要越大越好(acc)還是越小越好(loss)
        if self.mode == 'max':
            score = perform_matrix  
            self.monitor_tmp = 0
            #儲存模型
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(perform_matrix, model, performance_value)
            elif score < self.best_score + self.delta:
                
                self.counter += 1
                print(f'Early Stop counter:{self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(perform_matrix, model, performance_value)
                self.counter=0

        elif self.mode == 'min':
            score = perform_matrix
            self.monitor_tmp = np.Inf
        
            #儲存模型
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(perform_matrix, model, performance_value)
            elif score > self.best_score + self.delta:
                self.counter += 1
                print(f'Early Stop counter:{self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(perform_matrix, model, performance_value)
                self.counter=0

    def save_checkpoint(self,
                        perform_matrix,
                        model,
                        performance_value
                        ):
        if self.verbose:
            if self.mode=='min':
                print(f'{self.monitor} increased ({self.monitor_tmp:.6f} --> {perform_matrix:.6f}). Saving model...')
            elif self.mode=='max':
                print(f'{self.monitor} decreased ({self.monitor_tmp:.6f} --> {perform_matrix:.6f}). Saving model...')
            
        path = f'{self.save_path}/{self.weight_name}'
        if 'acc' in self.monitor :
            if perform_matrix >= 0.0000001:
                torch.save(model.state_dict(),path)
                self.monitor_tmp = perform_matrix
                print('model SAVE!')
        else:
            torch.save(model.state_dict(),path)
            self.monitor_tmp = perform_matrix
            print('model SAVE!')
