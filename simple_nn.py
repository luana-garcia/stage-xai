#usefull for preprocessing data, before using gems2
from tqdm import tqdm  
import torch
import torch.nn as nn

import sklearn as sk

import matplotlib.pyplot as plt
import numpy as np

import os

#1) define the NN classifier model
class MyNN(nn.Module):
    def __init__(self,p):
        super().__init__()  #p is the dimension of the inputs
        self.fc1 = nn.Linear(p, p)
        self.relu1 = nn.ReLU()
        self.dout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(p, p)
        self.relu2 = nn.ReLU()
        self.dout2 = nn.Dropout(0.05)
        self.fc3 = nn.Linear(p, p)
        self.relu3 = nn.ReLU()
        self.dout3 = nn.Dropout(0.05)
        self.fc4 = nn.Linear(p, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout1 = self.dout1(h1)
        a2 = self.fc2(dout1)
        h2 = self.relu2(a2)
        dout2 = self.dout2(h2)
        a3 = self.fc3(dout2)
        h3 = self.relu3(a3)
        dout3 = self.dout3(h3)
        a4 = self.fc4(dout3)
        y = self.out_act(a4)
        return y
        
    def predict(self,input_):
        with torch.no_grad():
            pred = self.forward(input_)
            return (pred > 0.5).float().clone().detach()
    
    def predict_proba(self,input_):
        with torch.no_grad():
            pred = self.forward(input_)
            return pred.clone().detach()

class SimpleNNclassifier:
    """
    Instanciate and train a simple NN classifier using a single command line. Although Pytorch is
    used in this class, all inputs and outputs are numpy arrays.
    
    * This class is initiated and trained with the inputs:
      -> A numpy array X of input observations
      -> A numpy array y of output observations with labels
    
    * As in sklearn, the predict method will used to predict y
    on new observations have the same structure as X.
    """
    
    def __init__(self,p):
        self.p = p
        print('SimpleNNclassifier created')
        
    #2) init method of SimpleNNclassifier
    def fit(self,X_train,y_train,epochs_nb=1000,batch_size=300,optimizer='SGD', save = False, state = ''):
        """
        parameters:
         - epochs_nb: epochs number
         - batch_size: batch size
         - optimizer: optimizer in ['SGD','ADAM']
        """
        #2.1) instantiate and parametrize the model
        X_trainS=sk.preprocessing.scale(X_train)
        
        n=X_trainS.shape[0]
        p=X_trainS.shape[1]

        #Initialize the model
        self.model = MyNN(self.p)
        #Define loss criterion
        #criterion = nn.BCELoss()
        criterion = nn.MSELoss()
        #Define the optimizer
        if optimizer=='SGD':
            optimizer = torch.optim.SGD(self.model.parameters(),lr=0.0001)
        else:
            optimizer = torch.optim.Adam(self.model.parameters())
        
        #2.3) train the model
        
        losses_train = []
        for i in tqdm(range(epochs_nb)):
            for beg_i in range(0, n-batch_size-1, batch_size):
                X = torch.from_numpy(X_trainS[beg_i:beg_i+batch_size,:].astype(np.float32))
                y = torch.from_numpy(y_train.reshape(-1,1)[beg_i:beg_i+batch_size,:].astype(np.float32))
                
                #Precit the output for Given input
                y_pred = self.model.forward(X)
                #Compute Cross entropy loss
                loss = criterion(y_pred,y)
                #Add loss to the list
                losses_train.append(loss.item())

                ###losses_women_train.append(loss_women.item())

                ###losses_men_train.append(loss_men.item())
                #Compute gradients
                loss.backward()
                #Adjust weights
                optimizer.step()
        plt.plot(losses_train)
        
        if save:
            dir = './plots'
            os.makedirs(dir, exist_ok=True)
            save_path = os.path.join(dir, f'loss_nn_{state}.png')
            plt.savefig(save_path)
        else:
            plt.show()
    
    #3) Prediction
    def predict(self,X_test):
        X_testS=sk.preprocessing.scale(X_test)
        X_test_torch = torch.from_numpy(X_testS.astype(np.float32))
        y_test_pred_torch=self.model.predict(X_test_torch)
        return y_test_pred_torch.numpy()

    def predict_proba(self,X_test):
        X_testS=sk.preprocessing.scale(X_test)
        X_test_torch = torch.from_numpy(X_testS.astype(np.float32))
        y_test_pred_torch=self.model.predict_proba(X_test_torch)
        return 1.-y_test_pred_torch.numpy()  #probability to be equal to 0 actually