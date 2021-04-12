import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0")


class custom_convlstm_cell(nn.Module):

    def __init__(self,input_channels,hidden_channels,kernel_size):
        """
        input_channels = number of input channels
        hidden_channels = number of hidden channels
        """
        super(custom_convlstm_cell, self).__init__()    
        
        self.in_channel = input_channels
        self.out_channel = hidden_channels
        self.kernel_size = kernel_size

        self.conv_x = nn.Conv2d(input_channels,hidden_channels*4, kernel_size=self.kernel_size,padding=1)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels*4, kernel_size=self.kernel_size, padding=1, bias=False)


    def forward(self, x_t, hidden):
        #x_t should be of shape batch_size,input_channels,d,d
        #hidden should be 2-tuple with each element of shape batch_size,hidden_channels,d,d
        (h_t,c_t) = hidden

        gates = self.conv_x(x_t) + self.conv_h(h_t) #shape of gates = batch_size,hidden_channels*4,d,d
        i,f,g,o = torch.chunk(gates,4,1)
        
        c_t = f*c_t + i*g        
        h_t = o*torch.tanh(c_t)       

        return h_t,c_t

class STAE(nn.Module):
    
    def __init__(self,input_channels):

        super(STAE, self).__init__()

        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(self.input_channels,64,kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64,32,kernel_size=5,stride=2)

        self.convlstm1 = custom_convlstm_cell(32,16,3)
        self.convlstm2 = custom_convlstm_cell(16,16,3)
        self.convlstm3 = custom_convlstm_cell(16,32,3)

        self.tconv1 = nn.ConvTranspose2d(32,64,kernel_size=5,stride=2)
        self.upsamp = nn.Upsample(scale_factor = 2, mode='nearest')
        self.tconv2 = nn.ConvTranspose2d(64,1,kernel_size=3)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, inp, hidden=None):

        #input_shape is assumed to be batch_size,len_seq,1,row,cols
        batch_size,seq_len,input_channel,h,w = inp.shape
        
        x = torch.reshape(inp,[batch_size*seq_len,input_channel,h,w])

        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.reshape(batch_size,seq_len,32,14,14)
        
        if hidden==None:

            h_t1,c_t1 = self.init_hidden_states(batch_size,16,14,14)
            h_t2,c_t2 = self.init_hidden_states(batch_size,16,14,14)
            h_t3,c_t3 = self.init_hidden_states(batch_size,32,14,14)

        output_seq = []

        for t in range(seq_len):
            x_t = x[:,t]
            h_t1,c_t1 = self.convlstm1(x_t,(h_t1,c_t1))
            h_t2,c_t2 = self.convlstm2(h_t1,(h_t2,c_t2))
            h_t3,c_t3 = self.convlstm3(h_t2,(h_t3,c_t3))

            output_seq.append(h_t3)
        x = torch.stack(output_seq, 1) #output is should be batch_size,len_seq,64,row,cols
        x = torch.reshape(x,[batch_size*seq_len,32,14,14])
        

        x = torch.relu(self.tconv1(x))
        x = self.upsamp(x)
        x = torch.sigmoid(self.tconv2(x))
        
        output_seq = torch.reshape(x,[batch_size,seq_len,input_channel,h,w])
        return output_seq
    
    def init_hidden_states(self,batch_size,hidden,h,w):
        
        h_t = torch.zeros(batch_size,hidden,h,w).to(device)
        c_t = torch.zeros(batch_size,hidden,h,w).to(device)
        
        return (h_t,c_t)
        