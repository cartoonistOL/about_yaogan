#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install d2l
# !pip install pyhdf
# !apt update
# !apt install gcc g++ make
# !pip3 install numpy
# !apt-get install pkg-config -y
# !apt-get install libsqlite3-dev -y
# !apt-get install sqlite3 -y
# !pip install --upgrade numpy


# In[1]:


import os
# !pip list
import numpy as np
import pyhdf
from pyhdf.SD import SD
import matplotlib.pyplot as plt


# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# In[3]:


def readhdf(path):
    hdf = SD(path)
    # print(hdf.info())  # 信息类别数
    data = hdf.datasets()
    img = []
    for i in data:
        # print(i)  # 具体类别,只有一个par
        img = hdf.select(i)[:]  # 图像数据
        # print(img.shape)
    return img

def readpar(root,city_ublr): #传入城市的切片索引
    u,b,l,r = city_ublr[0],city_ublr[1],city_ublr[2],city_ublr[3]
    """读取文件夹下的所有hdf，2013-2019并concatenate"""
    years = os.listdir(root)
    par_timesum = np.array([np.zeros((b-u,r-l))])
    for year in years:
        files = os.listdir(os.path.join(root,year))
        for file in files:
            parfile = os.path.join(root,year,file)
            if "checkpoints" in parfile:
                os.rmdir(parfile)
                continue
            par_timesum = np.concatenate((par_timesum,np.array([readhdf(parfile)[u:b,l:r]])),axis=0)
    par_timesum = np.delete(par_timesum,0,0)
    return par_timesum


# In[5]:



# par_timesum.shape


# In[6]:


def predone(par_timesum):
    """数据预处理"""
    return np.expand_dims((par_timesum.reshape(par_timesum.shape[0],-1)),axis=-1)

# par_datasets.shape


# In[108]:


# batch_size =句子个数
# num_step 每个batch的序列长度 = 每个batch的时间步数
# train_iter,vocab=d2l.load_data_time_machine(batch_size,time_step)

# vocab 把数字转为单词


# In[ ]:


# F.one_hot(torch.tensor([0,2]),len(vocab)) # 28个词的独热编码，第0和第2个下标为1


# In[94]:


# #模型输入形状的示例
# X=torch.arange(10).reshape((2,5))
# print(X)
# F.one_hot(X.T,28).shape  #（批量大小，时间步数）->（时间步数，批量大小）


# In[7]:


def get_params(vocab_size,num_hiddens,device):
    num_inputs=num_outputs=vocab_size # 每个词都被变成vocab_size的独热编码向量
    
    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01
    # 模型就是一个输入、一个隐藏层、 一个输出
    W_xh=normal((num_inputs,num_hiddens))
    W_hh=normal((num_hiddens,num_hiddens))  # 多的一个w_hh，用来拟合上一时刻的隐藏层权重与这一时刻的隐藏层权重
    b_h=torch.zeros(num_hiddens,device=device)
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)
    params=[W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# In[8]:


def init_rnn_state(batch_size,num_hiddens,device):    # 初始的上一时刻隐藏层状态
    return (torch.zeros((batch_size,num_hiddens),device=device),)


# In[9]:


def rnn(inputs,state,params):  #一个forward
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs: #inputs：（传进来的数据的时间长度，batch_size，vocab_size）
        # print("size:",X.size(),W_xh.size())
        H=torch.tanh(torch.mm(X,W_xh)+torch.mm(H,W_hh)+b_h) #计算当前时刻的隐藏层（状态）
        # print(H.size())
        Y=torch.mm(H,W_hq)+b_q
        outputs.append(Y)  
    return torch.cat(outputs,dim=0),(H,) # concat所有时间步数一起输出


# In[10]:


class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size,self.num_hiddens=vocab_size,num_hiddens
        self.params=get_params(vocab_size,num_hiddens,device)  # 获取初始化参数
        self.init_state,self.forward_fn=init_state,forward_fn
    def __call__(self,X,state):
        # X=F.one_hot(X,self.vocab_size).type(torch.float32) # 把输入X变成timestep为第一维的形状，加上onehot
        X=X.type(torch.float32)
        return self.forward_fn(X,state,self.params) # 输入网络，返回预测值和新的隐藏层状态
    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)


# In[ ]:


# net=RNNModelScratch(vocab_size,num_hiddens,d2l.try_gpu(),get_params,init_rnn_state,rnn)
# state=net.begin_state(X.shape[0],d2l.try_gpu())
# Y,new_state=net(X.to(d2l.try_gpu()),state)
# Y.shape,len(new_state),new_state[0].shape  # len(new_state)=1代表只有一层隐藏层


# In[11]:


def predict_ch8(prefix,num_preds,net,vocab,device):
    """对一句话的预测"""
    state=net.begin_state(batch_size=1,device=device)
    outputs=[vocab[prefix[0]]]
    get_input=lambda:torch.tensor([outputs[-1]],device=device).reshape((1,1)) # 传入上一个字母
    for y in prefix[1:]:  #根据前一个字母进行下一个字母的预测
        _,state=net(get_input(),state) # 给定的词，不需要预测值y，直接用给定的
        outputs.append(vocab[y])  
    for _ in range(num_preds):  
        y,state=net(get_input(),state)   # 返回预测值的独热编码（1，vocab_size）和新的隐藏层状态
        outputs.append(int(y.argmax(dim=1).reshape(1))) # 添加预测值最大的下标（类似分类）
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# predict_ch8('time traveller ',10,net,vocab,d2l.try_gpu())


# In[12]:


def grad_clipping(net,theta):
    if isinstance(net,nn.Module):
        params=[p for p in net.parameters() if p.requires_grad]
    else:
        params=net.params
    norm=torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm


# In[42]:


"""train one epoch"""
def train_epoch_ch8(net,par_datasets,loss,updater,device,time_len,time_step,use_random_iter):  # 传入的par_datasets已经转置
    state,timer=None,d2l.Timer()
    metric=d2l.Accumulator(2) # 长度为2的累加器
    for i in range(time_len-time_step):
        if (i+1)*time_step+1 > time_len:
            break
        batch_par_x = par_datasets[i:i+time_step]
        batch_par_y = par_datasets[i+1:i+time_step+1]
    # for X,Y in train_iter:  # X,Y 的size都是（32，35） Y是X的往后一步数据集
        if state is None or use_random_iter:  
            # 在第一次迭代或使用随机抽样时初始化state
            state=net.begin_state(batch_size=batch_par_x.shape[1],device=device)
        else:
            # 判断批量样本之间是否连续
            if isinstance(net,nn.Module) and not isinstance(state,tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_() # 丢弃梯度
        y=batch_par_y.reshape(-1)
        batch_par_x,y = torch.tensor(batch_par_x),torch.tensor(y)
        batch_par_x,y=batch_par_x.to(device),y.to(device)
        y_hat,state=net(batch_par_x,state)
        y_hat = torch.squeeze(y_hat)
        l=loss(y_hat.float(),y.float()).mean().float()
        # l = loss(y_hat, y.long()).mean()
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net,1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net,1)
            updater(batch_size=1)
        metric.add(l*y.numel(),y.numel())
    return metric[0]/metric[1],metric[1]/timer.stop()


# In[43]:


"""基础训练  结构"""
def train_ch8(net, par_datasets, vocab, lr, num_epochs, device,time_len,time_step,
              use_random_iter=False):
    # loss = nn.CrossEntropyLoss()
    loss = nn.SmoothL1Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    # predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net,par_datasets,loss,updater,device,time_len,time_step,use_random_iter)
        if (epoch + 1) % 10 == 0:
            # print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    # print(predict('time traveller'))
    # print(predict('traveller'))


# In[44]:

if __name__ == '__main__':
    beijing_ublr = [996, 1008, 5921, 5937]
    #TODO
    # 确认读取是不是按时间顺序的
    par_timesum = readpar("parr", beijing_ublr)
    time_len = par_timesum.shape[0]  #总时序长度
    batch_size=par_timesum.shape[1]*par_timesum.shape[2]
    time_step = 2  # 每个batch的时序长度
    assert time_step < time_len
    num_hiddens=512 # 隐藏状态输出形状
    vocab_size=1  # 独热编码维度长度
    num_epochs, lr = 500, 1
    vocab=None
    print(par_timesum.max(),par_timesum.min(),par_timesum.mean(),np.median(par_timesum))
    # par_datasets = predone(par_timesum)
    # net = RNNModelScratch(vocab_size, num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
    # train_ch8(net, par_datasets, vocab, lr, num_epochs, d2l.try_gpu(),time_len,time_step,
    #           use_random_iter=False)





