
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size,num_steps=32,35
# batch_size =句子个数
# num_step 序列长度 = 时间步数
train_iter,vocab=d2l.load_data_time_machine(batch_size,num_steps)

# vocab 把数字转为单词
#模型输入形状的示例
X=torch.arange(10).reshape((2,5))
print(X)
F.one_hot(X.T,28).shape  #（批量大小，时间步数）->（时间步数，批量大小）


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size  # 每个词都被变成vocab_size的独热编码向量

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 模型就是一个输入、一个隐藏层、 一个输出
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))  # 多的一个w_hh，用来拟合上一时刻的隐藏层权重与这一时刻的隐藏层权重
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size,num_hiddens,device):    # 初始的上一时刻隐藏层状态
    return (torch.zeros((batch_size,num_hiddens),device=device),)

def rnn(inputs,state,params):  #一个forward
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs: #inputs：（时间步数，batch_size，vocab_size）
        # print("size:",X.size(),W_xh.size())
        H=torch.tanh(torch.mm(X,W_xh)+torch.mm(H,W_hh)+b_h) #计算当前时刻的隐藏层（状态）
        # print(H.size())
        Y=torch.mm(H,W_hq)+b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,) # concat所有时间步数一起输出

class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size,self.num_hiddens=vocab_size,num_hiddens
        self.params=get_params(vocab_size,num_hiddens,device)  # 获取初始化参数
        self.init_state,self.forward_fn=init_state,forward_fn
    def __call__(self,X,state):
        X=F.one_hot(X.T,self.vocab_size).type(torch.float32) # 把输入X变成timestep为第一维的形状，加上onehot
        return self.forward_fn(X,state,self.params) # 输入网络，返回预测值和新的隐藏层状态
    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)

num_hiddens=512
net=RNNModelScratch(len(vocab),num_hiddens,d2l.try_gpu(),get_params,init_rnn_state,rnn)

# state=net.begin_state(X.shape[0],d2l.try_gpu())
# Y,new_state=net(X.to(d2l.try_gpu()),state)
# Y.shape,len(new_state),new_state[0].shape  # len(new_state)=1代表只有一层隐藏层

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

def grad_clipping(net,theta):
    if isinstance(net,nn.Module):
        params=[p for p in net.parameters() if p.requires_grad]
    else:
        params=net.params
    norm=torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm

"""train one epoch"""
def train_epoch_ch8(net,train_iter,loss,updater,device,use_random_iter):
    state,timer=None,d2l.Timer()
    metric=d2l.Accumulator(2) # 训练损失之和,词元数量
    for X,Y in train_iter:  # X,Y 的size都是（32，35） Y是X的往后一步数据集
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state=net.begin_state(batch_size=X.shape[0],device=device)
        else:
            # 判断批量样本之间是否连续
            if isinstance(net,nn.Module) and not isinstance(state,tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_() # 丢弃梯度
        y=Y.T.reshape(-1)
        X,y=X.to(device),y.to(device)
        y_hat,state=net(X,state)
        l=loss(y_hat,y.long()).mean()
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
        a,b = l * y.numel(), y.numel()
    return math.exp(metric[0]/metric[1]),metric[1]/timer.stop()

"""基础训练  结构"""
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())