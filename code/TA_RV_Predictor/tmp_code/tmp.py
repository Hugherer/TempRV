import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
###########################设置全局变量###################################

num_time_steps = 5    # 训练时时间窗的步长
input_size = 3          # 输入数据维度
hidden_size = 16        # 隐含层维度
output_size = 3         # 输出维度
num_layers = 1
lr=0.01
####################定义RNN类##############################################

class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        for p in self.rnn.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):

       out, hidden_prev = self.rnn(x, hidden_prev)
       # [b, seq, h]
       out = out.view(-1, hidden_size)
       out = self.linear(out)#[seq,h] => [seq,3]
       out = out.unsqueeze(dim=0)  # => [1,seq,3]
       return out, hidden_prev

####################初始化训练集#################################
def getdata():

    time_x = []
    tdl_y = []

    #J120 mission; J35 reconnaissance; J91 attack; J90 finish; J22 goback
    x_tmp = []
    tdl_list = [120, 35, 91, 90, 22]

    for _ in range(10):
        x_tmp.append([np.random.randint(0, 31)])
        time_x = sorted(x_tmp, key=lambda x: x[0])
        tdl_y.append([tdl_list[np.random.randint(0, 5)]])
    
    z = (np.zeros_like(time_x) + 2).reshape(10,1)
    data = np.concatenate((time_x, tdl_y, z), axis = 1)

    #x1 = np.linspace(1,10,30).reshape(30,1)
    #y1 = (np.zeros_like(x1) + 2) + np.random.rand(30,1)*0.1
    #z1 = (np.zeros_like(x1) + 2).reshape(30,1) 
    #tr1 = np.concatenate((x1, y1, z1), axis=1)

    # data = mm.fit_transform(tr1)   #数据归一化

    print(data)
    return data

#####################开始训练模型#################################
def train_RNN(data):
    model = Net(input_size, hidden_size, num_layers)  # 初始化RNN模型实例
    print('model:\n', model)                          # 打印模型结构，便于调试和查看

    criterion = nn.MSELoss()                          # 定义损失函数，这里使用均方误差（MSE）作为损失函数
    optimizer = optim.Adam(model.parameters(), lr)    # 定义优化器，这里选择Adam，并设置学习率lr

    # 初始化隐状态hidden_prev为全零张量，形状为 (num_layers, batch_size, hidden_size)
    hidden_prev = torch.zeros(1, 1, hidden_size)

    l = []                                            # 创建一个空列表l，用于存储每次迭代的损失值

    # 训练3000次
    for iter in range(3000):
        # loss = 0                                    # 注释掉的代码，原意可能是重置loss，但在此上下文中不必要

        # 随机选择一个起始点start，范围在0到9之间（包括0，不包括10）
        start = np.random.randint(5, size=1)[0]
        end = start + 4                             # 设置结束点end，即从start开始的15个时间步长

        # 构建输入x，取data中从start到end（不包括end）的数据，形状变为 (1, num_time_steps - 1, 3)
        x = torch.tensor(data[start:end]).float().view(1, num_time_steps - 1, 3)

        # 构建目标y，实际上是x向后移动了5个时间步长的数据，形状同上
        y = torch.tensor(data[start + 2:end + 2]).float().view(1, num_time_steps - 1, 3)

        output, hidden_prev = model(x, hidden_prev)   # 前向传播：计算模型输出和新的隐状态
        hidden_prev = hidden_prev.detach()            # 将新的隐状态与计算图分离，防止梯度累积

        loss = criterion(output, y)                   # 计算损失值
        model.zero_grad()                             # 清除所有参数的梯度信息，避免梯度累积
        loss.backward()                               # 反向传播：计算梯度
        optimizer.step()                              # 更新模型参数

        if iter % 100 == 0:
            print("Iteration: {} loss {}".format(iter, loss.item()))  # 每100次迭代打印一次损失值
            l.append(loss.item())                                     # 并将当前损失值添加到列表l中

    # Save the trained model's state dictionary
    save_path = 'tmp'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path} successfully ")
    ##############################绘制损失函数#################################
    #plt.plot(l,'r')
    #plt.xlabel('训练次数')
    #plt.ylabel('loss')
    #plt.title('RNN损失函数下降曲线')

    return save_path
#############################预测#########################################

def RNN_pre(model_path, data):
    # 从原始数据中选取一段作为测试数据，即data中的第19到28个时间步（共10个时间步）
    data_test = data[8:10]
    
    # 将选取的数据转换为张量，并增加一个维度以匹配模型输入的要求 (batch_size, seq_len, input_size)
    # 这里的 batch_size 是1，seq_len 是10，input_size 是3
    data_test = torch.tensor(np.expand_dims(data_test, axis=0), dtype=torch.float32)

    model = Net(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    hidden_prev = torch.zeros(1, 1, hidden_size)

    # 使用模型对data_test进行第一次预测，得到pred1和新的隐状态h1
    pred1, h1 = model(data_test, hidden_prev)
    print('pred1:', pred1)  # 打印预测结果的形状，帮助调试

    # 使用上一步的预测结果pred1作为新输入，再次调用模型进行预测，得到pred2和新的隐状态h2
    pred2, h2 = model(pred1, hidden_prev)
    print('pred2:', pred2)  # 同样打印预测结果的形状

    # 将pred1从张量转换为numpy数组，并调整形状为(10, 3)，表示10个时间步的预测值，每个时间步有3个特征
    pred1 = pred1.detach().numpy().reshape(2, 3)
    
    # 对于pred2也做同样的处理
    pred2 = pred2.detach().numpy().reshape(2, 3)

    # 将两次预测的结果拼接起来，形成一个包含20个时间步预测值的数组
    predictions = np.concatenate((pred1, pred2), axis=0)

    # 注释掉的代码：如果在训练时对数据进行了归一化处理，则在这里需要逆变换回原始尺度
    # predictions= mm.inverse_transform(predictions)
    #print('predictions', predictions)


    #############################预测可视化########################################
    #
    #fig = plt.figure(figsize=(9, 6))
    #ax = Axes3D(fig)
    #ax.scatter3D(data[:, 0],data[:, 1],data[:,2],c='red')
    #ax.scatter3D(predictions[:,0],predictions[:,1],predictions[:,2],c='y')
    #ax.set_xlabel('X')
    #ax.set_xlim(0, 8.5)
    #ax.set_ylabel('Y')
    #ax.set_ylim(0, 10)
    #ax.set_zlabel('Z')
    #ax.set_zlim(0, 4)
    #plt.title("RNN航迹预测")
    #plt.show()
    #
    #############################################################################


def main():
    data = getdata()                          # get train data

    start = datetime.datetime.now()
    model_path = train_RNN(data)
    end = datetime.datetime.now()
    print('The training time: %s' % str(end - start))

    #plt.show()

    #RNN_pre(model_path, data)

if __name__ == '__main__':
    main()
