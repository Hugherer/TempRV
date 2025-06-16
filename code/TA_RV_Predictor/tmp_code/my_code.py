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

train_size = 50         # 训练集大小
num_time_steps = 8      # 训练时时间窗的步长
input_size = 2          # 输入数据维度
output_size = 2         # 输出维度
hidden_size = 16        # 隐含层维度
num_layers = 2          # 隐含层层数
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

    tdl_data = []
    for _ in range(train_size):
        time_x = []
        tdl_y = []

        #J120 mission; J35 reconnaissance; J91 attack; J90 finish; J22 goback
        x_tmp = []
        y_tmp = []
        #tdl_list = [120, 35, 91, 90, 22]

        for mm in range(10):
            x_tmp.append([np.random.randint(0, 10) + 10*mm])
            time_x = sorted(x_tmp, key=lambda x: x[0],reverse=1)
            #tdl_y.append([tdl_list[np.random.randint(0, 5)]])
            y_tmp.append([np.random.randint(0, 10)])
            tdl_y = sorted(y_tmp, key=lambda x: x[0],reverse=1)

        data = np.concatenate((time_x, tdl_y), axis = 1)

        tdl_data.append(data)

        ######################################################
        
        lst = []
        for j in range(10):
            tmp_l = []
            tmp_l.append(time_x[j][0])
            tmp_l.append(tdl_y[j][0])
            lst.append(tmp_l)

        with open('input_data.txt', 'a') as file:
            file.write(str(lst))
            file.write('\n')
        

    return tdl_data

#####################开始训练模型#################################
def train_RNN(data):
    model = Net(input_size, hidden_size, num_layers)  # 初始化RNN模型实例
    print('model:\n', model)                          # 打印模型结构，便于调试和查看

    criterion = nn.MSELoss()                          # 定义损失函数，这里使用均方误差（MSE）作为损失函数
    optimizer = optim.Adam(model.parameters(), lr)    # 定义优化器，这里选择Adam，并设置学习率lr

    # 初始化隐状态hidden_prev为全零张量，形状为 (num_layers, batch_size, hidden_size)
    hidden_prev = torch.zeros(num_layers, 1, hidden_size)

    l = []                                            # 创建一个空列表l，用于存储每次迭代的损失值

    train_data = data[0 : int(train_size*0.8 + 1)]
    # 训练3000次
    start = 0
    for iter in range(3000):

        
        data_tmp = train_data[start]

        # 构建输入x
        x = torch.tensor(data_tmp[0:5]).float().view(1, 5, input_size)

        # 构建目标y
        y = torch.tensor(data_tmp[5:10]).float().view(1, 5, output_size)

        output, hidden_prev = model(x, hidden_prev)   # 前向传播：计算模型输出和新的隐状态
        hidden_prev = hidden_prev.detach()            # 将新的隐状态与计算图分离，防止梯度累积

        loss = criterion(output, y)                   # 计算损失值
        model.zero_grad()                             # 清除所有参数的梯度信息，避免梯度累积
        loss.backward()                               # 反向传播：计算梯度
        optimizer.step()                              # 更新模型参数

        if iter % 100 == 0:
            print("Iteration: {} loss {}".format(iter, loss.item()))  # 每100次迭代打印一次损失值
            l.append(loss.item())                                     # 并将当前损失值添加到列表l中

        start = (start + 1) % int(train_size*0.8)

    # Save the trained model's state dictionary
    save_path = 'model_path'
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
    data_test_tmp = data[40:50]
    
    # 将选取的数据转换为张量，并增加一个维度以匹配模型输入的要求 (batch_size, seq_len, input_size)
    # 这里的 batch_size 是1，seq_len 是10，input_size 是3
    
    model = Net(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    hidden_prev = torch.zeros(num_layers, 1, hidden_size)

    for i in range(10):
        data_test = torch.tensor(np.expand_dims(data_test_tmp[i][0:5], axis=0), dtype=torch.float32)

    # 使用模型对data_test进行第一次预测，得到pred和新的隐状态h
        pred, h = model(data_test, hidden_prev)
        print('######################## here is the real data ############################')
        print(data_test_tmp[i][5:10])
        print('######################## here is the predicted data #######################')
        print('pred:', pred)  # 打印预测结果的形状，帮助调试

        pred = pred.detach().numpy().reshape(5, output_size)


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

    #start = datetime.datetime.now()
    #model_path = train_RNN(data)
    #end = datetime.datetime.now()
    #print('The training time: %s' % str(end - start))
    #RNN_pre(model_path, data)

if __name__ == '__main__':
    main()
