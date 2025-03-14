## main.py
主程序 命令行python main.py运行整个程序.main.py运行后调用data.py处理数据, 调用model.py训练模型, 并得到每一轮epoch训练之后的accuracy和loss, 将结果以h5py文件形式存储在./data/output/h5文件夹, 同时plot.py绘图并保存在./data/output/figure, 最后保存网络.
## data.py
用于下载数据以及将数据集分割为给定参数大小的batch, 下载的数据存储在./data/FashionMINST
## model.py
用于构建网络, 内含def trainBatch, def testBatch方法, 用于网络的训练和测试
## plot.py
绘图, 横坐标是epoch, 纵坐标是每一轮epoch循环之后的accuracy和loss值.

## 训练参数
遍历
learning_rate = [0.01, 0.05, 0.1],
batch_size = [64, 128, 256],
共九组


