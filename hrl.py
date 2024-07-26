import torch
from torch import nn
from IPython import display
from d2l import torch as d2l


class train_epoch_ch3_hrl:
    """自定义训练数据"""

    def __init__(self, net, train_iter, test_iter, loss, num_epochs, updater):
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.loss = loss
        self.num_epochs = num_epochs
        self.updater = updater

    class Accumulator:  # @save
        """在n个变量上累加"""

        def __init__(self, n):
            self.data = [0.0] * n

        def add(self, *args):
            self.data = [a + float(b) for a, b in zip(self.data, args)]

        def reset(self):
            self.data = [0.0] * len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def accuracy(self, y_hat, y):  # @save
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    def test_epoch_ch3(self, net, test_iter, loss, updater):  # @save
        """训练模型一个迭代周期（定义见第3章）"""
        # 将模型设置为训练模式
        if isinstance(net, torch.nn.Module):
            net.eval()
        # 训练损失总和、训练准确度总和、样本数
        metric = self.Accumulator(3)

        for X, y in test_iter:
            # 计算梯度并更新参数
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                # 使用PyTorch内置的优化器和损失函数
                l.mean().backward()
            else:
                # 使用定制的优化器和损失函数
                l.sum().backward()
            metric.add(float(l.sum()), self.accuracy(y_hat, y), y.numel())
            # 返回训练损失和训练精度

        return metric[0] / metric[2], metric[1] / metric[2]

    def train_epoch_ch3(self, net, train_iter, loss, updater):  # @save
        """训练模型一个迭代周期（定义见第3章）"""
        # 将模型设置为训练模式
        if isinstance(net, torch.nn.Module):
            net.train()
        # 训练损失总和、训练准确度总和、样本数
        metric = self.Accumulator(3)

        for X, y in train_iter:
            # 计算梯度并更新参数
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                # 使用PyTorch内置的优化器和损失函数
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                # 使用定制的优化器和损失函数
                l.sum().backward()
                updater(X.shape[0])
            metric.add(float(l.sum()), self.accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练精度
        return metric[0] / metric[2], metric[1] / metric[2]

    class Animator:  # @save
        """在动画中绘制数据"""

        def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                     ylim=None, xscale='linear', yscale='linear',
                     fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                     figsize=(3.5, 2.5)):
            # 增量地绘制多条线
            if legend is None:
                legend = []
            d2l.use_svg_display()
            self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
            if nrows * ncols == 1:
                self.axes = [self.axes, ]
            # 使用lambda函数捕获参数
            self.config_axes = lambda: d2l.set_axes(
                self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
            self.X, self.Y, self.fmts = None, None, fmts

        def add(self, x, y):
            # 向图表中添加多个数据点
            if not hasattr(y, "__len__"):
                y = [y]
            n = len(y)
            if not hasattr(x, "__len__"):
                x = [x] * n
            if not self.X:
                self.X = [[] for _ in range(n)]
            if not self.Y:
                self.Y = [[] for _ in range(n)]
            for i, (a, b) in enumerate(zip(x, y)):
                if a is not None and b is not None:
                    self.X[i].append(a)
                    self.Y[i].append(b)
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                self.axes[0].plot(x, y, fmt)
            self.config_axes()
            display.display(self.fig)
            display.clear_output(wait=True)

        """训练模型（定义见第3章）"""


    def main(self):
        animator = self.Animator(xlabel='epoch', xlim=[1, self.num_epochs], ylim=[0.3, 0.9],
                                 legend=['train loss', 'train acc', 'test loss', 'test acc'])

        for epoch in range(self.num_epochs, ):
            train_metrics = self.train_epoch_ch3(self.net, self.train_iter, self.loss, self.updater)
            test_metrics = self.test_epoch_ch3(self.net, self.test_iter, self.loss, self.updater)
            test_loss, test_acc = test_metrics
            animator.add(epoch + 1, train_metrics + test_metrics)
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc

        print(f"train loss:{train_loss}\ntrain acc:{train_acc}\ntest loss:{test_loss}\ntest acc:{test_acc}")


def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])