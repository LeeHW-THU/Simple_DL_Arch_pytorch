import torch
import torch.nn.functional as F
def test(model, test_loader):
    """测试"""
    device = torch.device('cpu')
    # 测试模式
    model.eval()
    model=model.to(device)
    # 存放正确个数
    correct = 0

    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:

            # 加速
            data, target = data.to(device), target.to(device)

            # 获取结果
            output = model(data)

            test_loss += F.nll_loss(output,target,reduction='sum').item()

            # 预测结果
            pred = output.argmax(dim=1, keepdim=True)

            # 计算准确个数
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算准确率
    test_loss /= len(test_loader.dataset)

    # 输出准确
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('{{"metric": "Eval - NLL Loss", "value": {}}}'.format(
        test_loss))
    print('{{"metric": "Eval - Accuracy", "value": {}}}\n'.format(
        100. * correct / len(test_loader.dataset)))