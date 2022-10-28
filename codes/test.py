import torch
import torch.nn.functional as F
from tqdm import tqdm


def test(model, config, test_loader):
    """Start test"""
    if config['mode']['name'] == 'gpu':
        device = torch.device('cuda:{}'.format(str(config['mode']['ids'])))
    else:
        device = torch.device('cpu')

    # Test mode
    model.eval()
    model = model.to(device)

    # Define
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            # SpeedUp
            data, target = data.to(device), target.to(device)

            # Input model
            output = model(data)

            # Calculate Loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # Get result
            pred = output.argmax(dim=1, keepdim=True)

            # Calculate accuracy
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate Avg Loss
    test_loss /= len(test_loader.dataset)

    # Print
    tqdm.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    tqdm.write('{{"metric": "Eval - NLL Loss", "value": {}}}'.format(
        test_loss))
    tqdm.write('{{"metric": "Eval - Accuracy", "value": {}}}\n'.format(
        100. * correct / len(test_loader.dataset)))

