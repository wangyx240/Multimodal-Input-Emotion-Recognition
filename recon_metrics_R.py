import torch


def accuracy(output, target):
    # target = target.cpu().repeat_interleave(15, 0).repeat_interleave(8, 1).cuda().long()
    # assert output.shape[0] == target.shape[0]
    # with torch.no_grad():
    #     output = torch.argmax(output, dim=1)
    #     assert output.shape[0] == len(target)
    #     correct = 0
    #     correct += torch.sum(output == target).item()
    # return correct / len(target)
    with torch.no_grad():
        _, idx = torch.max(output, 1, keepdim=True)
    return sum(target == idx.squeeze()), len(target)
