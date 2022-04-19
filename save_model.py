import torch
import torchvision

# res18 = torchvision.models.resnet18(pretrained=True)
#
# torch.save(res18.state_dict(), './resnet18.pth')
#
#
# c = torch.load('./resnet18.pth')
# res18.load_state_dict(c)


res50 = torchvision.models.resnet50(pretrained=True)

torch.save(res50.state_dict(), './resnet50.pth')


c = torch.load('./resnet50.pth')
res50.load_state_dict(c)



