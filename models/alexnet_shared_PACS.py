import torch
import torch.nn as nn
import torch.optim as optim
import pathlib

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


normalize = transforms.Normalize(
	mean=[0.485, 0.456, 0.406],
	std=[0.229, 0.224, 0.225]
)

data_transforms = {
	'train':
	transforms.Compose([
		transforms.Resize((224,224)),
		# transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.ToTensor(),
		# normalize
	]),
	'validation':
	transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		# normalize
	]),
}

root = pathlib.Path(__file__).parent.parent.resolve()
dataset_root = f"{root}/datasets/PACS_shared"
model_save_dir = f"{root}/model_checkpoints"

image_datasets = {
	'train': 
	datasets.ImageFolder(f'{dataset_root}/Train', data_transforms['train']),
	'test': 
	datasets.ImageFolder(f'{dataset_root}/Test', data_transforms['validation']),
}

batch_size = 16
dataloaders = {
	'train':
	torch.utils.data.DataLoader(image_datasets['train'],
								batch_size=batch_size,
								shuffle=True, num_workers=4),
	'test':
	torch.utils.data.DataLoader(image_datasets['test'],
								batch_size=32,
								shuffle=False, num_workers=4),
}

class AlexNet_Shared(nn.Module):
	def __init__(self):
		super(AlexNet_Shared, self).__init__()
		self.backbone = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 ),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.domain_classifier = nn.Sequential(
			nn.Linear(in_features = 6400, out_features = 4096),
			nn.ReLU(inplace=True),
			nn.Linear(in_features = 4096, out_features = 128),
			nn.ReLU(inplace=True),
			nn.Linear(in_features = 128, out_features = 4)
		)
		self.object_classifier = nn.Sequential(
			nn.Linear(in_features = 6400, out_features = 4096),
			nn.ReLU(inplace=True),
			nn.Linear(in_features = 4096, out_features = 128),
			nn.ReLU(inplace=True),
			nn.Linear(in_features = 128, out_features = 7)
		)

	def forward(self,x):
		x = self.backbone(x)
		x = x.view(x.size(0), -1)
		domain_output = self.domain_classifier(x)
		object_output = self.object_classifier(x)
		return domain_output, object_output
	
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #training with either cpu or cuda

model = AlexNet_Shared() #to compile the model
model = model.to(device=device) #to send the model for training on either cuda or cpu

## Loss and optimizer
learning_rate = 1e-4 #I picked this because it seems to be the most used by experts

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate) #Adam seems to be the most popular for deep learning

EPOCHS = 50
for epoch in range(EPOCHS):
	loss_ep = 0
	
	train_dom_correct = 0
	train_obj_correct = 0
	train_both_correct = 0
	train_num_samples = len(dataloaders['train'].dataset)

	for batch_idx, data in enumerate(dataloaders['train']):
		image, target = data

		image = image.to(device=device)
		target = target.to(device=device)

		optimizer.zero_grad()

		dom_scores, obj_scores = model(image)
		# there are 4 domains and 7 objects
		# dom_target = target // 7 # reecived warning
		dom_target = torch.div(target, 7, rounding_mode='floor')
		obj_target = target % 7
		
		dom_loss = criterion(dom_scores, dom_target)
		obj_loss = criterion(obj_scores, obj_target)

		total_loss = dom_loss/4 + obj_loss/7 # so that is fair
		total_loss.backward()
		optimizer.step()
		loss_ep += total_loss.item()

		# check acc
		_, dom_pred = dom_scores.max(1)
		_, obj_pred = obj_scores.max(1)

		train_dom_correct += (dom_pred == dom_target).sum()
		train_obj_correct += (obj_pred == obj_target).sum()
		train_both_correct += torch.logical_and(dom_pred == dom_target, obj_pred == obj_target).sum()
	
	print(f"""
		Loss in epoch {epoch} :::: {loss_ep/batch_size}, 
		train dom acc ::: {train_dom_correct/float(train_num_samples):.4f}
		train obj acc ::: {train_obj_correct/float(train_num_samples):.4f}
		train_both_acc ::: {train_both_correct/float(train_num_samples):.4f}
	""")
	
	with torch.no_grad():
		test_dom_correct = 0
		test_obj_correct = 0
		test_both_correct = 0
		test_num_samples = len(dataloaders['test'].dataset)
		for batch_idx, data in enumerate(dataloaders['test']):
			image, target = data

			image = image.to(device=device)
			target = target.to(device=device)
			
			## Forward Pass
			dom_scores, obj_scores = model(image)
			_, dom_pred = dom_scores.max(1)
			_, obj_pred = obj_scores.max(1)

			dom_target = torch.div(target, 7, rounding_mode='floor')
			obj_target = target % 7

			test_dom_correct += (dom_pred == dom_target).sum()
			test_obj_correct += (obj_pred == obj_target).sum()
			test_both_correct += torch.logical_and(dom_pred == dom_target, obj_pred == obj_target).sum()

		print(f"""
			test dom acc ::: {test_dom_correct}/{test_num_samples} being {test_dom_correct/float(test_num_samples):.4f}
			test obj acc ::: {test_obj_correct}/{test_num_samples} being {test_obj_correct/float(test_num_samples):.4f}
			test_both_acc ::: {test_both_correct}/{test_num_samples} being {test_both_correct/float(test_num_samples):.4f}
		""")

torch.save(model.state_dict(), f"{model_save_dir}/alexnet_shared_pacs.pth")
