# Work log
#
# 10.4.2019
#   - Added train + validate functions
#   - Changed labels from string to indices in generate_sets()
#   - We kept the checkpoint savefunctionality, however we need to look into it some more




import argparse
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data
import os
import scipy
from scipy import ndimage
import scipy.misc
import torch
import torchvision


kMinSamplesPerPerson = 400
kTrainSamplesCount = 300

kBatchSizeTrain = 40
kBatchSizeTest = 40

bUseCuda = False

learning_rate = 0.01

save_path = './state_'
PATH_TO_PHOTOS = "../data/train"

knEpochsCount = 1
knStartEpoch = 0

plot_x = []
plot_y = []

load_name = 'default'

def train_loader(path):
	img = Image.open(path)
	# if args.aug != "off":
	#	pix = np.array(img)
	#	pix_aug = img_augmentation(pix)
	#	img = Image.fromarray(np.uint8(pix_aug))
	# print pix
	return img


def test_loader(path):
	img = Image.open(path)
	return img


class ImageList(data.Dataset):
	def __init__(self, fileList, transform=None, image_loader=None):
		self.imgList = fileList
		self.image_loader = image_loader
		self.transform = transform

	def __getitem__(self, index):
		imgPath, label = self.imgList[index]

		# Read the image
		path_to_photo = os.path.join(imgPath)
		if self.image_loader:
			img = self.image_loader(path_to_photo)
		else:
			img = Image.open(os.path.join(path_to_photo))

		if self.transform is not None:
			img = self.transform(img)

		return img, label

	def __len__(self):
		return len(self.imgList)


def generate_sets(dir_path):
	
	train_set, test_set = [], []
	
	candidates = os.listdir(dir_path)
	classes = []
	idx_of_label = {candidates[idx]: idx for idx in range(len(candidates))}
	
	for d in candidates:

		if d.startswith('.'):
			continue

		subdir_path = os.path.join(dir_path, d)

		file_list = os.listdir(subdir_path)
		
		if len(file_list) >= kMinSamplesPerPerson:

            
            
			classes.append(d)
			class_idx = len(classes) - 1
			file_list_permute = np.random.permutation(file_list)
			train_set += [(os.path.join(subdir_path, f), class_idx) for f in
						 file_list_permute[:kTrainSamplesCount]]

			test_set += [(os.path.join(subdir_path, f), class_idx) for f in
						 file_list_permute[kTrainSamplesCount:]]
            
        if len(classes) == 100:
            break
            
	return train_set, test_set, classes


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = 0.1 * (0.1 ** (epoch // 3))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def train(train_dataloader, forward_pass, criterion, optimizer, epoch):
	print("Start training!")
	running_loss = 0.0
	iteration_number= 0

	for i, data in enumerate(train_dataloader):
		img, label = data
		
		if bUseCuda:
			img, label = Variable(img).cuda(), Variable(label).cuda()
		else:
			img, label = Variable(img), Variable(label)
			
		optimizer.zero_grad()
		output = forward_pass(img)
		softmax = criterion(output) # 20* 5749
		loss = -torch.mean(softmax[range(len(label)), label]) #
		loss.backward()
		optimizer.step()

		current_loss = loss.item()
		print("Epoch: {}, current iter: {}/{}\n Current loss {}\n".format(epoch, i, len(train_dataloader), current_loss))
		running_loss += current_loss
		# print i, loss.data[0], "/", len(train_dataloader)

		plot_x.append(len(plot_x)+1)
		plot_y.append(current_loss)
		
	return running_loss


def validate(test_dataloader, forward_pass, criterion):
	print("Start validation!")
	correct_predictions = 0
	total = 0

	if load_name != 'default':
		checkpoint = torch.load(load_name)
		args.start_epoch = checkpoint['epoch']
		forward_pass.load_state_dict(checkpoint['state_dict'])

	for i, data in enumerate(test_dataloader):
		img, label = data

		if bUseCuda:
			img, label = Variable(img).cuda(), Variable(label).cuda()
		else:
			img, label = Variable(img), Variable(label)

		output = forward_pass(img)

		# 1. Calculate loss (as in train)
		softmax = criterion(output)
		loss = -torch.mean(softmax[range(len(label)), label])


		# 2. Calculate performance metrics
		_, predicted = torch.max(output, 1)
		correct_predictions += torch.sum(predicted == label)
		total += len(label)

		print("Current iter: {}/{}\n Current correct predictions {}/{}\n".format(
			i, len(test_dataloader), correct_predictions, total))

	return correct_predictions, total



def main():
	global knEpochsCount
	train_set, test_set, classes = generate_sets(PATH_TO_PHOTOS)
	
	train_dataloader = torch.utils.data.DataLoader(
						ImageList(fileList=train_set,
								transform=transforms.Compose([ 
								transforms.Scale((128,128)),
								transforms.ToTensor(),            ])),
						shuffle=True,
						num_workers=0,
						batch_size=kBatchSizeTrain)

	test_dataloader = torch.utils.data.DataLoader(
						ImageList(fileList=test_set,
								transform=transforms.Compose([ 
								transforms.Scale((128,128)),
								transforms.ToTensor(),            ])),
						shuffle=True,
						num_workers=0,
						batch_size=kBatchSizeTest)
	
	if bUseCuda:
		forward_pass = torchvision.models.alexnet(pretrained=True).cuda()
	else:
		forward_pass = torchvision.models.alexnet(num_classes=len(classes))
		
	optimizer = optim.Adam(forward_pass.parameters(), lr=learning_rate)
	
	criterion = nn.LogSoftmax()# TBD change criterion!!
	

	validate_plotx = []
	validate_ploty = []
	training_plot = "p1b_trainloss.txt"
	validate_plot = "p1b_validate.txt"
	if load_name != "default":
		knEpochsCount = 1
	for epoch in range(knStartEpoch, knEpochsCount):

		adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		if load_name == "default":
			running_loss = train(train_dataloader, forward_pass, criterion, optimizer, epoch)
		correct, total = validate(test_dataloader, forward_pass, criterion)

		validate_plotx.append(epoch+1)
		validate_ploty.append(float(correct)/total)

		print("correct matches: ", correct, "total matches: ", total)
		print("total accuracy = ", float(correct)/total)
		
		# evaluate on validation set
		save_name = save_path + str(epoch) + '_checkpoint.pth.tar'
		torch.save({
			'epoch': epoch + 1,
	#         'arch': args.arch,
	        'state_dict': forward_pass.state_dict(),
			# 'prec1': prec1,
			}, save_name)
	with open(training_plot, 'w') as f:
		for i in range(0, len(plot_x)):
			f.write(" ".join([str(plot_x[i]), str(plot_y[i])]))
			f.write('\n')

	with open(validate_plot, 'w') as f:
		for i in range(0,len(validate_plotx)):
			f.write(" ".join([str(validate_plotx[i]),str(validate_ploty[i])]))
			f.write('\n')
	print("done")


def plot_training_loss():
	txt_file = 'p1b_trainloss.txt'
	plot_x = []
	plot_y = []
	with open(txt_file, 'r') as f:
		for line in f:
			data = line.strip()
			data = data.split(' ')
			plot_x.append(int(data[0]))
			plot_y.append(float(data[1]))
	plt.plot(plot_x, plot_y, 'b', label = "training loss")
	plt.title('training loss')
	plt.show()


def plot_text_loss():
	txt_file = 'p1b_validate.txt'
	plot_x = []
	plot_y = []
	with open(txt_file, 'r') as f:
		for line in f:
			data = line.strip()
			data = data.split(' ')
			plot_x.append(int(data[0]))
			plot_y.append(float(data[1]))
	plt.plot(plot_x, plot_y, 'r', label = "validate accuracy")
	plt.title('validate accuracy')
	plt.show()


if __name__ == '__main__':
	main()
	plot_training_loss()
	plot_text_loss()
