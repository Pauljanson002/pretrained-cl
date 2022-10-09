import torch
from torchvision.datasets import CIFAR10,MNIST,FashionMNIST,SVHN
import torchvision
import os,urllib
import numpy as np
from PIL import Image
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F

class notMNIST_(torch.utils.data.Dataset):

    def __init__(self, root, task_num, num_samples_per_class, train,transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = "https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/master/data/notMNIST.zip"
        self.filename = 'notMNIST.zip'
        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()

        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')

        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')


        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

        self.num_classes = len(set(self.targets))


        if num_samples_per_class:
            x, y, tt, td = [], [], [], []
            for l in range(self.num_classes):
                indices_with_label_l = np.where(np.array(self.targets)==l)

                x_with_label_l = [self.data[item] for item in indices_with_label_l[0]]

                # If we need a subset of the dataset with num_samples_per_class we use this and then concatenate it with a complete dataset
                shuffled_indices = np.random.permutation(len(x_with_label_l))[:num_samples_per_class]
                x_with_label_l = [x_with_label_l[item] for item in shuffled_indices]
                y_with_label_l = [l]*len(shuffled_indices)

                x.append(x_with_label_l)
                y.append(y_with_label_l)

            self.data = np.array(sum(x,[]))
            self.labels = sum(y,[])


        self.tt = [task_num for _ in range(len(self.data))]
        self.td = [task_num + 1 for _ in range(len(self.data))]

    def __getitem__(self, index):
        img, target, tt, td = self.data[index], self.targets[index], self.tt[index], self.td[index]

        img = Image.fromarray(img)#.convert('RGB')
        img = self.transform(img)

        return img, target, tt, td

    def __len__(self):
        return len(self.data)

    def download(self):
        """Download the notMNIST data if it doesn't exist in processed_folder already."""

        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()

class MNIST_RGB(MNIST):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(),mode='L').convert('RGB')
        img = self.transform(img)
        return img, target

class FashionMNIST_RGB(FashionMNIST):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(),mode='L').convert('RGB')
        img = self.transform(img)
        return img, target

cifar10_train_ds = CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
cifar10_test_ds = CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))



not_mnist_train_ds = notMNIST_(root='data', task_num=1, num_samples_per_class=None, train=True, download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
not_mnist_train_ds.targets = [item + 10 for item in not_mnist_train_ds.targets] # remapping targets
not_mnist_test_ds = notMNIST_(root='data', task_num=1, num_samples_per_class=None, train=False, download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
not_mnist_test_ds.targets = [item + 10 for item in not_mnist_test_ds.targets] #remapping targets

svhn_train_ds = SVHN(root='./data', split='train', download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
svhn_train_ds.targets = [item + 30 for item in svhn_train_ds.labels] #remapping targets
svhn_test_ds = SVHN(root='./data', split='test', download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
svhn_test_ds.targets = [item + 30 for item in svhn_test_ds.labels] #remapping targets

mnist_train_ds = MNIST_RGB(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
mnist_test_ds = MNIST_RGB(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))

mnist_train_ds.targets = [item + 20 for item in mnist_train_ds.targets] #remapping targets
mnist_test_ds.targets = [item + 20 for item in mnist_test_ds.targets] #remapping targets

fashionmnist_train_ds = FashionMNIST_RGB(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
fashionmnist_test_ds = FashionMNIST_RGB(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
fashionmnist_train_ds.targets = [item + 40 for item in fashionmnist_train_ds.targets] #remapping targets
fashionmnist_test_ds.targets = [item + 40 for item in fashionmnist_test_ds.targets] #remapping targets

vit_b_16 = timm.create_model("vit_base_patch16_224_in21k",pretrained=True).cuda()
correct,total = 0,0
class_mean_set = []
accuracy_history = []

## Task 0 CIFAR10
train_loader = torch.utils.data.DataLoader(cifar10_train_ds, batch_size=1024, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(cifar10_test_ds, batch_size=1024, shuffle=False, num_workers=4)



X = []
y = []
for (img_batch,label) in train_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    X.append(out)
    y.append(label)
X = np.concatenate(X)
y = np.concatenate(y)
for i in range(0,10):
    image_class_mask = (y == i)
    class_mean_set.append(np.mean(X[image_class_mask],axis=0))
for (img_batch,label) in test_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    predictions = []
    for single_image in out:
        distance = single_image - class_mean_set
        norm = np.linalg.norm(distance,ord=2,axis=1)
        pred = np.argmin(norm)
        predictions.append(pred)
    predictions = torch.tensor(predictions)
    correct += (predictions.cpu() == label.cpu()).sum()
    total += label.shape[0]
print(f"Accuracy at 0 {correct/total}")
accuracy_history.append(correct/total)

## Task 1   NotMNIST

train_loader = torch.utils.data.DataLoader(not_mnist_train_ds, batch_size=1024, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(not_mnist_test_ds, batch_size=1024, shuffle=False, num_workers=4)

X = []
y = []
for (img_batch,label,tt,td) in train_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    X.append(out)
    y.append(label)
X = np.concatenate(X)
y = np.concatenate(y)
for i in range(0,10):
    image_class_mask = (y == i)
    class_mean_set.append(np.mean(X[image_class_mask],axis=0))
for (img_batch,label,tt,td) in test_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    predictions = []
    for single_image in out:
        distance = single_image - class_mean_set
        norm = np.linalg.norm(distance,ord=2,axis=1)
        pred = np.argmin(norm)
        predictions.append(pred)
    predictions = torch.tensor(predictions)
    correct += (predictions.cpu() == label.cpu()).sum()
    total += label.shape[0]
print(f"Accuracy at 1 {correct/total}")
accuracy_history.append(correct/total)
    

## Task 2 MNIST

train_loader = torch.utils.data.DataLoader(mnist_train_ds, batch_size=1024, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(mnist_test_ds, batch_size=1024, shuffle=False, num_workers=4)



X = []
y = []
for (img_batch,label) in train_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    X.append(out)
    y.append(label)
X = np.concatenate(X)
y = np.concatenate(y)
for i in range(0,10):
    image_class_mask = (y == i)
    class_mean_set.append(np.mean(X[image_class_mask],axis=0))
for (img_batch,label) in test_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    predictions = []
    for single_image in out:
        distance = single_image - class_mean_set
        norm = np.linalg.norm(distance,ord=2,axis=1)
        pred = np.argmin(norm)
        predictions.append(pred)
    predictions = torch.tensor(predictions)
    correct += (predictions.cpu() == label.cpu()).sum()
    total += label.shape[0]
print(f"Accuracy at 2 {correct/total}")
accuracy_history.append(correct/total)

## Task 3 SVHN
train_loader = torch.utils.data.DataLoader(svhn_train_ds, batch_size=1024, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(svhn_test_ds, batch_size=1024, shuffle=False, num_workers=4)



X = []
y = []
for (img_batch,label) in train_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    X.append(out)
    y.append(label)
X = np.concatenate(X)
y = np.concatenate(y)
for i in range(0,10):
    image_class_mask = (y == i)
    class_mean_set.append(np.mean(X[image_class_mask],axis=0))
    #class_mean_set[20+i] = np.mean(X[image_class_mask],axis=0)
for (img_batch,label) in test_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    predictions = []
    for single_image in out:
        distance = single_image - class_mean_set
        norm = np.linalg.norm(distance,ord=2,axis=1)
        pred = np.argmin(norm)
        predictions.append(pred)
    predictions = torch.tensor(predictions)
    correct += (predictions.cpu() == label.cpu()).sum()
    total += label.shape[0]
print(f"Accuracy at 3 {correct/total}")
accuracy_history.append(correct/total)


## Task 4 FASHION_MNIST

train_loader = torch.utils.data.DataLoader(fashionmnist_train_ds, batch_size=1024, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(fashionmnist_test_ds, batch_size=1024, shuffle=False, num_workers=4)



X = []
y = []
for (img_batch,label) in train_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    X.append(out)
    y.append(label)
X = np.concatenate(X)
y = np.concatenate(y)
for i in range(0,10):
    image_class_mask = (y == i)
    class_mean_set.append(np.mean(X[image_class_mask],axis=0))
for (img_batch,label) in test_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    predictions = []
    for single_image in out:
        distance = single_image - class_mean_set
        norm = np.linalg.norm(distance,ord=2,axis=1)
        pred = np.argmin(norm)
        predictions.append(pred)
    predictions = torch.tensor(predictions)
    correct += (predictions.cpu() == label.cpu()).sum()
    total += label.shape[0]
print(f"Accuracy at 0 {correct/total}")
accuracy_history.append(correct/total)

print(f"Average incremental accuracy {np.mean(accuracy_history)}")