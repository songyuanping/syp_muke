import torch
import visdom, time, torchvision
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}  # "sq...":0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)
        # image,label
        self.images, self.labels = self.load_csv('images.csv')
        if mode == 'train':
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.gif'))
            print('total images:',len(images))
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2] # 获得bulbasaur
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)
        # 对每张图片的路径和标签分别进行保存
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)
    # 将数据进行去标准化，以便在visdom上进行显示
    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hat = (x-mean)/std
        # x = x_hat*std + mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __getitem__(self, index):
        # index~[0~len(images))
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        img, label = self.images[index], self.labels[index]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path => image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            # 图片旋转一个较小的角度
            transforms.RandomRotation(15),
            # 对图片从中心进行裁剪
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label


def main():
    batch_size = 32
    viz = visdom.Visdom()

    # 加载图片方式1
    tf = transforms.Compose([
                    transforms.Resize((100,100)),
                    transforms.ToTensor(),
    ])
    db = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    # num_workers并行读取图片
    loader = DataLoader(db, batch_size=32, shuffle=True,num_workers=8)
    # 打印出类的编码信息
    print(db.class_to_idx)

    for x,y in loader:
        viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)


    # 加载图片方式2
    # db = Pokemon('pokemon', 100, 'train')
    # x, y = next(iter(db))
    # print('sample:', x.shape, y.shape, y)
    # # 对数据进行去标准化，以便显示
    # viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    # loader = DataLoader(db, batch_size=batch_size, shuffle=True, num_workers=8)
    # for x, y in loader:
    #     viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #     time.sleep(10)


if __name__ == '__main__':
    main()
