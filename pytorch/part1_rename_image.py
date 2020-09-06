import os


class BatchRename():
    def __init__(self):
        self.path = r'D:\C盘文件和桌面程序\桌面文件\Desktop\tensorflow学习资料\labelimg\images'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.png') or item.endswith('.jpeg') or item.endswith('.gif'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '0' + format(str(i), '0>3s') + '.jpg')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s..' % (src, dst))
                    i += 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
