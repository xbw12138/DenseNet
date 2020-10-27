import warnings

warnings.filterwarnings("ignore")
import sys
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
from torchvision import transforms as transforms
import argparse
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder

from DenseNet import DenseNet121, DenseNet169, DenseNet201, DenseNet161

time_origin = datetime.datetime.now()
time_now = time_origin.strftime('%Y-%m-%d-%H-%M-%S')
time_now_log = time_origin.strftime('%Y.%m.%d')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def main():
    #python main.py --num_classes 2 --dataset_validation Photos/Validation/class-2/400X --dataset_test Photos/Test/class-2/400X --dataset_train Photos/Augmentation_2_400X_224_32 --size 224 train
    parser = argparse.ArgumentParser(description="Densenet for BreaKhis")
    parser.add_argument("action", type=str, help="输入train or test")
    parser.add_argument('--lr', default=0.001, type=float, help='学习率')
    parser.add_argument('--num_classes', default=8, type=int, help='分类')
    parser.add_argument('--epoch', default=100, type=int, help='训练轮数')
    parser.add_argument('--dataset_train', default='Photos/Augmentation_2_40X_224_64', type=str, help='训练集路径')
    parser.add_argument('--dataset_validation', default='Photos/Validation', type=str, help='验证集集路径')
    parser.add_argument('--dataset_test', default='Photos/Test', type=str, help='测试集路径')
    parser.add_argument('--train_batch_size', default=64, type=int, help='训练集批数')
    parser.add_argument('--validation_batch_size', default=1, type=int, help='验证集批数')
    parser.add_argument('--test_batch_size', default=1, type=int, help='测试集批数')
    parser.add_argument('--split_num', default=64, type=int, help='随机分割图像个数')
    parser.add_argument('--size', default=256, type=int, help='训练图像尺寸')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='是否用GPU训练')
    parser.add_argument('--ckpt', type=str, default="model.pth", help='训练模型保存文件,以及测试集需要模型，默认model.pth')
    parser.add_argument('--recover', type=str2bool, default='False', help='中断后继续跑')
    args = parser.parse_args()#解析参数

    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.action = config.action
        self.lr = config.lr
        self.num_classes = config.num_classes
        self.epochs = config.epoch
        self.dataset_train = config.dataset_train
        self.dataset_validation = config.dataset_validation
        self.dataset_test = config.dataset_test
        self.train_batch_size = config.train_batch_size
        self.validation_batch_size = config.validation_batch_size
        self.test_batch_size = config.test_batch_size
        self.split_num = config.split_num
        self.size = config.size
        self.cuda = config.cuda
        self.recover = config.recover
        self.ckpt = config.ckpt
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.start_num_epochs = 1
        self.class_label = []

    def load_data(self):
        train_set = torchvision.datasets.ImageFolder(self.dataset_train,
                                                     transform=transforms.Compose([transforms.ToTensor()]))
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True,
                                                        num_workers=2)

        validation_set = torchvision.datasets.ImageFolder(self.dataset_validation,
                                                          transform=transforms.Compose([transforms.ToTensor()]))
        self.validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.validation_batch_size,
                                                             shuffle=True, num_workers=2)

        test_set = torchvision.datasets.ImageFolder(self.dataset_test,
                                                    transform=transforms.Compose([transforms.ToTensor()]))
        self.class_label = test_set.classes

        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.test_batch_size, shuffle=True,
                                                       num_workers=2)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # 中断恢复
        if self.recover:
            checkpoint = torch.load(self.ckpt)
            self.model = checkpoint['model']
            self.optimizer = checkpoint['optimizer']
            self.scheduler = checkpoint['scheduler']
            self.start_num_epochs = checkpoint['epoch'] + 1
        else:
            if self.size == 224:
                model = DenseNet121(num_classes=self.num_classes)
            elif self.size == 256:
                model = DenseNet121(4, 4, self.num_classes)
            elif self.size == 512:
                model = DenseNet121(4, 8, self.num_classes)

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            self.model = model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
            self.start_num_epochs = 1

        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        self.model.train()
        epoch_loss, epoch_acc, step = 0, 0, 0
        for data, target in tqdm(self.train_loader):
            step += 1
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            prediction = torch.max(output, 1)
            epoch_acc += (prediction[1] == target).sum().item() / target.size(0)

        train_loss = epoch_loss / step
        train_acc = epoch_acc / step
        return train_loss, train_acc

    def test(self, dataloader):
        self.model.eval()
        correct_count = 0
        # 数据随机增强
        image_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(self.size)
        ])
        # PIL图像转tensor
        loader = transforms.Compose([transforms.ToTensor()])
        # tensor转PIL图像
        unloader = transforms.ToPILImage()

        target_total = []
        prediction_total = []
        output_total = []
        with torch.no_grad():
            for data, target in tqdm(dataloader):
                image = unloader(data[0].squeeze(0))
                merge = loader(image_aug(image)).unsqueeze(0).to(self.device, torch.float)
                for i in range(self.split_num - 1):
                    merge = torch.cat((merge, loader(image_aug(image)).unsqueeze(0).to(self.device, torch.float)), 0)

                outputs = self.model(merge)  ##outpust = [0.2, 0.8]
                prediction = torch.max(outputs, 1)[1].cpu().numpy().tolist()
                result = max(prediction, key=prediction.count)
                if result == target[0]:
                    correct_count += 1
                target_total.append(target[0])
                prediction_total.append(result)
                output_total.append([prediction.count(index)/len(prediction) for index in range(4)])
            #print(classification_report(target.class_to_idx[0][1], outputs.cpu().numpy(), target_names=target_names))
        self.image_level(target_total, prediction_total)
        self.report_roc(target_total, prediction_total, np.array(output_total))
        self.plot_confusion_matrix(target_total, prediction_total, "Confusion Matrix")

        return correct_count / len(dataloader)


    def run(self):
        self.load_data()
        self.load_model()
        if self.action == "train":
            self.info()
            time.sleep(0.1)
            max_acc, max_validation_acc = 0, 0
            for epoch in range(self.start_num_epochs, self.epochs + 1):
                print('Epoch {}/{}'.format(epoch, self.epochs))
                print('-' * 10)
                train_loss, train_acc = self.train()
                validation_result = self.test(self.validation_loader)
                if max_acc < train_acc or max_validation_acc < validation_result:
                    self.save(epoch)
                if max_acc < train_acc:
                    max_acc = train_acc
                if max_validation_acc < validation_result:
                    max_validation_acc = validation_result

                time.sleep(0.1)
                print("&@& epoch %d loss:%0.4f, train_acc:%0.4f, validation_acc:%0.4f &@&" % (
                    epoch, train_loss, train_acc, validation_result))
                time.sleep(0.1)

                self.scheduler.step(epoch)
        else:
            checkpoint = torch.load(self.ckpt)
            # self.model = checkpoint['model']
            self.model.load_state_dict(checkpoint['model_dict'])
            test_result = self.test(self.test_loader)
            print("acc:%0.4f" % test_result)
            #print(classification_report(y_test, y_predict, target_names=labels))

    def save(self, epoch):
        # 全保存，使用torch.load(args.ckpt)去加载
        # 模型保存
        state = {
            'epoch': epoch,
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'model_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'scheduler_dict': self.scheduler.state_dict()
        }
        if self.recover:
            torch.save(state, "{}".format(self.ckpt))
        else:
            torch.save(state, "model/state_{}_{}_{}".format(time_now, str(self.size), self.ckpt))

    def info(self):
        print("")
        print("### {} (第i次实验)".format(time_now_log))
        print("-" * 50)
        print("* [x] action: {}".format(self.action))
        print("* [x] lr: {}".format(self.lr))
        print("* [x] num_classes: {}".format(self.num_classes))
        print("* [x] epoch: {}".format(self.epochs))
        print("* [x] dataset_train: {}".format(self.dataset_train))
        print("* [x] dataset_validation: {}".format(self.dataset_validation))
        print("* [x] dataset_test: {}".format(self.dataset_test))
        print("* [x] train_batch_size: {}".format(self.train_batch_size))
        print("* [x] validation_batch_size: {}".format(self.validation_batch_size))
        print("* [x] test_batch_size: {}".format(self.test_batch_size))
        print("* [x] split_num: {}".format(self.split_num))
        print("* [x] size: {}".format(self.size))
        print("* [x] cuda: {}".format(self.cuda))
        print("* [x] ckpt: {}".format("model/state_{}_{}_{}".format(time_now, str(self.size), self.ckpt)))
        print("* [x] recover: {}".format(self.recover))
        print("")
        print("")
        print("```")
        print("python "+" ".join(sys.argv))
        print("```")
        print("-" * 50)

    def patient_level(self, y_test, y_predict, p_test):
        # Patient level F1 score
        result = {}
        for true, pred, people in zip(y_test, y_predict, p_test):
            if people in result:
                result[people][0].append(true)
                result[people][1].append(pred)
            else:
                if pred == true:
                    result[people] = [[true],[pred]]
        f1_scores = []
        for people, true_pred in result.items():
            true_class = np.asarray(true_pred[0])
            predicted_class = np.asarray(true_pred[1])
            f1_scores.append(max(f1_score(true_class, predicted_class, pos_label=1, average=None)))
        mean_f1_patient_level = sum(f1_scores)/len(f1_scores)
        print("Patient Level Mean F1 Score: {:.4f}".format(mean_f1_patient_level))

    def image_level(self, y_test, y_predict):
        # Image Level F1 Score
        print("Image Level F1 Score: {:.4f}".format(max(f1_score(y_test, y_predict, pos_label=1, average=None))))

    def report_roc(self, y_test, y_predict, y_predict_proba):
        print(classification_report(y_test, y_predict, target_names=self.class_label))
        # ROC curve
        fpr, tpr, _ = metrics.roc_curve(y_test, y_predict_proba[:,1])
        roc_auc = metrics.auc(fpr, tpr)
        print("ROC curve fpr: {}".format(','.join([str(i) for i in fpr])))
        print("ROC curve tpr: {}".format(','.join([str(i) for i in tpr])))
        print("ROC curve auc: {}".format(str(roc_auc)))
        self.draw(fpr, tpr, roc_auc)
        
    def draw(self, fpr, tpr, roc_auc):    
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Breast Canner Classification ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig("roc-{}.png".format(time_now))

    def plot_confusion_matrix(self, target_total, prediction_total, title):
        plt.figure()
        cm = confusion_matrix(target_total, prediction_total)
        print(cm)
        labels_name = ["benign", "malignant"]
        cm_1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
        plt.imshow(cm_1, interpolation='nearest')    # 在特定的窗口上显示图像
        plt.title(title)    # 图像标题
        plt.colorbar()
        num_local = np.array(range(len(labels_name)))
        plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
        plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i][j], ha="center", va="center", color="white")
        plt.savefig("matrix-{}.png".format(time_now))

if __name__ == '__main__':
    main()