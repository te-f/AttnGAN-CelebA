import csv
import glob
import math
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
import torch.nn.functional as F

import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    d = loadmat(mat_path)

    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


def mk_dir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        pass


def search_best_weights(dir_path):
    weights_path = os.path.join(dir_path, "weights.*.hdf5")
    weights_files = glob.glob(weights_path)
    weights_files = [re.search("weights.+", x).group() for x in weights_files]
    weights_files = [(re.search("weights.[0-9]+", x).group(), x) for x in weights_files]
    weights_files = np.asarray([(re.search("[0-9]+", x[0]).group(), x[1]) for x in weights_files])
    best_weights_path = weights_files[np.argmax(weights_files[:, 0], axis=0), 1]

    return best_weights_path


def prewhiten(x, range=False):
    if range:
        y = 1.0 * x / 128.0 - 1.0
    else:
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def flip(image, random_flip, seed):
    np.random.seed(seed)
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    seed += 1
    return image, seed


def erase(random_erase, image, seed, p=0.5, sl=0.02, sh=0.3, r1=1/3, r2=3):
    np.random.seed(seed)
    erase_img = image
    if random_erase and (np.random.rand() <= p):
        H = image.shape[0]
        W = image.shape[1]
        S = H * W
        Se = np.random.uniform(sl, sh) * S
        seed += 1
        re = np.random.uniform(r1, r2)
        seed += 1
        He = int(round(math.sqrt(Se * re)))
        We = int(round(math.sqrt(Se / re)))
        xe = np.random.randint(W)
        seed += 1
        ye = np.random.randint(H)
        seed += 1
        if (xe + We < W) and (ye + He < H):
            #if ((pointer_coord[0] < xe) or (xe + We < pointer_coord[0])) and ((pointer_coord[1] < ye) or (ye + He < pointer_coord[1])):
            erase_img[ye:ye + He, xe:xe + We, :] = np.random.randint(256)
            #print("erase")
            seed += 1
            #print("no erase")
    seed += 1

    return erase_img, seed


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, array):
        self.name = name
        self.array = array


def sortedStringList(array=[]):
    sortDict = OrderedDict()
    for splitList in array:
        objList = []
        for obj in re.split("(\d+)", splitList):
            try:
                objList.append(int(obj))
            except:
                objList.append(obj)
        sortDict[splitList] = objList

    returnList = []
    for sortObjKey, sortObjValue in sorted(sortDict.items(), key=lambda x:x[1]):
        returnList.append(sortObjKey)

    return returnList

## path　だけとっている (指定したフォルダの中から必要なものだけとれるようにしたい)　TODO
def get_dataset(dataset_name, dataset_dir,list_img_name, img_type=None,use_multi_data = None):
    dataset = []
    
    if dataset_name == 'CELEBA':
        if img_type is None:
            img_dir = 'CelebA/Img/img_align_celeba_png/img_align_celeba_png' #こっち
        else:
            img_dir = f'CelebA/Img/{img_type}'
        print(f"Loading dataset by '{img_dir}'")

        if use_multi_data : ##自分で作成したデータを使うとき
            path = dataset_dir
        else:
            path = os.path.join(dataset_dir, img_dir)
        image_paths = get_image_paths(path,list_img_name)
        dataset.append(ImageClass('CELEBA', image_paths))
        print("Complete load dataset: %d images" % len(dataset[0].array))

    elif dataset_name == 'LFWA':
        path = os.path.join(dataset_dir, 'lfw2/List/lfw_attributes.txt')
        f = open(path)
        next(f)
        next(f)
        att_labels = []
        line = f.readline()
        while line:
            att = line.split('\t')
            nb_att = len(att)
            att[nb_att - 1] = att[nb_att - 1].strip()
            att[0] = att[0].replace(' ', '_')
            att_labels.append(att)
            line = f.readline()
        f.close()

        print('Loading dataset by lfw-deepfunneled')
        path = os.path.join(dataset_dir, 'lfw-deepfunneled')
        image_paths = []
        for lab in att_labels:
            file_name = "{}_{:04d}.jpg".format(lab[0], int(lab[1]))
            img_path = os.path.join(path, lab[0], file_name)
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                print("ERROR: image file not exist")
                sys.exit()
        dataset.append(ImageClass('LFWA', image_paths))

    return dataset


def get_attribute_labels(dataset_name, dataset_dir,nb_attributes = 40,use_my_attdataset_dir = None,dataset_dir_path = None): # 属性をロード、要変更
    att_list = get_attribute_list(dataset_name)
    att_labels = []
    list_img_name = []

    if dataset_name == 'CELEBA':
        if use_my_attdataset_dir is not None:
            print("Loading attribute data by '\\nari\jaxa\zhang\Datasets\CelebA\CelebA_Bias_50000\CelebA_Bias_attr.txt'")
            path = use_my_attdataset_dir
        else:
            print("Loading attribute data by 'CelebA/Anno/list_attr_celeba.txt'")
            path = os.path.join(dataset_dir, 'CelebA/Anno/list_attr_celeba.txt')
        atts = np.loadtxt(path, dtype=[('col1', 'S15'), ('col2', 'i1'), ('col3', 'i1'), ('col4', 'i1'), ('col5', 'i1'),
                                       ('col6', 'i1'), ('col7', 'i1'), ('col8', 'i1'), ('col9', 'i1'), ('col10', 'i1'),
                                       ('col11', 'i1'), ('col12', 'i1'), ('col13', 'i1'), ('col14', 'i1'), ('col15', 'i1'),
                                       ('col16', 'i1'), ('col17', 'i1'), ('col18', 'i1'), ('col19', 'i1'), ('col20', 'i1'),
                                       ('col21', 'i1'), ('col22', 'i1'), ('col23', 'i1'), ('col24', 'i1'), ('col25', 'i1'),
                                       ('col26', 'i1'), ('col27', 'i1'), ('col28', 'i1'), ('col29', 'i1'), ('col30', 'i1'),
                                       ('col31', 'i1'), ('col32', 'i1'), ('col33', 'i1'), ('col34', 'i1'), ('col35', 'i1'),
                                       ('col36', 'i1'), ('col37', 'i1'), ('col38', 'i1'), ('col39', 'i1'), ('col40', 'i1'),
                                       ('col41', 'i1')], skiprows=2)
        for i in tqdm(range(len(att_list))):
            att_labels.append(ImageClass(att_list[i], []))
            for j in range(len(atts)):
                att = 1 if atts[j][i + 1] == 1 else 0
                att_labels[i].array.append(att)
                if i == 0:
                    list_img_name.append((atts[j][0].decode('utf-8')).replace('.jpg', '.png'))#最初のループにファイル名を格納
        print("Complete loading attribute labels")


    elif dataset_name == 'LFWA':
        print("Loading attribute data by 'lfw2/List/lfw_attributes.txt'")
        path = os.path.join(dataset_dir, 'lfw2/List/lfw_attributes.txt')
        f = open(path)
        next(f)
        next(f)
        atts = []
        line = f.readline()
        while line:
            att = line.split('\t')
            nb_att = len(att)
            att = att[2:nb_att]
            att[nb_att - 3] = att[nb_att - 3].strip()
            atts.append(att)
            line = f.readline()
        f.close()
        for i in tqdm(range(len(att_list))):
            att_labels.append(ImageClass(att_list[i], []))
            for j in range(len(atts)):
                att = 1 if float(atts[j][i]) > 0 else 0
                att_labels[i].array.append(att)
        att_labels = lfwa2celeba(att_labels)
        print("Complete loading attribute labels")

    return att_labels[:nb_attributes],list_img_name ##変更　属性数と合わせる


def split_attribute_dataset(dataset_name, dataset_dir, dataset, att_labels, seed=100):
    train_set = []
    test_set = []
    train_labels = []
    test_labels = []

    if dataset_name == 'CELEBA':
        print('Loading partition data by CelebA/Eval/list_eval_partition.txt')
        part = np.loadtxt(os.path.join(dataset_dir, 'CelebA/Eval/list_eval_partition.txt'),
                          dtype=[('col1', 'S15'), ('col2', 'i1')])  ##text ファイルが　202571.jpg 2　のようになっている

        k = 0
        l = 0
        for i in tqdm(range(len(dataset[0].array))):
            if part[i][1] == 2:    #2はtest
                test_set.append(dataset[0].array[i]) ##test_setは画像のパスの list
                test_labels.append([])
                for j in range(len(att_labels)):
                    test_labels[k].append(att_labels[j].array[i])
                k += 1
            else:   #0 or 1 はtest
                train_set.append(dataset[0].array[i])#画像のパスの list
                train_labels.append([])
                for j in range(len(att_labels)):
                    train_labels[l].append(att_labels[j].array[i])
                l += 1
        train_labels = np.asarray(train_labels) #size (182637, 40) 182637枚 40 は属性の数
        test_labels = np.asarray(test_labels) #size (19962, 40) 19962枚 40 は属性の数

    elif dataset_name == 'LFWA':
        np.random.seed(seed)
        train_idx = np.random.choice(np.array(range(len(dataset[0].array))), 6263, replace=False)
        train_set = np.asarray(dataset[0].array)[train_idx]
        for i in range(len(att_labels)):
            train_labels.append(np.asarray(att_labels[i].array)[train_idx])
        test_set = np.delete(dataset[0].array, train_idx)
        for i in range(len(att_labels)):
            test_labels.append(np.delete(att_labels[i].array, train_idx))
        train_labels = np.asarray(train_labels).T
        test_labels = np.asarray(test_labels).T

    return np.asarray(train_set), np.asarray(test_set), train_labels, test_labels


def get_image_paths(facedir,list_img_name): #path だけを取り出す
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        # for img in tqdm(images):
        #     if img in list_img_name:
        #         image_paths.append(os.path.join(facedir, img))
        #     else:
        #         pass
        # image_paths = [os.path.join(facedir, img) for img in tqdm(images)]
        image_paths = [os.path.join(facedir, img) for img in tqdm(list_img_name)]
    # image_paths = sortedStringList(image_paths)
    return image_paths


def split_dataset(dataset_name, dataset_dir, dataset):
    train_set = []
    test_set = []
    if dataset_name == 'CELEBA':
        part = np.loadtxt(os.path.join(dataset_dir, 'CelebA/Eval/list_eval_partition.txt'),
                          dtype=[('col1', 'S15'), ('col2', 'i1')])
        for i in range(len(dataset)):
            test_set.append(dataset[i]) if part[i][1] == 2 else train_set.append(dataset[i])
    elif dataset_name == 'LFWA':
        train_names = np.loadtxt(os.path.join(dataset_dir, 'lfw2/List/peopleDevTrain.txt'),
                                 dtype=[('col1', 'S35'), ('col2', 'i1')], skiprows=1)
        i = 0
        for cls in dataset:
            if (i < 4038 and cls.name == train_names[i][0].decode('utf-8')):
                train_set.append(cls)
                i += 1
            else:
                test_set.append(cls)
    print('Complete split dataset -- train: %d / test: %d' % (len(train_set), len(test_set)))

    return train_set, test_set


def lfwa2celeba(att_labels):
    convert_list = [64, 35, (55, 56), 59, 12,
                    29, 40, 38, 9, 10,
                    20, 11, 34, 19, 48,
                    14, 46, 58, 60, 68,
                    0, 42, 16, 36, 45,
                    50, 63, 39, 28, 61,
                    30, 17, 27, 26, 70,
                    49, 66, 72, 71, (4, 5, 6)]
    att_list = get_attribute_list('LFWA')

    atts = []
    for i in convert_list:
        if type(i) is tuple:
            or_array = np.zeros(len(att_labels[0].array)).astype(bool)
            for j in range(len(i)):
                or_array = np.logical_or(or_array, np.asarray(att_labels[i[j]].array).astype(bool)).astype(int)
            if i[0] == 55:
                name = "Attractive"
            else:
                name = "Young"
        else:
            or_array = att_labels[i].array
            name = att_list[i]

        atts.append(ImageClass(name=name, array=or_array))

    return atts


def get_attribute_list(dataset_name):
    att_list = ''
    if dataset_name == 'CELEBA':
        att_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    elif dataset_name == 'LFWA':
        att_list = ['Male', 'Asian', 'White', 'Black', 'Baby',  # 0
                    'Child', 'Youth', 'Middle Aged', 'Senior', 'Black Hair',  # 5
                    'Blond Hair', 'Brown Hair', 'Bald', 'No Eyewear', 'Eyeglasses',  # 10
                    'Sunglasses', 'Mustache', 'Smiling', 'Frowning', 'Chubby',  # 15
                    'Blurry', 'Harsh Lighting', 'Flash', 'Soft Lighting', 'Outdoor',   # 20
                    'Curly Hair', 'Wavy Hair', 'Straight Hair', 'Receding Hairline', 'Bangs',  # 25
                    'Sideburns', 'Fully Visible Forehead', 'Partially Visible Forehead', 'Obstructed Forehead', 'Bushy Eyebrows',  # 30
                    'Arched Eyebrows', 'Narrow Eyes', 'Eyes Open', 'Big Nose', 'Pointy Nose',  # 35
                    'Big Lips', 'Mouth Closed', 'Mouth Slightly Open', 'Mouth Wide Open', 'Teeth Not Visible',  # 40
                    'No Beard', 'Goatee', 'Round Jaw', 'Double Chin', 'Wearing Hat',  # 45
                    'Oval Face', 'Square Face', 'Round Face', 'Color Photo', 'Posed Photo',  # 50
                    'Attractive Man', 'Attractive Woman', 'Indian', 'Gray Hair', 'Bags Under Eyes',  # 55
                    'Heavy Makeup', 'Rosy Cheeks', 'Shiny Skin', 'Pale Skin', '5 o Clock Shadow',  # 60
                    'Strong Nose-Mouth Lines', 'Wearing Lipstick', 'Flushed Face', 'High Cheekbones', 'Brown Eyes',  # 65
                    'Wearing Earrings', 'Wearing Necktie', 'Wearing Necklace']  # 70
    return att_list


def split_dataset_by_seed(paths, labels, split_rate=0.5, seed=50):
    np.random.seed(seed)
    idx_d1 = np.asarray(range(len(paths)))
    nb_d2 = int(round(len(paths) * split_rate))
    idx_d2 = np.random.choice(idx_d1, nb_d2, replace=False) # random sampling
    idx_d1 = np.delete(idx_d1, idx_d2) #これ何の意味があるかわからん

    paths_d1 = np.asarray(paths)[idx_d1]
    paths_d2 = np.asarray(paths)[idx_d2]
    labels_d1 = np.asarray(labels)[idx_d1]
    labels_d2 = np.asarray(labels)[idx_d2]

    return paths_d1, paths_d2, labels_d1, labels_d2


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_tensor_as_csv(tensor, epoch, save_dir):
    array = tensor.to('cpu').detach().numpy().copy()
    with open(os.path.join(save_dir, '%03d.csv' % epoch), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for arr in array:
            writer.writerows(arr)

def save_acc_as_csv(tensor, num, save_dir):
    array = tensor.to('cpu').detach().numpy().copy()
    with open(os.path.join(save_dir, '%03d.csv' % num), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(array)

def load_tensor_from_csv(epoch, save_dir):
    with open(os.path.join(save_dir, '%03d.csv' % epoch), 'r') as f:
        reader = csv.reader(f)
        tensor = torch.tensor(np.asarray([row for row in reader]).astype('float'), dtype=torch.float32)
    return tensor


class AccuracyCalc(object):
    def __init__(self, nb_attributes, device):
        self.nb_attributes = nb_attributes
        self.device = device
        self.reset()

    def reset(self):
        self.correct = torch.zeros(self.nb_attributes, dtype=torch.float32).to(self.device)
        self.count_inputs = 0

    def update(self, logit, target):
        self.count_inputs += logit.size(1)
        self.correct += sum(torch.argmax(logit.transpose(1, 0), dim=2) == target).type(torch.float32)
        return self.output_accuracy()

    def output_accuracy(self):
        return torch.mean(self.correct / self.count_inputs)

    def output_accuracy_list(self):
        return self.correct / self.count_inputs



class Checkpoint(object):
    def __init__(self, args):
        self.args = args                                                     # ファイルの保存先を確認するために利用
        self.best_loss = float('inf')                                        # 現在までの損失の最小値（初期値は正の無限大）
        self.early_stopping = False if args.stop_patience is None else True  # Early Stoppingを行うかどうか
        self.stop_patience = args.stop_patience                              # 何エポック連続で損失が改善しない場合Early Stoppingするか
        self.count_p = 0                                                     # 何エポック連続で損失が改善していないか

    def set_best_loss(self, best_loss):
        self.best_loss = best_loss  # 学習を途中から再開する場合

    def check(self, loss, model, optimizer, scheduler, optimizer_opt=None, scheduler_opt=None):
        if loss < self.best_loss:
            self.save_model(os.path.join(self.args.m_save_dir, 'checkpoint.pth'), model, optimizer, scheduler, optimizer_opt, scheduler_opt)  # 損失が改善した場合は最良パラメータファイルを更新する
            self.best_loss = loss           # best_lossを更新
            self.count_p = 0                # count_pをリセット
        else:
            self.count_p += 1      # count_pをインクリメント

        if self.early_stopping and (self.count_p > self.stop_patience):
            return False  # Early Stoppingする場合はFalseを返す
        else:
            return True   # Early Stoppingしない場合はTrueを返す

    def save_model(self, save_path, model, optimizer, scheduler, optimizer_opt=None, scheduler_opt=None):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),          # パラメータのstate_dictをまとめたdictを作成する
            'scheduler': scheduler.state_dict()
        }

        if optimizer_opt is not None:
            state['optimizer_opt'] = optimizer_opt.state_dict()
        if scheduler_opt is not None:
            state['scheduler_opt'] = scheduler_opt.state_dict()

        torch.save(state, save_path)                     # 指定パスにdictを保存


class LossAccMeter(object): #loss などを観測
    def __init__(self, nb_attributes):
        self.nb_att = nb_attributes
        self.count_step = 0.0     # 現在までのステップ数
        self.sum_loss = 0.0       # 現在までのlossの総和
        self.count_img = 0.0      # 現在までの画像数（＝バッチサイズ×ステップ数）
        self.count_correct = torch.zeros(nb_attributes)  # 現在までの正解数

        self.tp = torch.zeros(nb_attributes)  # True Positiveの数
        self.tn = torch.zeros(nb_attributes)  # True Negativeの数
        self.fp = torch.zeros(nb_attributes)  # False Positiveの数
        self.fn = torch.zeros(nb_attributes)  # False Negativeの数


    def update(self, outputs, targets=None, loss=None):
        outputs = outputs.detach().cpu()
        #targets = targets.detach().cpu().transpose(0, 1)
        targets = targets.detach().cpu()
        self.count_step += 1                                                # ステップ数をインクリメント
        if targets is not None:
            #self.count_correct += torch.sum(torch.argmax(outputs, dim=2) == targets, dim=1)
            self.count_correct += torch.sum(torch.where(torch.sigmoid(outputs) >= 0.5, 1.0, 0.0) == targets, dim=0)
            ##torch.sigmoid(outputs)で確立を確認できる　## 変更
            #self.count_img += outputs.size(1)
            self.count_img += outputs.size(0)

            predictions = torch.where(torch.sigmoid(outputs) >= 0.5, 1.0, 0.0)
            self.tp += torch.sum((predictions == 1) & (targets == 1), dim=0)
            self.tn += torch.sum((predictions == 0) & (targets == 0), dim=0)
            self.fp += torch.sum((predictions == 1) & (targets == 0), dim=0)
            self.fn += torch.sum((predictions == 0) & (targets == 1), dim=0)

        if loss is not None:
            self.sum_loss += loss.detach().cpu()                                  # 損失（ステップごとの平均値）を加算


    def output_tp_fp_fn(self):
        return {
            'tp': self.tp.tolist(),  # True Positives
            'tn': self.tn.tolist(),  # True Negatives
            'fp': self.fp.tolist(),  # False Positives
            'fn': self.fn.tolist(),  # False Negatives
        }
    # F1スコア
    def output_f1_score(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)  # 0で割るのを避けるために小さな数を加える
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        return f1_score.tolist()

    def output_acc(self):
        acc = self.count_correct / self.count_img  # 現在までの正解率を計算
        if type(acc) == torch.Tensor:
            return acc.tolist()  # Tensorの場合は通常のスカラ値へ変換
        else:
            return acc

    def output_loss(self):
        loss = self.sum_loss / self.count_step  # 現在までのlossの平均値を計算
        if type(loss) == torch.Tensor:
            return loss.item()  # Tensorの場合は通常のスカラ値へ変換
        else:
            return loss
