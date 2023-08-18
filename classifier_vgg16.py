
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms

import numpy as np
from tqdm import tqdm
from PIL import Image


def train(model, dataloader, otpimizer, criterion, num_epochs, device):
    """
    model:学習モデル
    dataloader:学習、評価データのdataloader
    optimizer:最適化関数
    crierion:ロス関数
    num_epochs:学習回数
    device:CPUかGPUか
    """
    best_acc = 0.0
    # 学習を繰り返す
    for epoch in range(num_epochs):
        # trainとvalを繰り返す
        for phase in ['train', 'val']:
            # モデルを学習モードか評価モードに切り替える
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            # 精度計算用
            loss_sum = 0.0
            acc_sum = 0.0
            total = 0

            # 進捗の表示
            with tqdm(total=len(dataloaders[phase]),unit="batch") as pbar:
                pbar.set_description(f"Epoch[{epoch}/{num_epochs}]({phase})")
                
                # dataloadersからバッチサイズに応じてデータを取得
                for inputs, labels in dataloaders[phase]:
                    # 画像とラベルをGPU/CPUか切り替え
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 予測
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    # ロス算出
                    loss = criterion(outputs, labels)
                    
                    # 予測とラベルの差を使って学習 
                    if phase == 'train':
                        # ここは決まり文句
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # ロス、精度を算出
                    total += inputs.size(0)
                    loss_sum += loss.item() * inputs.size(0)
                    acc_sum += torch.sum(preds == labels.data).item()
                    
                    # 進捗の表示
                    pbar.set_postfix({"loss":loss_sum/float(total),"accuracy":float(acc_sum)/float(total)})
                    pbar.update(1)

            # 1エポックでのロス、精度を算出
            epoch_loss = loss_sum / dataset_sizes[phase]
            epoch_acc = acc_sum / dataset_sizes[phase]
            
            # 一番良い制度の時にモデルデータを保存
            if phase == 'val' and epoch_acc > best_acc:
                print(f"save model epoch:{epoch} loss:{epoch_loss} acc:{epoch_acc}")
                torch.save(model, 'best_model.pth')



if __name__ == '__main__':

    img_dir        = 'C://My_WorkDir//002//AE//img//'
    dataset_dir    = 'C://My_WorkDir//002//AE//data//'
    train_data_dir = dataset_dir + 'train//'
    val_data_dir   = dataset_dir + 'val//'

    size = (224, 224)
    data_transforms  = {
        'train': transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': torchvision.datasets.ImageFolder(train_data_dir, transform=data_transforms['train']),
        'val': torchvision.datasets.ImageFolder(val_data_dir, transform=data_transforms['val'])
    }
            
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=10, shuffle=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=5)
    }

    dataset_sizes = {
        'train': len(image_datasets['train']),
        'val': len(image_datasets['val'])
    }

    class_names = image_datasets['train'].classes
    print('分類種類:', class_names)


    # GPU/CPUが使えるかどうか確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # VGG16の読み込み
    model = models.vgg16(pretrained=True)

    # パラメータの固定
    for param in model.parameters():
        param.requires_grad = False

    # 最後の全結合層を固定しない＞ここだけ学習する
    last_layer = list(model.children())[-1]
    for param in last_layer.parameters():
        param.requires_grad = True


    # 分類数を1000から変更
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # lossを定義
    criterion = nn.CrossEntropyLoss()

    # 色々な最適化関数 lrが学習率 0.001 0.0001などで調整
    optimizer = optim.Adam(model.parameters(), lr=0.0001,)
    # optimizer = optim.SGD(model.parameters(), lr=0.001,)

                    
    num_epochs = 10
    train(model, dataloaders, optimizer, criterion, num_epochs, device)


    # 今回学習したモデルでテスト
    best_model = torch.load('best_model.pth')

    # 対象画像
    filename = val_data_dir+'img_002_0000.png'

    # 読み込み画像をリサイズやtensorなどの方に変換
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # GPU使える場合はGPUを使う
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        best_model.to('cuda')

    # AIの判定
    with torch.no_grad():
        output = best_model(input_batch)
    output = torch.nn.functional.softmax(output[0], dim=0)
    print(output.shape)

    # 出力結果から2種類のうちどれかを数値で取得
    output = output.to('cpu').detach().numpy().copy()
    ind = np.argmax(output)
    print(class_names[ind])