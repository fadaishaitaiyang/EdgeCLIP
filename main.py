import os,csv
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.nets.gatector_1 import GaTectorBody
from lib.nets.yolo_training import YOLOLoss, weights_init
from lib.utils.callbacks import LossHistory
from lib.dataloader import GaTectorDataset, gatector_dataset_collate
from lib.utils.utils import get_anchors, get_classes
from lib.utils.utils_fit1 import fit_one_epoch

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='arguement')
    parser.add_argument('--train_mode',type=int,default=0)
    # parser.add_argument('--train_dir', type=str, default='./data/goo_dataset/gooreal/train_data/')
    # parser.add_argument('--train_annotation', type=str, default='./data/goo_dataset/gooreal/train.pickle')
    # parser.add_argument('--test_dir', type=str, default='./data/goo_dataset/gooreal/test_data/')
    # parser.add_argument('--test_annotation', type=str, default='./data/goo_dataset/gooreal/test.pickle')
    parser.add_argument('--train_dir', type=str, default='./data/goo_dataset/goosynth/GazeDatasetsV2/images/')
    parser.add_argument('--train_annotation', type=str, default='./data/goo_dataset/goosynth/picklefile/goosynth_train_v2_no_segm.pkl')
    parser.add_argument('--test_dir', type=str, default='./data/goo_dataset/goosynth/test/images/')
    parser.add_argument('--test_annotation', type=str, default='./data/goo_dataset/goosynth/picklefile/goosynth_test_v2_no_segm.pkl')
    
    args=parser.parse_args()
    
    train_mode=args.train_mode
    
    train_dir = args.train_dir
    train_annotation = args.train_annotation
    test_dir = args.test_dir
    test_annotation = args.test_annotation
    #Create performence file
    if train_mode==0:
        if not os.path.exists('./data/logsSynth_1'):
            os.mkdir('./data/logsSynth_1')
            table_head=['AUC','Dist','Ang']
            f=open('./data/logsSynth_1/performance_1.csv','a+')
            writer=csv.writer(f)
            writer.writerow(table_head)
    Cuda = True
    classes_path = './data/anchors/voc_classes.txt'
    if train_mode==0:
        anchors_path = './data/anchors/yolo_anchors.txt'
        anchors_mask = [[0, 1, 2]]
    if train_mode==1:
        anchors_path = './data/anchors/yolo_anchors_3.txt'
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model_path = ''
    input_shape = [224, 224]
    Cosine_lr = True
    label_smoothing = 0

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 64
    Freeze_lr = 1e-4

    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 64
    Unfreeze_lr = 1e-4
    Freeze_Train = True
    num_workers = 8
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    model = GaTectorBody(anchors_mask, num_classes,train_mode)

    weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    if train_mode == 0:
        if not os.path.exists('./data/logsSynth_1/'):
            os.mkdir('./data/logsSynth_1/')
        loss_history = LossHistory("./data/logsSynth_1/")
    if train_mode==1:
        if not os.path.exists('./data/logsReal/'):
            os.mkdir('./data/logsReal/')
        loss_history=LossHistory("./data/logsReal/")

    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = GaTectorDataset(train_dir, train_annotation, input_shape, num_classes, train_mode,train=True)
        val_dataset = GaTectorDataset(test_dir, test_annotation, input_shape, num_classes, train_mode,train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=gatector_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=gatector_dataset_collate)

        epoch_step = len(train_dataset) // batch_size
        epoch_step_val = len(val_dataset) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The data set is too small for training. Please expand the data set.")

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, train_mode, Cuda)
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = GaTectorDataset(train_dir, train_annotation, input_shape, num_classes, train_mode,train=True)
        val_dataset = GaTectorDataset(test_dir, test_annotation, input_shape, num_classes, train_mode,train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=gatector_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=gatector_dataset_collate)

        epoch_step = len(train_dataset) // batch_size
        epoch_step_val = len(val_dataset) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The data set is too small for training. Please expand the data set.")

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, train_mode,Cuda)
            lr_scheduler.step()
