import reconmodels
from recon_data_generator import transform, clean_video_list, get_samples
from moviepy.editor import VideoFileClip
import numpy as np
import torch
import re
import torch.nn.functional as F
import recon_data_generator
import recon_losses_R
import recon_metrics_R
import yaml
import time
from tensorboardX import SummaryWriter
import warnings
import cv2


if __name__ == '__main__':

    warnings.filterwarnings("ignore")    # todo

    torch.cuda.empty_cache()

    writer = SummaryWriter(log_dir='./resultsimg')

    with open('config.yaml') as f:
        config = yaml.load(f)
    frame_data_set = recon_data_generator.FrameDataSet(config)
    frame_train_loader = torch.utils.data.DataLoader(frame_data_set, batch_size=32,
                                                     shuffle=True, num_workers=0)# 0 when use at windows
    eval_frame_data_set = recon_data_generator.FrameDataSet(config, status="eval")
    eval_frame_train_loader = torch.utils.data.DataLoader(eval_frame_data_set, batch_size=32,
                                                           shuffle=True, num_workers=0)
    print('number of samples in TRAIN data set : ', frame_data_set.__len__())
    print('number of samples in EVAL data set : ', eval_frame_data_set.__len__())
    # rnn_model = reconmodels.RecurrentModel(reconmodels.CombinedModel(reconmodels.AudioModel(), reconmodels.resnet50withcbam()))
    image_model = reconmodels.resnet50withcbam()
    print("TOTAL NUMBER OF PARAMETERS : {}".format(sum(x.numel() for x in image_model.parameters())))
    # print("NUMBER OF PARAMETERS AUDIO & VIDEO: {}".format(sum(x.numel() for x in rnn_model.combine_model.parameters())))

    # rnn_model.load_state_dict(torch.load(config['save_path'] + 'best.pth'))
    # rnn_model.eval()
    optimizer = torch.optim.Adam(image_model.parameters(),lr=0.0001)

    # rnn_model.cuda()
    # get function handles of loss and metrics
    image_model.cuda()
    max_epoch = 50
    total_step = 0
    # ratio = 4
    print('*' * 20)
    best_eval_acc = torch.tensor(0).float()

    for epoch in range(max_epoch):
        # print('*' * 10)
        # running_loss = 0.0
        # running_acc = 0.0
        for i, data in enumerate(frame_train_loader):
            # image, label = get_samples(data)
            #print(data[0])
            #cv2.imshow('image', data[0][0])
            #cv2.waitKey(5000)
            image = data[0].unsqueeze(1)
            #print(image.shape)

            label = data[1]
            # temp_audio = torch.tensor(temp_audio)
            image = torch.tensor(image).cuda()
            label = torch.tensor(label)

            optimizer.zero_grad()
            total_step += 1
            # frame = frame.cuda()
            # label = label.cuda()
            torch.cuda.empty_cache()
            # print(label.shape)
            label = label.to(device=torch.cuda.current_device(), dtype=torch.long)
            out = image_model.forward(image)
            #print(out)
            # print(label.shape)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(out, label)

            writer.add_scalar("train loss", loss.item(), total_step)

            acc, _ = recon_metrics_R.accuracy(out, label)
            writer.add_scalar("train acc", acc, total_step)
            # print(torch.backends.cudnn.is_acceptable(loss))
            loss.backward()
            # running_loss += loss.item()
            # running_acc += acc
            optimizer.step()
        # print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(
        #     epoch + 1, running_loss / (len(frame_data_set)), running_acc / len(frame_data_set)
        # ))
        # Evaluate Step
        total_eval_sample = 0
        total_eval_correct = 0

        for i, data in enumerate(eval_frame_train_loader):
            with torch.no_grad():
                image = data[0].unsqueeze(1)
                label = data[1]
                # temp_audio = torch.tensor(temp_audio)
                image = torch.tensor(image).cuda()
                label = torch.tensor(label)

                #optimizer.zero_grad()
                total_step += 1
                torch.cuda.empty_cache()
                # print(label.shape)
                label = label.to(device=torch.cuda.current_device(), dtype=torch.long)
                out = image_model.forward(image)
                #print(out)
                criterion = torch.nn.CrossEntropyLoss()
                # print(out.shape, label.shape)
                loss = criterion(out, label)

                writer.add_scalar("eval loss", loss.item(), total_step)

                acc, num_acc = recon_metrics_R.accuracy(out, label)
                total_eval_sample += num_acc
                total_eval_correct += acc
        if total_eval_correct.__float__()/torch.tensor(total_eval_sample).float() > best_eval_acc:
            best_eval_acc = total_eval_correct.__float__()/torch.tensor(total_eval_sample).float()
            torch.save(image_model.state_dict(), config['save_path'] + 'imgbest.pth')
        writer.add_scalar("eval acc", total_eval_correct.__float__()/torch.tensor(total_eval_sample).float(), total_step)
        print('epoch{} completed,eval num : {},eval corr:{}, eval acc : {}, best eval acc : {}'.format(epoch + 1, torch.tensor(total_eval_sample).float(),total_eval_correct.__float__(),total_eval_correct.__float__()/torch.tensor(total_eval_sample).float(), best_eval_acc))

    writer.close()
