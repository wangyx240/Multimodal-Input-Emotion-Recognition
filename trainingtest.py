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



# def get_samples(sample_name):
#     clip = VideoFileClip(sample_name)
#     audio = np.array(list(clip.audio.set_fps(16000).iter_frames()))
#     audio = audio.mean(1).astype(np.float64)
#     count = 0
#     sample_tuple = []
#     for frame in clip.iter_frames():
#         frame_audio = audio[count*16000//30: (count+1)*16000//30]
#         pattern = r'.+?-.+?-(.+?)-(.+?)-'
#         label = re.match(pattern, sample_name).groups()
#         sample_tuple.append([frame_audio,frame, label[0],label[1]])
#         count += 1
#     return sample_tuple

# def train_step(model):
#     loss = 0
#     acc = 0
#     return loss, acc


# def train(model,frame_train_loader,max_epochs, optimizer):
#     sum_loss, sum_acc =0., 0.
#     st = time.time()
#     for epoch_i in range(max_epochs):
#         sum_loss, sum_acc = 0., 0.
#         model.train()
#         # for i, data in enumerate(frame_train_loader):
#             # train_clip = get_samples(data[0][0])
#             # for frame in train_clip:
#             #     model.forward(frame)


if __name__ == '__main__':

    warnings.filterwarnings("ignore")    # todo

    torch.cuda.empty_cache()
    # label_max = 4
    # label_min = 4
    writer = SummaryWriter(log_dir='./results')

    with open('config.yaml') as f:
        config = yaml.load(f)
    frame_data_set = recon_data_generator.FrameDataSet(config)
    frame_train_loader = torch.utils.data.DataLoader(frame_data_set, batch_size=1,
                                                     shuffle=True, num_workers=6)# 0 when use at windows
    eval_frame_data_set = recon_data_generator.FrameDataSet(config, status="eval")
    eval_frame_train_loader = torch.utils.data.DataLoader(eval_frame_data_set, batch_size=1,
                                                          shuffle=True, num_workers=6)
    print('number of samples in TRAIN data set : ', frame_data_set.__len__())
    print('number of samples in EVAL data set : ', eval_frame_data_set.__len__())
    rnn_model = reconmodels.RecurrentModel(reconmodels.CombinedModel(reconmodels.AudioModel(), reconmodels.resnet50withcbam()))
    print("TOTAL NUMBER OF PARAMETERS : {}".format(sum(x.numel() for x in rnn_model.parameters())))
    print("NUMBER OF PARAMETERS AUDIO & VIDEO: {}".format(sum(x.numel() for x in rnn_model.combine_model.parameters())))

    rnn_model.cuda()
    # get function handles of loss and metrics
    loss_plo = []
    max_epoch = 50
    total_step = 0
    ratio = 4
    print('*' * 20)
    best_eval_acc = 0
    for epoch in range(max_epoch):
        # print('*' * 10)
        # running_loss = 0.0
        # running_acc = 0.0
        for i, data in enumerate(frame_train_loader):

            temp_audio, temp_video, temp_label = get_samples(data[0])
            temp_audio = torch.tensor(temp_audio)
            temp_video = torch.tensor(temp_video)
            temp_label = torch.tensor(temp_label)
            for clip_num in range((temp_audio.shape[0] - ratio)//2):
                # print(clip_num,temp_audio.shape[0] // ratio,clip_num * 2, clip_num * 2 + ratio,temp_audio.shape[0])
                audio = temp_audio[clip_num * 2:clip_num * 2 + ratio]
                frame = temp_video[clip_num * 2:clip_num * 2 + ratio]
                label = temp_label[clip_num * 2:clip_num * 2 + ratio]
                rnn_model.optimizer.zero_grad()
                total_step += 1
                # audio,video,label=data
                # audio = torch.tensor([], dtype=torch.float64)
                # frame = torch.tensor([], dtype=torch.float64)
                # label = torch.tensor([], dtype=torch.float64)
                # if torch.max(label)>label_max:
                #     label_max = torch.max(label)
                # if torch.min(label)<label_min:
                #     label_min = torch.min(label)
                audio = audio.cuda()
                frame = frame.cuda()
                label = label.cuda()
                torch.cuda.empty_cache()
                label = label.to(device=torch.cuda.current_device(), dtype=torch.long).squeeze(-1)
                out = rnn_model.forward(audio, frame)
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(out, label)
                # print(out)
                # print(label)
                writer.add_scalar("train loss", loss.item(), total_step)

                acc, _ = recon_metrics_R.accuracy(out, label)
                writer.add_scalar("train acc", acc, total_step)
                # print(torch.backends.cudnn.is_acceptable(loss))
                loss.backward()
                # running_loss += loss.item()
                # running_acc += acc
                rnn_model.optimizer.step()
            # print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(
            #     epoch + 1, running_loss / (len(frame_data_set)), running_acc / len(frame_data_set)
            # ))
        # Evaluate Step
        total_eval_sample = 0
        total_eval_correct = 0
        for i, data in enumerate(eval_frame_train_loader):
            with torch.no_grad():
                temp_audio, temp_video, temp_label = get_samples(data[0])
                temp_audio = torch.tensor(temp_audio)
                temp_video = torch.tensor(temp_video)
                temp_label = torch.tensor(temp_label)
                for clip_num in range((temp_audio.shape[0] - ratio)//2):
                    audio = temp_audio[clip_num * 2:clip_num * 2 + ratio]
                    frame = temp_video[clip_num * 2:clip_num * 2 + ratio]
                    label = temp_label[clip_num * 2:clip_num * 2 + ratio]
                    # rnn_model.optimizer.zero_grad()
                    # total_step += 1
                    audio = audio.cuda()
                    frame = frame.cuda()
                    label = label.cuda()
                    torch.cuda.empty_cache()
                    label = label.to(device=torch.cuda.current_device(), dtype=torch.long).squeeze(-1)
                    out = rnn_model.forward(audio, frame)
                    criterion = torch.nn.CrossEntropyLoss()
                    loss = criterion(out, label)
                    writer.add_scalar("eval loss", loss.item(), total_step)
                    acc, num_acc = recon_metrics_R.accuracy(out, label)
                    total_eval_sample += num_acc
                    total_eval_correct += acc
        if total_eval_correct/total_eval_sample > best_eval_acc:
            best_eval_acc = total_eval_correct/total_eval_sample
        writer.add_scalar("eval acc", total_eval_correct/total_eval_sample, total_step)
        print('epoch{} completed, eval acc : {}, best eval acc : {}'.format(epoch + 1, total_eval_correct/total_eval_sample, best_eval_acc))
        torch.save(rnn_model.state_dict(), config['save_path'] + 'best.pth')
    writer.close()

#     train(rnn_model, frame_train_loader, max_epoch)
# t2 = torch.randn((15, 3, 256, 256))
# t1 = torch.randn((15, 1, 533))
#
# rnn_model = RecurrentModel(CombinedModel(onesAudioModel(), resnet50withcbam()))
# rnn_model.train()
#
# rnn_model.optimizer.zero_grad()
#
# print(t1[0].shape)
# output = rnn_model.forward(t1, t2)
# print(output.shape)
#
# label = torch.ones((15,1,8)).float()
# loss = F.l1_loss(output, label)
# loss.backward()
# rnn_model.optimizer.step()
