import numpy as np
import torch
from moviepy.editor import VideoFileClip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import os

import re
#
# Video_path = Path('./dataset/RAVDESS/')
# Tracking_path = Path('./dataset/RAVDESS/FacialTracking_Actors_01-24/')

'''
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
'''

def transform(img):
    return np.asarray(Image.fromarray(img).crop((256,0,1024,720)).resize((192,180)))


def clean_video_list(video_path):
    video_file_list = os.listdir(video_path)
    pattern = r'01.+'
    return [x for i, x in enumerate(video_file_list) if re.match(pattern, x)]


def get_samples(sample_name):     # every pixel / 255   ???
    clip = VideoFileClip(sample_name)
    audio = np.array(list(clip.audio.set_fps(16000).iter_frames())).mean(1)
    count = 0
    audio_sample_tuple = []
    frame_sample_tuple = []
    label_tuple = []
    sub_clip_frame = []
    sub_clip_audio = []
    frame_count = 0
    capture_size = True
    for frame in clip.iter_frames():
        frame_count += 1
        frame = transform(frame)
        frame = frame / 255.
        frame_audio = audio[count * 16000 // 30: count * 16000 // 30 + 533]
        pattern = r'.+?-.+?-(.+?)-(.+?)-'
        label = re.match(pattern, sample_name).groups()

        sub_clip_audio.append(frame_audio)
        sub_clip_frame.append(np.moveaxis(np.moveaxis(frame,-1,0),-1,-2))
        if frame_count == 15:
            frame_count = 0
            if capture_size:
                audio_size = np.size(sub_clip_audio)
                capture_size = False
            # np.concatenate((np.expand_dims(np.array(sub_clip_frame), 0),
            #                   np.expand_dims(np.array(sub_clip_frame), 0)),0)
            if not capture_size and np.size(sub_clip_audio) != audio_size:
                break
            audio_sample_tuple.append(sub_clip_audio)
            frame_sample_tuple.append(sub_clip_frame)
            # label_tuple.append([np.float64(label[0]), np.float64(label[1])])
            label_tuple.append(np.float64(label[0]))
            sub_clip_frame = []
            sub_clip_audio = []
        count += 1
    return np.array(audio_sample_tuple),np.array(frame_sample_tuple), np.array([i - 1 for i in label_tuple])


class FrameDataSet(Dataset):                # return audio(1,15,533) frame(1,15,3,720,1280) label(1,2)
    def __init__(self, config, status='train'):
        self.directory = config['directory']
        pattern = r'Video_Speech_Actor_.+'
        self.dir_list = [i for i in os.listdir(self.directory) if re.match(pattern, i)]
        self.full_file_name_list = []
        self.full_data_set_audio = torch.tensor([], dtype=torch.float64)
        self.full_data_set_frame = torch.tensor([], dtype=torch.float64)
        self.full_label_set = torch.tensor([], dtype=torch.float64)
        for i in self.dir_list:
            # current_list = clean_video_list(self.directory + i + '/')
            current_list = clean_video_list(self.directory + i + '/Actor_' + i[-2:] + '/')
            for j in current_list:
                self.full_file_name_list.append((self.directory + i + '/Actor_' + i[-2:] + '/' + j))
        portion = round(config['portion']*round(len(self.full_file_name_list)))
        if status == 'train':
            self.full_file_name_list = self.full_file_name_list[0:portion]
        elif status == 'eval':
            self.full_file_name_list = self.full_file_name_list[portion:]
        else:
            print('??????????????????WTF/????????????')
# count = 1
# for file_name in self.full_file_name_list:
#
#     temp_audio, temp_frame, temp_label = get_samples(file_name)
#     self.full_data_set_audio = torch.cat((self.full_data_set_audio, torch.tensor(temp_audio)), 0)
#     self.full_data_set_frame = torch.cat((self.full_data_set_frame, torch.tensor(temp_frame)), 0)
#     self.full_label_set = torch.cat((self.full_label_set, torch.tensor(temp_label)), 0)

            # if count ==1:
            #     self.full_data_set_audio,self.full_data_set_frame, self.full_label_set = get_samples(file_name)
            #     count +=1
            # else:
            #     temp_audio,temp_frame, temp_label = get_samples(file_name)

            #     self.full_data_set_audio = np.concatenate((self.full_data_set_audio, temp_audio), 0)
            #     self.full_data_set_frame = np.concatenate((self.full_data_set_frame, temp_frame), 0)
            #     self.full_label_set = np.concatenate((self.full_label_set, temp_label), 0)
        # self.full_data_set_audio = torch.tensor(self.full_data_set_audio)
        # self.full_data_set_frame = torch.tensor(self.full_data_set_frame)
        # self.full_label_set = torch.tensor(self.full_label_set)
        # if torch.cuda.is_available():
        #     self.full_data_set_audio = self.full_data_set_audio.cuda()
        #     self.full_data_set_frame = self.full_data_set_frame.cuda()
        #     self.full_label_set = self.full_label_set.unsqueeze(-1).cuda()

        # print(min(self.full_label_set))
        # print(max(self.full_label_set))

    def __getitem__(self, index):
        # file_name = re.search(r'.+(/.+?\.mp4)', str(self.full_file_name_list[index])).groups()[0]
        # pattern = r'.+?-.+?-(.+?)-(.+?)-'
        # label = re.match(pattern, file_name).groups()
        # return str(self.full_file_name_list[index]), label[0], label[1]
        return self.full_file_name_list[index]

#
# return self.full_data_set_audio[index], self.full_data_set_frame[index],
#        self.full_label_set[index]

    def __len__(self):
        return len(self.full_file_name_list)

#
# return self.full_label_set.shape[0]
#



# print(frame_data_set.__len__())
# print(frame_train_loader)
# for _,i,_ in frame_train_loader:
#     print(i)
# def get_samples(sample_name):
#     audio = np.array(list(clip.audio.set_fps(16000).iter_frames()))
#     audio = audio.mean(1).astype(np.float32)
#
#     count = 0
#     sample_tuple = []
#     for frame in clip.iter_frames():
#         frame_audio = audio[count*16000//30: (count+1)*16000//30]
#         pattern = r'.+?-.+?-(.+?)-(.+?)-'
#         label = re.match(pattern, file_name).groups()
#         sample_tuple.append({'frame':frame, 'audio':frame_audio, 'emotion_class':label[0], 'emotion_intensity':label[1]})
#         count += 1
#     return sample_tuple
