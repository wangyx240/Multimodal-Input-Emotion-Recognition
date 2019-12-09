import reconmodels
from sklearn.metrics import confusion_matrix
from recon_data_generator import transform, clean_video_list, get_samples
from moviepy.editor import VideoFileClip
import numpy as np
import torch
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import re
import recon_data_generator
import recon_losses_R
import recon_metrics_R
import yaml
import time
import warnings


if __name__ == '__main__':
    warnings.filterwarnings("ignore")    # todo

    torch.cuda.empty_cache()

    with open('config.yaml') as f:
        config = yaml.load(f)

    eval_frame_data_set = recon_data_generator.FrameDataSet(config, status="eval")
    eval_frame_train_loader = torch.utils.data.DataLoader(eval_frame_data_set, batch_size=64,
                                                           shuffle=True, num_workers=0)
    print('number of samples in EVAL data set : ', eval_frame_data_set.__len__())
    # rnn_model = reconmodels.RecurrentModel(reconmodels.CombinedModel(reconmodels.AudioModel(), reconmodels.resnet50withcbam()))
    image_model = reconmodels.resnet50withcbam()
    # print("NUMBER OF PARAMETERS AUDIO & VIDEO: {}".format(sum(x.numel() for x in rnn_model.combine_model.parameters())))

    image_model.load_state_dict(torch.load(config['save_path'] + 'imgbest.pth'))
    # rnn_model.eval()

    # rnn_model.cuda()
    # get function handles of loss and metrics
    image_model.cuda()
    # ratio = 4
    print('*' * 20)

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

            torch.cuda.empty_cache()
            # print(label.shape)
            label = label.to(device=torch.cuda.current_device(), dtype=torch.long)
            out = image_model.forward(image)
            _, idx = torch.max(out, 1, keepdim=True)
            if i ==0 :
                y_pred = idx.squeeze().cpu().numpy()
                y_true = label.cpu().numpy()
            else:
                y_pred = np.concatenate((y_pred, idx.squeeze().cpu().numpy()))
                y_true = np.concatenate((y_true, label.cpu().numpy()))

            # print(out.shape, label.shape)
            acc, num_acc = recon_metrics_R.accuracy(out, label)
            print(acc)
            print(num_acc)
            total_eval_sample += num_acc
            total_eval_correct += acc
    mat = confusion_matrix(y_true, y_pred)
    print('total acc: ', total_eval_correct.__float__()/torch.tensor(total_eval_sample).float())
    label_name = ['neutral','calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    mat = pd.DataFrame(mat,index=label_name, columns=label_name)

    print(mat)
    fig = plt.figure(figsize=(20,14))
    sn.heatmap(mat,annot=True)
    plt.show()
    fig.savefig('confusionmat.jpg')


