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
"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch

from misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam


# if __name__ == '__main__':
#     # Get params
#     target_example = 0  # Snake
#     (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
#         get_example_params(target_example)
#     # Grad cam
#     grad_cam = GradCam(pretrained_model, target_layer=11)
#     # Generate cam mask
#     cam = grad_cam.generate_cam(prep_img, target_class)
#     # Save mask
#     save_class_activation_images(original_image, cam, file_name_to_export)
#     print('Grad cam completed')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")    # todo

    torch.cuda.empty_cache()

    with open('config.yaml') as f:
        config = yaml.load(f)

    eval_frame_data_set = recon_data_generator.FrameDataSet(config, status="eval")
    eval_frame_train_loader = torch.utils.data.DataLoader(eval_frame_data_set, batch_size=1, shuffle=True, num_workers=0)
    print('number of samples in EVAL data set : ', eval_frame_data_set.__len__())
    image_model = reconmodels.resnet50withcbam()
    image_model.load_state_dict(torch.load(config['save_path'] + 'imgbest.pth'))

    image_model.cuda()
    image_model.eval()
    print('*' * 20)

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
            grad_cam = GradCam(image_model, target_layer=4)
            # Generate cam mask
            #cam = grad_cam.generate_cam(prep_img, target_class)
            # Save mask
            #save_class_activation_images(original_image, cam, file_name_to_export)
            #print('Grad cam completed')





            out = image_model.forward(image)










            _, idx = torch.max(out, 1, keepdim=True)
            # if i ==0 :
            #     y_pred = idx.squeeze().cpu().numpy()
            #     y_true = label.cpu().numpy()
            # else:
            #     y_pred = np.concatenate((y_pred, idx.squeeze().cpu().numpy()))
            #     y_true = np.concatenate((y_true, label.cpu().numpy()))

            # print(out.shape, label.shape)
            acc, num_acc = recon_metrics_R.accuracy(out, label)
            total_eval_sample += num_acc
            total_eval_correct += acc
    #mat = confusion_matrix(y_true, y_pred)
    print('total acc: ', total_eval_correct.__float__()/torch.tensor(total_eval_sample).float())

    # label_name = ['neutral','calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # mat = pd.DataFrame(mat,index=label_name, columns=label_name)
    # fig = plt.figure(figsize=(20,14))
    # sn.heatmap(mat,annot=True)
    # plt.show()
    # fig.savefig('confusionmat.jpg')


