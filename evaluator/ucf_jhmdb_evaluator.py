import os
import torch
import glob
from PIL import Image
import numpy as np
from scipy.io import loadmat

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from utils.box_ops import rescale_bboxes
from utils.box_ops import rescale_bboxes

from .utils import bbox_iou
from .cal_mAP import get_mAP
from .cal_video_mAP import evaluate_videoAP


class UCF_JHMDB_Evaluator(object):
    def __init__(self,
                 device=None,
                 data_root=None,
                 dataset='ucf24',
                 model_name='yowo',
                 img_size=320,
                 len_clip=1,
                 conf_thresh=0.01,
                 iou_thresh=0.5,
                 transform=None,
                 redo=False,
                 gt_folder=None,
                 dt_folder=None,
                 save_path=None):
        self.device = device
        self.data_root = data_root
        self.dataset = dataset
        self.model_name = model_name
        self.img_size = img_size
        self.len_clip = len_clip
        self.transform = transform
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.gt_file = os.path.join(data_root, 'splitfiles/finalAnnots.mat')
        self.testlist = os.path.join(data_root, 'splitfiles/testlist01.txt')

        self.redo = redo
        self.gt_folder = gt_folder
        self.dt_folder = dt_folder
        self.save_path = save_path

        # dataset
        self.testset = UCF_JHMDB_Dataset(
            data_root=data_root,
            dataset=dataset,
            img_size=img_size,
            transform=transform,
            is_train=False,
            len_clip=len_clip,
            sampling_rate=1)
        self.num_classes = self.testset.num_classes


    @torch.no_grad()
    def evaluate_frame_map(self, model, epoch=1, show_pr_curve=False):
        epoch_size = len(self.testset)

        # initalize model
        model.set_inference_mode(mode='stream')

        # inference
        prev_frame_id = ''
        for iter_i, (frame_id, video_clip, target) in enumerate(self.testset):
            # orignal frame size
            orig_size = target['orig_size']  # width, height

            # ex: frame_id = Basketball_v_Basketball_g01_c01_00048.txt
            if iter_i == 0:
                prev_frame_id = frame_id[:-10]
                model.initialization = True

            if frame_id[:-10] != prev_frame_id:
                # a new video
                prev_frame_id = frame_id[:-10]
                model.initialization = True

            # prepare
            video_clip = video_clip.unsqueeze(0).to(self.device) # [B, T, 3, H, W], B=1

            with torch.no_grad():
                # inference
                scores, labels, bboxes = model(video_clip)

                # rescale bbox
                orig_size = target['orig_size'].tolist()
                bboxes = rescale_bboxes(bboxes, orig_size)

                if not os.path.exists('results'):
                    os.mkdir('results')

                if self.dataset == 'ucf24':
                    detection_path = os.path.join('results', 'ucf_detections', self.model_name, 'detections_' + str(epoch), frame_id)
                    current_dir = os.path.join('results', 'ucf_detections',  self.model_name, 'detections_' + str(epoch))
                    os.makedirs(current_dir, exist_ok=True)

                else:
                    detection_path = os.path.join('results', 'jhmdb_detections',  self.model_name, 'detections_' + str(epoch), frame_id)
                    current_dir = os.path.join('results', 'jhmdb_detections',  self.model_name, 'detections_' + str(epoch))
                    os.makedirs(current_dir, exist_ok=True)

                with open(detection_path, 'w+') as f_detect:
                    for score, label, bbox in zip(scores, labels, bboxes):
                        x1 = round(bbox[0])
                        y1 = round(bbox[1])
                        x2 = round(bbox[2])
                        y2 = round(bbox[3])
                        cls_id = int(label) + 1

                        f_detect.write(
                            str(cls_id) + ' ' + str(score) + ' ' \
                                + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')

                if iter_i % 1000 == 0:
                    log_info = "[%d / %d]" % (iter_i, epoch_size)
                    print(log_info, flush=True)

        print('calculating Frame mAP ...')
        metric_list = get_mAP(self.gt_folder, current_dir, self.iou_thresh,
                              self.save_path, self.dataset, show_pr_curve)
        for metric in metric_list:
            print(metric)

        return current_dir


    @torch.no_grad()
    def evaluate_video_map(self, model, epoch=1):
        video_testlist = []
        with open(self.testlist, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.rstrip()
                video_testlist.append(line)

        detected_boxes = {}
        gt_videos = {}

        gt_data = loadmat(self.gt_file)['annot']
        n_videos = gt_data.shape[1]
        print('loading gt tubes ...')
        for i in range(n_videos):
            video_name = gt_data[0][i][1][0]
            if video_name in video_testlist:
                n_tubes = len(gt_data[0][i][2][0])
                v_annotation = {}
                all_gt_boxes = []
                for j in range(n_tubes):  
                    gt_one_tube = [] 
                    tube_start_frame = gt_data[0][i][2][0][j][1][0][0]
                    tube_end_frame = gt_data[0][i][2][0][j][0][0][0]
                    tube_class = gt_data[0][i][2][0][j][2][0][0]
                    tube_data = gt_data[0][i][2][0][j][3]
                    tube_length = tube_end_frame - tube_start_frame + 1
                
                    for k in range(tube_length):
                        gt_boxes = []
                        gt_boxes.append(int(tube_start_frame+k))
                        gt_boxes.append(float(tube_data[k][0]))
                        gt_boxes.append(float(tube_data[k][1]))
                        gt_boxes.append(float(tube_data[k][0]) + float(tube_data[k][2]))
                        gt_boxes.append(float(tube_data[k][1]) + float(tube_data[k][3]))
                        gt_one_tube.append(gt_boxes)
                    all_gt_boxes.append(gt_one_tube)

                v_annotation['gt_classes'] = tube_class
                v_annotation['tubes'] = np.array(all_gt_boxes)
                gt_videos[video_name] = v_annotation

        # set inference mode
        model.set_inference_mode(mode='stream')

        # inference
        print('inference ...')
        for i, line in enumerate(lines):
            line = line.rstrip()
            if i % 50 == 0:
                print('Video: [%d / %d] - %s' % (i, len(lines), line))
            
            # initalize model
            model.initialization = True

            # load a video clip
            img_folder = os.path.join(self.data_root, 'rgb-images', line)

            if self.dataset == 'ucf24':
                label_paths = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
            elif self.dataset == 'jhmdb21':
                label_paths = sorted(glob.glob(os.path.join(img_folder, '*.png')))

            for image_path in label_paths:
                video_split = line.split('/')
                video_class = video_split[0]
                video_file = video_split[1]
                # for windows:
                # img_split = image_path.split('\\')  # ex. [..., 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
                # for linux
                img_split = image_path.split('/')  # ex. [..., 'Basketball', 'v_Basketball_g08_c01', '00070.txt']

                # image name
                img_id = int(img_split[-1][:5])
                max_num = len(os.listdir(img_folder))
                if self.dataset == 'ucf24':
                    img_name = os.path.join(video_class, video_file, '{:05d}.jpg'.format(img_id))
                elif self.dataset == 'jhmdb21':
                    img_name = os.path.join(video_class, video_file, '{:05d}.png'.format(img_id))

                # load video clip
                video_clip = []
                for i in reversed(range(self.len_clip)):
                    # make it as a loop
                    img_id_temp = img_id - i
                    if img_id_temp < 1:
                        img_id_temp = 1
                    elif img_id_temp > max_num:
                        img_id_temp = max_num

                    # load a frame
                    if self.dataset == 'ucf24':
                        path_tmp = os.path.join(self.data_root, 'rgb-images', video_class, video_file ,'{:05d}.jpg'.format(img_id_temp))
                    elif self.dataset == 'jhmdb21':
                        path_tmp = os.path.join(self.data_root, 'rgb-images', video_class, video_file ,'{:05d}.png'.format(img_id_temp))
                    frame = Image.open(path_tmp).convert('RGB')
                    ow, oh = frame.width, frame.height

                    video_clip.append(frame)

                # transform
                video_clip, _ = self.transform(video_clip)
                # List [T, 3, H, W] -> [T, 3, H, W]
                video_clip = torch.stack(video_clip)
                orig_size = [ow, oh]  # width, height

                # prepare
                video_clip = video_clip.unsqueeze(0).to(self.device) # [B, T, 3, H, W], B=1

                with torch.no_grad():
                    # inference
                    scores, labels, bboxes = model(video_clip)

                    # rescale bbox
                    bboxes = rescale_bboxes(bboxes, orig_size)

                    img_annotation = {}
                    for cls_idx in range(self.num_classes):
                        inds = np.where(labels == cls_idx)[0]
                        c_bboxes = bboxes[inds]
                        c_scores = scores[inds]
                        # [n_box, 5]
                        boxes = np.concatenate([c_bboxes, c_scores[..., None]], axis=-1)
                        img_annotation[cls_idx+1] = boxes
                    detected_boxes[img_name] = img_annotation

        iou_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
        print('calculating video mAP ...')
        for iou_th in iou_list:
            per_ap = evaluate_videoAP(gt_videos, detected_boxes, self.num_classes, iou_th, True)
            video_mAP = sum(per_ap) / len(per_ap)
            print('-------------------------------')
            print('V-mAP @ {} IoU:'.format(iou_th))
            print('--Per AP: ', per_ap)
            print('--mAP: ', round(video_mAP, 2))


if __name__ == "__main__":
    pass