import os
import torch

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from utils.box_ops import rescale_bboxes
from utils.box_ops import rescale_bboxes

from .cal_mAP import get_mAP
from .utils import bbox_iou


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
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

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


if __name__ == "__main__":
    pass