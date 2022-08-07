import sys
import os
import pickle
import cv2
import numpy as np
from copy import deepcopy
import torch

from dataset.ucf24 import UCF24, UCF24_CLASSES
from utils.box_ops import rescale_bboxes
from .utils import iou2d, pr_to_ap, nms3dt, iou3dt
from .utils import load_frame_detections, build_tubes


class UCFEvaluator(object):
    def __init__(self,
                 cfg=None,
                 len_clip=1,
                 img_size=224,
                 thresh=0.5,
                 transform=None,
                 metric='frame_map',
                 save_dir=None):
        self.cfg = cfg
        self.len_clip = len_clip
        self.thresh = thresh
        self.transform = transform
        self.metric = metric
        self.save_dir = save_dir
        self.num_classes = 24
        self.class_names = UCF24_CLASSES
        self.frame_map = 0.0
        self.video_map = {'@0.5': 0.0,
                          '@0.5:0.95': 0.0}

        # dataset
        self.dataset = UCF24(
            cfg=cfg,
            img_size=img_size,
            len_clip=len_clip,
            is_train=False,
            transform=None,
            debug=False
            )

    @torch.no_grad()
    def inference(self, device, model, image_list):
        # preprocess
        image_list, _ = self.transform(image_list)

        video_clip = torch.stack(image_list, dim=1)     # [3, T, H, W]
        video_clip = video_clip.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

        # inference
        scores, labels, bboxes = model(video_clip)

        return scores, labels, bboxes


    def evaluate(self, model):
        device = model.device

        # inference
        num_videos = self.dataset.num_videos
        for index in range(num_videos):
            # load a video
            video_name = self.dataset.load_video(index)
            num_frames = self.dataset.nframes[video_name]

            if (index) % 50 == 0:
                print('Video {:d}/{:d}: {}'.format(index+1, num_videos, video_name))

            # path to video dir
            video_path = os.path.join(self.dataset.image_path, video_name)
            
            # prepare
            image_list = []

            # inference with video stream
            detections = {}
            for fid in range(1, num_frames + 1):
                image_file = os.path.join(video_path, '{:0>5}.jpg'.format(fid))
                cur_frame = cv2.imread(image_file)
                
                assert cur_frame is not None

                orig_h, orig_w = cur_frame.shape[:2]
                orig_size = [orig_w, orig_h]

                image_list.append(cur_frame)
                            
                if len(image_list) < self.len_clip:
                    continue
                elif len(image_list) == self.len_clip:
                    pass
                else:
                    del image_list[0]

                # inference
                scores, labels, bboxes = self.inference(device, model, image_list)

                # rescale
                bboxes = rescale_bboxes(bboxes, orig_size)

                # save per frame detection results
                outputs = {}
                for c in range(self.num_classes):
                    keep = (labels == c)
                    c_scores = scores[keep]
                    c_bboxes = bboxes[keep]
                    output = np.concatenate([c_bboxes, c_scores[..., None]], axis=-1)
                    outputs[c] = output
                detections[fid] = outputs

            # save this video detection results
            os.makedirs(os.path.join(self.save_dir, video_name), exist_ok=True)          
            for i in detections.keys():
                outfile = os.path.join(self.save_dir, video_name, str(i).zfill(5) + '.pkl')

                with open(outfile, 'wb') as file:
                    # print(detections[i])
                    pickle.dump(detections[i], file)

        vlist = self.dataset.video_list

        # load per-frame detections
        frame_detections_file = os.path.join(self.save_dir, 'frame_detections.pkl')
        all_dets = load_frame_detections(self.dataset, vlist, self.save_dir)
        try:
            with open(frame_detections_file, 'wb') as fid:
                pickle.dump(all_dets, fid, protocol=4)
        except:
            print("OverflowError: cannot serialize a bytes object larger than 4 GiB")

        # evaluation
        if self.metric == 'frame_map':
            print('-----------------------------------------')
            print('calculating Frame mAP ...')
            frame_map = self.frameAP(all_dets, vlist)
            self.frame_map = frame_map

        elif self.metric == 'frame_ap_error':
            self.frameAP_error()

        elif self.metric == 'video_map':
            print('-----------------------------------------')
            print('building tubes ...')
            # First, we build tubelets
            build_tubes(self.dataset, self.save_dir)
            
            print('calculating Video mAP ...')
            # Next, we calculate the video mAP@0.5
            video_map = self.videoAP()
            self.video_map['@0.5'] = video_map
            self.video_map['@0.5:0.95'] = video_map


    def frameAP(self, all_dets, video_list):
        results = {}
        # compute AP for each class
        for ilabel, label in enumerate(self.dataset.labels):
            # detections of this class
            detections = all_dets[all_dets[:, 2] == ilabel, :]

            # load ground-truth of this class
            gt = {}
            for iv, v in enumerate(video_list):
                tubes = self.dataset.gttubes[v]

                if ilabel not in tubes:
                    continue

                for tube in tubes[ilabel]:
                    # tube: [T, 5] -> [[fid, x1, y1, x2, y2], ...]
                    for i in range(tube.shape[0]):
                        k = (iv, int(tube[i, 0])) # [video_index, frame_index]
                        
                        # ignore init frames
                        if k[1] < self.len_clip:
                            continue

                        if k not in gt:
                            gt[k] = []
                        gt[k].append(tube[i, 1:5].tolist())

            for k in gt:
                gt[k] = np.array(gt[k])

            # pr will be an array containing precision-recall values
            pr = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)  # precision,recall
            pr[0, 0] = 1.0
            pr[0, 1] = 0.0
            fn = sum([g.shape[0] for g in gt.values()])  # false negatives
            fp = 0  # false positives
            tp = 0  # true positives

            for i, j in enumerate(np.argsort(-detections[:, 3])):
                k = (int(detections[j, 0]), int(detections[j, 1]))
                box = detections[j, 4:8].astype(np.int32)
                ispositive = False

                if k in gt:
                    ious = iou2d(gt[k], box)
                    amax = np.argmax(ious)

                    if ious[amax] >= self.thresh:
                        ispositive = True
                        gt[k] = np.delete(gt[k], amax, 0)

                        if gt[k].size == 0:
                            del gt[k]

                if ispositive:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                pr[i + 1, 0] = float(tp) / float(tp + fp)
                pr[i + 1, 1] = float(tp) / float(tp + fn)

            results[label] = pr

        # display results
        print('>>>>>>>>>>>>> FrameAP_{} <<<<<<<<<<<<<<'.format(self.thresh))
        aps = 100 * np.array([pr_to_ap(results[label]) for label in self.dataset.labels])
        for i, ap in enumerate(aps):
            print('{}: {:.2f}'.format(self.class_names[i], ap))
        frame_mAP = np.mean(aps)
        print()
        print("{:20s} {:8.2f}".format("mAP", frame_mAP))

        return frame_mAP


    def frameAP_error(self):
        eval_file = os.path.join(self.save_dir, "frameAP{:g}ErrorAnalysis.pkl".format(self.thresh))

        if os.path.isfile(eval_file):
            print('load previous linking results...')
            with open(eval_file, 'rb') as fid:
                res = pickle.load(fid)
        else:
            vlist = self.dataset.video_list

            # load per frame detections
            frame_detections_file = os.path.join(self.save_dir, 'frame_detections.pkl')
            if os.path.isfile(frame_detections_file):
                print('load frameAP pre-result')
                with open(frame_detections_file, 'rb') as fid:
                    alldets = pickle.load(fid)
            else:
                alldets = load_frame_detections(self.dataset, vlist, self.save_dir)
                with open(frame_detections_file, 'wb') as fid:
                    pickle.dump(alldets, fid)
            res = {}
            # alldets: list of numpy array with <video_index> <frame_index> <ilabel> <score> <x1> <y1> <x2> <y2>
            # compute AP for each class
            print(len(self.dataset.labels))
            for ilabel, label in enumerate(self.dataset.labels):
                # detections of this class
                detections = alldets[alldets[:, 2] == ilabel, :]

                gt = {}
                othergt = {}
                labellist = {}

                # iv,v : 0 Basketball/v_Basketball_g01_c01
                for iv, v in enumerate(vlist):
                    # tubes: dict {ilabel: (list of)<frame number> <x1> <y1> <x2> <y2>}
                    tubes = self.dataset.gttubes[v]
                    # labellist[iv]: label list for v
                    labellist[iv] = tubes.keys()

                    for il in tubes:
                        # tube: list of <frame number> <x1> <y1> <x2> <y2>
                        for tube in tubes[il]:
                            for i in range(tube.shape[0]):
                                # k: (video_index, frame_index)
                                k = (iv, int(tube[i, 0]))
                                if il == ilabel:
                                    if k not in gt:
                                        gt[k] = []
                                    gt[k].append(tube[i, 1:5].tolist())
                                else:
                                    if k not in othergt:
                                        othergt[k] = []
                                    othergt[k].append(tube[i, 1:5].tolist())

                for k in gt:
                    gt[k] = np.array(gt[k])
                for k in othergt:
                    othergt[k] = np.array(othergt[k])

                dupgt = deepcopy(gt)

                # pr will be an array containing precision-recall values and 4 types of errors:
                # localization, classification, timing, others
                pr = np.empty((detections.shape[0] + 1, 6), dtype=np.float32)  # precision, recall
                pr[0, 0] = 1.0
                pr[0, 1:] = 0.0

                fn = sum([g.shape[0] for g in gt.values()])  # false negatives
                fp = 0  # false positives
                tp = 0  # true positives
                EL = 0  # localization errors
                EC = 0  # classification error: overlap >=0.5 with an another object
                EO = 0  # other errors
                ET = 0  # timing error: the video contains the action but not at this frame

                for i, j in enumerate(np.argsort(-detections[:, 3])):
                    k = (int(detections[j, 0]), int(detections[j, 1]))
                    box = detections[j, 4:8]
                    ispositive = False

                    if k in dupgt:
                        if k in gt:
                            ious = iou2d(gt[k], box)
                            amax = np.argmax(ious)
                        if k in gt and ious[amax] >= self.thresh:
                            ispositive = True
                            gt[k] = np.delete(gt[k], amax, 0)
                            if gt[k].size == 0:
                                del gt[k]
                        else:
                            EL += 1

                    elif k in othergt:
                        ious = iou2d(othergt[k], box)
                        if np.max(ious) >= self.thresh:
                            EC += 1
                        else:
                            EO += 1
                    elif ilabel in labellist[k[0]]:
                        ET += 1
                    else:
                        EO += 1
                    if ispositive:
                        tp += 1
                        fn -= 1
                    else:
                        fp += 1

                    pr[i + 1, 0] = float(tp) / float(tp + fp)  # precision
                    pr[i + 1, 1] = float(tp) / float(tp + fn)  # recall
                    pr[i + 1, 2] = float(EL) / float(tp + fp)
                    pr[i + 1, 3] = float(EC) / float(tp + fp)
                    pr[i + 1, 4] = float(ET) / float(tp + fp)
                    pr[i + 1, 5] = float(EO) / float(tp + fp)

                res[label] = pr

            # save results
            with open(eval_file, 'wb') as fid:
                pickle.dump(res, fid)

        # display results
        AP = 100 * np.array([pr_to_ap(res[label][:, [0, 1]]) for label in self.dataset.labels])
        othersap = [100 * np.array([pr_to_ap(res[label][:, [j, 1]]) for label in self.dataset.labels]) for j in range(2, 6)]

        EL = othersap[0]
        EC = othersap[1]
        ET = othersap[2]
        EO = othersap[3]
        # missed detections = 1 - recall
        EM = 100 - 100 * np.array([res[label][-1, 1] for label in self.dataset.labels])

        LIST = [AP, EL, EC, ET, EO, EM]

        print('Error Analysis')

        print("")
        print("{:20s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format('label', '   AP   ', '  Loc.  ', '  Cls.  ', '  Time  ', ' Other ', ' missed '))
        print("")
        for il, label in enumerate(self.dataset.labels):
            print("{:20s} ".format(label) + " ".join(["{:8.2f}".format(L[il]) for L in LIST]))

        print("")
        print("{:20s} ".format("mean") + " ".join(["{:8.2f}".format(np.mean(L)) for L in LIST]))
        print("")


    def videoAP(self):
        # video name list
        vlist = self.dataset.video_list
        # load detections
        # all_dets = for each label in 1..nlabels, list of tuple (v,score,tube as Kx5 array)
        all_dets = {ilabel: [] for ilabel in range(len(self.dataset.labels))}
        for v in vlist:
            tubename = os.path.join(self.save_dir, v + '_tubes.pkl')
            if not os.path.isfile(tubename):
                print("ERROR: Missing extracted tubes " + tubename)
                sys.exit()

            with open(tubename, 'rb') as fid:
                tubes = pickle.load(fid)
                
            for ilabel in range(len(self.dataset.labels)):
                ltubes = tubes[ilabel]
                idx = nms3dt(ltubes, 0.3)
                all_dets[ilabel] += [(v, ltubes[i][1], ltubes[i][0]) for i in idx]

        # compute AP for each class
        res = {}
        for ilabel in range(len(self.dataset.labels)):
            detections = all_dets[ilabel]
            # load ground-truth
            gt = {}
            for v in vlist:
                tubes = self.dataset.gttubes[v]

                if ilabel not in tubes:
                    continue

                gt[v] = tubes[ilabel]

                if len(gt[v]) == 0:
                    del gt[v]

            # precision,recall
            pr = np.empty((len(detections) + 1, 2), dtype=np.float32)
            pr[0, 0] = 1.0
            pr[0, 1] = 0.0

            fn = sum([len(g) for g in gt.values()])  # false negatives
            fp = 0  # false positives
            tp = 0  # true positives

            for i, j in enumerate(np.argsort(-np.array([dd[1] for dd in detections]))):
                v, score, tube = detections[j]
                ispositive = False

                if v in gt:
                    ious = [iou3dt(g, tube) for g in gt[v]]
                    amax = np.argmax(ious)
                    if ious[amax] >= self.thresh:
                        ispositive = True
                        del gt[v][amax]
                        if len(gt[v]) == 0:
                            del gt[v]

                if ispositive:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1

                pr[i + 1, 0] = float(tp) / float(tp + fp)
                pr[i + 1, 1] = float(tp) / float(tp + fn)

            res[self.dataset.labels[ilabel]] = pr

        # display results
        print('>>>>>>>>>>>>> VideoAP_{} <<<<<<<<<<<<<<'.format(self.thresh))
        aps = 100 * np.array([pr_to_ap(res[label]) for label in self.dataset.labels])
        for i, ap in enumerate(aps):
            print('{}: {:.2f}'.format(self.class_names[i], ap))
        print()
        video_map = np.mean(aps)

        print("{:20s} {:8.2f}".format("mAP", video_map))

        return video_map


    def videpAP_050_095(self):
        ap = 0
        for i in range(10):
            self.thresh = 0.5 + 0.05 * i
            ap += self.videoAP()
        ap = ap / 10.0
        print('VideoAP_0.50:0.95 \n')
        print("\n{:20s} {:8.2f}\n\n".format("mAP", ap))


if __name__ == "__main__":
    pass
