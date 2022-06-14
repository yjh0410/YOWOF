from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import pickle
import numpy as np
from progress.bar import Bar

from dataset.ACT_utils import nms_tubelets, iou2d


def load_frame_detections(dataset, vlist, inference_dir):
    all_dets = []

    for iv, v in enumerate(vlist): # video_index, video_name

        # load results for each starting frame
        for i in range(1, dataset.nframes[v]+ 1):
            pkl = os.path.join(inference_dir, v, "{:0>5}.pkl".format(i))
            if not os.path.isfile(pkl):
                print("ERROR: Missing extracted tubelets " + pkl)
                sys.exit()

            with open(pkl, 'rb') as fid:
                dets = pickle.load(fid)

            for label in dets:
                # dets: {label:  N, 4+1}, 4+1 : x1, y1, x2, y2, score
                # [N, 4+1]
                box_score = dets[label]
                # [N, 1]
                labels = np.empty((box_score.shape[0], 1), dtype=np.int32)
                labels[:, 0] = label - 1
                score_box = np.concatenate([box_score[:, -1:], box_score[:, :-1]], axis=1)
                label_score_box = np.concatenate([labels, score_box], axis=1)

                num_objs = box_score.shape[0]
                video_index = iv * np.ones((num_objs, 1), dtype=np.float32)  # [N, 1]
                frame_index =  i * np.ones((num_objs, 1), dtype=np.float32)  # [N, 1]

                # [N, 8]
                cdets = np.concatenate([video_index, frame_index, label_score_box], axis=1)
                all_dets.append(cdets)
    
    all_dets = np.concatenate(all_dets, axis=0)

    return np.concatenate(all_dets, axis=0)


def compute_score_one_class(bbox1, bbox2, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbx: <x1> <y1> <x2> <y2> <class score>
    n_bbox1 = bbox1.shape[0]
    n_bbox2 = bbox2.shape[0]
    # for saving all possible scores between each two bbxes in successive frames
    scores = np.zeros([n_bbox1, n_bbox2], dtype=np.float32)
    for i in range(n_bbox1):
        box1 = bbox1[i, :4]
        for j in range(n_bbox2):
            box2 = bbox2[j, :4]
            bbox_iou_frames = bbox_iou(box1, box2, x1y1x2y2=True)
            sum_score_frames = bbox1[i, 4] + bbox2[j, 4]
            mul_score_frames = bbox1[i, 4] * bbox2[j, 4]
            scores[i, j] = w_iou * bbox_iou_frames + w_scores * sum_score_frames + w_scores_mul * mul_score_frames

    return scores


def link_bbxes_between_frames(bbox_list, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbx_list: list of bounding boxes <x1> <y1> <x2> <y2> <class score>
    # check no empty detections
    ind_notempty = []
    nfr = len(bbox_list)
    for i in range(nfr):
        if np.array(bbox_list[i]).size:
            ind_notempty.append(i)
    # no detections at all
    if not ind_notempty:
        return []
    # miss some frames
    elif len(ind_notempty)!=nfr:     
        for i in range(nfr):
            if not np.array(bbox_list[i]).size:
                # copy the nearest detections to fill in the missing frames
                ind_dis = np.abs(np.array(ind_notempty) - i)
                nn = np.argmin(ind_dis)
                bbox_list[i] = bbox_list[ind_notempty[nn]]

    
    detect = bbox_list
    nframes = len(detect)
    res = []

    isempty_vertex = np.zeros([nframes,], dtype=np.bool)
    edge_scores = [compute_score_one_class(detect[i], detect[i+1], w_iou=w_iou, w_scores=w_scores, w_scores_mul=w_scores_mul) for i in range(nframes-1)]
    copy_edge_scores = edge_scores

    while not np.any(isempty_vertex):
        # initialize
        scores = [np.zeros([d.shape[0],], dtype=np.float32) for d in detect]
        index = [np.nan*np.ones([d.shape[0],], dtype=np.float32) for d in detect]
        # viterbi
        # from the second last frame back
        for i in range(nframes-2, -1, -1):
            edge_score = edge_scores[i] + scores[i+1]
            # find the maximum score for each bbox in the i-th frame and the corresponding index
            scores[i] = np.max(edge_score, axis=1)
            index[i] = np.argmax(edge_score, axis=1)
        # decode
        idx = -np.ones([nframes], dtype=np.int32)
        idx[0] = np.argmax(scores[0])
        for i in range(0, nframes-1):
            idx[i+1] = index[i][idx[i]]
        # remove covered boxes and build output structures
        this = np.empty((nframes, 6), dtype=np.float32)
        this[:, 0] = 1 + np.arange(nframes)
        for i in range(nframes):
            j = idx[i]
            iouscore = 0
            if i < nframes-1:
                iouscore = copy_edge_scores[i][j, idx[i+1]] - bbox_list[i][j, 4] - bbox_list[i+1][idx[i+1], 4]

            if i < nframes-1: edge_scores[i] = np.delete(edge_scores[i], j, 0)
            if i > 0: edge_scores[i-1] = np.delete(edge_scores[i-1], j, 1)
            this[i, 1:5] = detect[i][j, :4]
            this[i, 5] = detect[i][j, 4]
            detect[i] = np.delete(detect[i], j, 0)
            isempty_vertex[i] = (detect[i].size==0) # it is true when there is no detection in any frame
        res.append( this )
        if len(res) == 3:
            break
        
    return res


def BuildTubes(dataset, inference_dir):
    # video name list
    vlist = dataset.video_list
    bar = Bar('{}'.format('BuildTubes'), max=len(vlist))
    for iv, v in enumerate(vlist):
        outfile = os.path.join(inference_dir, v + "_tubes.pkl")
        if os.path.isfile(outfile):
            continue

        RES = {}
        nframes = dataset.nframes[v]

        # load detected tubelets
        VDets = {}
        for startframe in range(1, nframes + 1):
            resname = os.path.join(inference_dir, v, "{:0>5}.pkl".format(startframe))
            if not os.path.isfile(resname):
                print("ERROR: Missing extracted tubelets " + resname)
                sys.exit()

            with open(resname, 'rb') as fid:
                VDets[startframe] = pickle.load(fid)

        for ilabel in range(len(dataset.labels)):
            FINISHED_TUBES = []
            CURRENT_TUBES = []  # tubes is a list of tuple (frame, lstubelets)

            # calculate average scores of tubelets in tubes
            def tubescore(tt):
                return np.mean(np.array([tt[i][1][-1] for i in range(len(tt))]))

            for frame in range(1, dataset.nframes[v] + 1):
                # load boxes of the new frame and do nms while keeping Nkeep highest scored
                # [N, 8]
                ltubelets = VDets[frame][ilabel + 1]  # [:,range(4*K) + [4*K + 1 + ilabel]]  Nx(4K+1) with (x1 y1 x2 y2)*K ilabel-score

                ltubelets = nms_tubelets(ltubelets, 0.6, top_k=10)

                # just start new tubes
                if frame == 1:
                    for i in range(ltubelets.shape[0]):
                        CURRENT_TUBES.append([(1, ltubelets[i, :])])
                    continue

                # sort current tubes according to average score
                avgscore = [tubescore(t) for t in CURRENT_TUBES]
                argsort = np.argsort(-np.array(avgscore))
                CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]
                # loop over tubes
                finished = []
                for it, t in enumerate(CURRENT_TUBES):
                    # compute ious between the last box of t and ltubelets
                    last_frame, last_tubelet = t[-1]
                    ious = []
                    offset = frame - last_frame
                    if offset < K:
                        nov = K - offset
                        ious = sum([iou2d(ltubelets[:, 4 * iov:4 * iov + 4], last_tubelet[4 * (iov + offset):4 * (iov + offset + 1)]) for iov in range(nov)]) / float(nov)
                    else:
                        ious = iou2d(ltubelets[:, :4], last_tubelet[4 * K - 4:4 * K])

                    valid = np.where(ious >= 0.5)[0]

                    if valid.size > 0:
                        # take the one with maximum score
                        idx = valid[np.argmax(ltubelets[valid, -1])]
                        CURRENT_TUBES[it].append((frame, ltubelets[idx, :]))
                        ltubelets = np.delete(ltubelets, idx, axis=0)
                    else:
                        if offset >= opt.K:
                            finished.append(it)

                # finished tubes that are done
                for it in finished[::-1]:  # process in reverse order to delete them with the right index why --++--
                    FINISHED_TUBES.append(CURRENT_TUBES[it][:])
                    del CURRENT_TUBES[it]

                # start new tubes
                for i in range(ltubelets.shape[0]):
                    CURRENT_TUBES.append([(frame, ltubelets[i, :])])

            # all tubes are not finished
            FINISHED_TUBES += CURRENT_TUBES

            # build real tubes
            output = []
            for t in FINISHED_TUBES:
                score = tubescore(t)

                # just start new tubes
                if score < 0.005:
                    continue

                beginframe = t[0][0]
                endframe = t[-1][0] + K - 1
                length = endframe + 1 - beginframe

                # delete tubes with short duraton
                if length < 15:
                    continue

                # build final tubes by average the tubelets
                out = np.zeros((length, 6), dtype=np.float32)
                out[:, 0] = np.arange(beginframe, endframe + 1)
                n_per_frame = np.zeros((length, 1), dtype=np.int32)
                for i in range(len(t)):
                    frame, box = t[i]
                    for k in range(K):
                        out[frame - beginframe + k, 1:5] += box[4 * k:4 * k + 4]
                        out[frame - beginframe + k, -1] += box[-1]  # single frame confidence
                        n_per_frame[frame - beginframe + k, 0] += 1
                out[:, 1:] /= n_per_frame
                output.append([out, score])
                # out: [num_frames, (frame idx, x1, y1, x2, y2, score)]

            RES[ilabel] = output
        # RES{ilabel:[(out[length,6],score)]}ilabel[0,...]
        with open(outfile, 'wb') as fid:
            pickle.dump(RES, fid)
        Bar.suffix = '[{0}/{1}]:{2}|Tot: {total:} |ETA: {eta:} '.format(
            iv + 1, len(vlist), v, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()


