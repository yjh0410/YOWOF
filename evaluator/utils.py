import sys
import os
import pickle
import numpy as np


def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)


def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""

    xmin = np.maximum(b1[:, 0], b2[:, 0])
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    xmax = np.minimum(b1[:, 2] + 1, b2[:, 2] + 1)
    ymax = np.minimum(b1[:, 3] + 1, b2[:, 3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height


def iou2d(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2"""

    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d(b1, b2)

    return ov / (area2d(b1) + area2d(b2) - ov)


def nms2d(boxes, overlap=0.6):
    """Compute the soft nms given a set of scored boxes,
    as numpy array with 5 columns <x1> <y1> <x2> <y2> <score>
    return the indices of the tubelets to keep
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + 1

    while order.size > 0:
        i = order[0]

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        index = np.where(iou > overlap)[0]
        weight[order[index + 1]] = 1 - iou[index]

        index2 = np.where(iou <= overlap)[0]
        order = order[index2 + 1]

    boxes[:, 4] = boxes[:, 4] * weight

    return boxes


def nms_tubelets(dets, overlapThresh=0.3, top_k=None):
    """Compute the NMS for a set of scored tubelets
    scored tubelets are numpy array with 4K+1 columns, last one being the score
    return the indices of the tubelets to keep
    """

    # If there are no detections, return an empty list
    if len(dets) == 0:
        dets
    if top_k is None:
        top_k = len(dets)

    K = int((dets.shape[1] - 1) / 4)

    # Coordinates of bounding boxes
    x1 = [dets[:, 4 * k] for k in range(K)]
    y1 = [dets[:, 4 * k + 1] for k in range(K)]
    x2 = [dets[:, 4 * k + 2] for k in range(K)]
    y2 = [dets[:, 4 * k + 3] for k in range(K)]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    # area = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = dets[:, -1]
    area = [(x2[k] - x1[k] + 1) * (y2[k] - y1[k] + 1) for k in range(K)]
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + 1
    counter = 0

    while order.size > 0:
        i = order[0]
        counter += 1

        # Compute overlap
        xx1 = [np.maximum(x1[k][i], x1[k][order[1:]]) for k in range(K)]
        yy1 = [np.maximum(y1[k][i], y1[k][order[1:]]) for k in range(K)]
        xx2 = [np.minimum(x2[k][i], x2[k][order[1:]]) for k in range(K)]
        yy2 = [np.minimum(y2[k][i], y2[k][order[1:]]) for k in range(K)]

        w = [np.maximum(0, xx2[k] - xx1[k] + 1) for k in range(K)]
        h = [np.maximum(0, yy2[k] - yy1[k] + 1) for k in range(K)]

        inter_area = [w[k] * h[k] for k in range(K)]
        ious = sum([inter_area[k] / (area[k][order[1:]] + area[k][i] - inter_area[k]) for k in range(K)])
        index = np.where(ious > overlapThresh * K)[0]
        weight[order[index + 1]] = 1 - ious[index]

        index2 = np.where(ious <= overlapThresh * K)[0]
        order = order[index2 + 1]

    dets[:, -1] = dets[:, -1] * weight

    new_scores = dets[:, -1]
    new_order = np.argsort(new_scores)[::-1]
    dets = dets[new_order, :]

    return dets[:top_k, :]


def iou3d(b1, b2):
    """Compute the IoU between two tubes with same temporal extent"""

    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d(b1[:, 1:5], b2[:, 1:5])

    return np.mean(ov / (area2d(b1[:, 1:5]) + area2d(b2[:, 1:5]) - ov))


def iou3dt(b1, b2, spatialonly=False):
    """Compute the spatio-temporal IoU between two tubes"""

    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])

    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0]) + 1

    tube1 = b1[int(np.where(b1[:, 0] == tmin)[0]): int(np.where(b1[:, 0] == tmax)[0]) + 1, :]
    tube2 = b2[int(np.where(b2[:, 0] == tmin)[0]): int(np.where(b2[:, 0] == tmax)[0]) + 1, :]

    return iou3d(tube1, tube2) * (1. if spatialonly else temporal_inter / temporal_union)


def nms3dt(tubes, overlap=0.5):
    """Compute NMS of scored tubes. Tubes are given as list of (tube, score)
    return the list of indices to keep
    """

    if not tubes:
        return np.array([], dtype=np.int32)

    I = np.argsort([t[1] for t in tubes])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0

    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([iou3dt(tubes[ii][0], tubes[i][0]) for ii in I[:-1]])
        I = I[np.where(ious <= overlap)[0]]

    return indices[:counter]


def pr_to_ap(pr):
    """Compute AP given precision-recall
    pr is a Nx2 array with first row being precision and second row being recall
    """

    prdif = pr[1:, 1] - pr[:-1, 1]
    prsum = pr[1:, 0] + pr[:-1, 0]

    return np.sum(prdif * prsum * 0.5)


def bbox_iou(bbox1, bbox2):
    """
        bbox1 : [4] = [x1, y1, x2, y2]
        bbox2 : [4] = [x1, y1, x2, y2]
    """
    tl = np.maximum(bbox1[:2], bbox2[:2])
    br = np.minimum(bbox1[2:], bbox2[2:])
    area_a = np.prod(bbox1[2:] - bbox1[:2], axis=0)
    area_b = np.prod(bbox2[2:] - bbox2[:2], axis=0)

    en = (tl < br).astype(tl.dtype).prod(axis=0)
    area = np.prod(br - tl, axis=1) * en
    iou = area / (area_a + area_b - area)

    return iou


def load_frame_detections(dataset, vlist, inference_dir):
    all_dets = []
    len_clip = dataset.len_clip

    for iv, v in enumerate(vlist): # video_index, video_name

        # load results for each starting frame
        for i in range(len_clip, dataset.nframes[v]+ 1):
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
                labels[:, 0] = label
                score_box = np.concatenate([box_score[:, -1:], box_score[:, :-1]], axis=1)
                label_score_box = np.concatenate([labels, score_box], axis=1)

                num_objs = box_score.shape[0]
                video_index = iv * np.ones((num_objs, 1), dtype=np.float32)  # [N, 1]
                frame_index =  i * np.ones((num_objs, 1), dtype=np.float32)  # [N, 1]

                # [N, 8]
                cdets = np.concatenate([video_index, frame_index, label_score_box], axis=1)
                all_dets.append(cdets)
    
    all_dets = np.concatenate(all_dets, axis=0)

    return all_dets


def build_tubes(dataset, save_dir):
    # video name list
    vlist = dataset.video_list

    for iv, v in enumerate(vlist):
        outfile = os.path.join(save_dir, v + "_tubes.pkl")
        if os.path.isfile(outfile):
            continue

        RES = {}
        nframes = dataset.nframes[v]

        # load detected tubelets
        VDets = {}
        for fid in range(dataset.len_clip, nframes + 1):
            resname = os.path.join(save_dir, v, "{:0>5}.pkl".format(fid))
            if not os.path.isfile(resname):
                print("ERROR: Missing extracted tubelets " + resname)
                sys.exit()

            with open(resname, 'rb') as file:
                # detection results of per frame
                VDets[fid] = pickle.load(file)

        # VDets = {fid: {label1: {4+1}, label2: {4+1}, ...},
        #          fid: {label1: {4+1}, label2: {4+1}, ...},
        #           ...}
        for ilabel in range(len(dataset.labels)):
            FINISHED_TUBES = []
            CURRENT_TUBES = []  # tubes is a list of tuple (frame, lstubelets)

            def tubescore(tt):
                # calculate average scores of tubelets in tubes
                return np.mean(np.array([tt[i][1][-1] for i in range(len(tt))]))

            for fid in range(dataset.len_clip, nframes + 1):
                # load boxes of the new frame
                # [N, 4+1]
                cur_preds = VDets[fid][ilabel]

                # just start new tubes
                if fid == 1:
                    for i in range(cur_preds.shape[0]):
                        CURRENT_TUBES.append([(1, cur_preds[i, :])])
                    continue

                # sort current tubes according to average score
                avgscore = [tubescore(t) for t in CURRENT_TUBES]
                argsort = np.argsort(-np.array(avgscore))
                CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]

                # loop over tubes
                finished = []
                for it, t in enumerate(CURRENT_TUBES):
                    # compute ious between the last box of t and ltubelets
                    last_fid, last_tubelet = t[-1]
                    ious = []
                    offset = fid - last_fid

                    ious = iou2d(cur_preds[:, :4], last_tubelet[:4])

                    valid = np.where(ious >= 0.5)[0]

                    if valid.size > 0:
                        # take the one with maximum score
                        idx = valid[np.argmax(cur_preds[valid, -1])]
                        CURRENT_TUBES[it].append((fid, cur_preds[idx, :]))
                        cur_preds = np.delete(cur_preds, idx, axis=0)
                    else:
                        if offset >= dataset.len_clip:
                            finished.append(it)

                # finished tubes that are done
                for it in finished[::-1]:  # process in reverse order to delete them with the right index why --++--
                    FINISHED_TUBES.append(CURRENT_TUBES[it][:])
                    del CURRENT_TUBES[it]

                # start new tubes
                for i in range(cur_preds.shape[0]):
                    CURRENT_TUBES.append([(fid, cur_preds[i, :])])

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
                endframe = t[-1][0]
                length = len(t)
                length = endframe - beginframe + 1

                # delete tubes with short duraton
                if length < 15:
                    continue

                # re-organize tubes
                out = np.zeros((length, 6), dtype=np.float32)  
                out[:, 0] = np.arange(beginframe, endframe + 1)
                for i in range(len(t)):
                    fid, box = t[i]
                    try:
                        out[fid- beginframe, 1:5] = box[:4]   # bbox of per frame
                        out[fid- beginframe, -1] = box[-1]    # bbox score
                    except:
                        print(t)
                        print(len(t), out.shape, fid)
                        exit()
                output.append([out, score])
                # out: [num_frames, (frame idx, x1, y1, x2, y2, score)]

            RES[ilabel] = output

        # RES = {ilabel:[(out[length,6],score)]}, ilabel = [0,...]
        with open(outfile, 'wb') as fid:
            pickle.dump(RES, fid)
