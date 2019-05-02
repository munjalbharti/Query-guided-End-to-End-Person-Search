import os.path as osp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import loadmat
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from sklearn.preprocessing import normalize

from datasets.imdb import imdb


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area
    return inter_area * 1.0 / union_area

class prw_test(imdb):
    def __init__(self, root_dir=None):
        super(prw_test, self).__init__('prwtest')
        self._root_dir = root_dir
        self.data_path = osp.join(self._root_dir, 'frames')
        self._classes = ('__background__', 'person')

        assert osp.isdir(self._root_dir), "Path does not exist: {}".format(self._root_dir)
        assert osp.isdir(self.data_path), "Path does not exist: {}".format(self.data_path)

        self._image_index = self.load_image_set_index()
        self.probes = self.load_probes()


        self.targets_db = self.gt_roidb()

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        return self._image_index[i]

    def image_path_from_index(self, index):
        image_path = osp.join(self.data_path, index) + '.jpg'
        assert osp.isfile(image_path), "Path does not exist: {}".format(image_path)
        return image_path

    def gt_roidb(self):

        gt_roidb = []

        for index, im_name in enumerate(self.image_index):
            boxes_pid = loadmat(osp.join(self._root_dir, 'annotations',
                                      im_name + '.jpg.mat'))
            if 'box_new'in boxes_pid:
                boxes_pid = boxes_pid['box_new']
            elif 'anno_file' in boxes_pid:
                boxes_pid = boxes_pid['anno_file']
            else:
                boxes_pid = boxes_pid['anno_previous']

            boxes = boxes_pid[:,1:5]
            boxes = boxes.astype(np.int32)

            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]

            pids = boxes_pid[:, 0]
            num_objs = len(boxes)

            gt_classes = np.ones((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            overlaps[:, 1] = 1.0
            overlaps = csr_matrix(overlaps)


            gt_roidb.append({
                'im_name': im_name,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'gt_pids': pids,
            })

        return gt_roidb

    def load_image_set_index(self):
        test = loadmat(osp.join(self._root_dir, 'frame_test.mat'))
        test = test['img_index_test'].squeeze()
        test = [str(test_image[0]) for test_image in test]
        return test

    def load_probes(self):
        probes = []

        file = open(osp.join(self._root_dir, 'query_info.txt'), "r")
        for line_no, line in enumerate(file):
            data = line.split()
            ID = int(data[0])
            probe_roi = np.array([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
            probe_im_name = data[5]
            probe_roi = probe_roi.astype(np.int32)
            probe_roi[2:] += probe_roi[:2]
            probes.append((probe_im_name, probe_roi, ID))  # set fixed query box

        return probes


    def eval_search(self, gallery_det, gallery_feat, probe_feat, det_thresh=0.5):

        gallery_det = list(map(lambda x: np.array(x), gallery_det))
        gallery_feat = list(map(lambda x: np.array(x), gallery_feat))

        name_to_det_feat = {}
        for entry, det, feat in zip(self.targets_db, gallery_det, gallery_feat):
            name = entry['im_name']
            pids = entry['gt_pids']
            scores = det[:, 4].ravel()
            inds = np.where(scores >= det_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds], pids)

        aps = []
        accs = []
        topk = [1, 5, 10]

        for i in tqdm(range(len(self.probes))):
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0

            feat_p = probe_feat[i].ravel()
            feat_p = normalize(feat_p[:, np.newaxis], axis=0)  # (256,1)

            probe_imname = self.probes[i][0].split('/')[-1]
            probe_pid = self.probes[i][2]

            # gather gt
            gallery_imgs = filter(lambda x: probe_pid in x['gt_pids'] and x['im_name'] != probe_imname, self.targets_db)
            probe_gts = {}
            for item in gallery_imgs:
                probe_gts[item['im_name']] = item[
                    'boxes'][item['gt_pids'] == probe_pid]

            tested = set([probe_imname])

            # Select the gallery set
            gallery_imgs = filter(lambda x: x['im_name'] != probe_imname, self.targets_db)


            # Gothrough the selected gallery
            for item in gallery_imgs:
                gallery_imname = item['im_name']
                # some contain the probe (gt not empty), some not
                count_gt += (gallery_imname in probe_gts)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat:
                    continue
                det, feat_g, pids_g = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])

                feat_g = normalize(feat_g, axis=1)

                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gallery_imname in probe_gts:
                    gt = probe_gts[gallery_imname].ravel()
                    w, h = gt[2] - gt[0], gt[3] - gt[1]
                    iou_thresh = min(0.5, (w * h * 1.0) /
                                     ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))
                tested.add(gallery_imname)

            # 3. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else \
                average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])


        print('search ranking:')
        mAP = np.mean(aps)
        print('  mAP = {:.2%}'.format(mAP))
        accs_ = np.mean(accs, axis=0)
        cmc_info = {}
        for i, k in enumerate(topk):
            print('  top-{:2d} = {:.2%}'.format(k, accs_[i]))
            cmc_info['top-{:2d}'.format(k)] = accs_[i]

        return mAP, cmc_info


