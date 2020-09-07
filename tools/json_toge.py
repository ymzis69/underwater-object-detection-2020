import mmcv
import numpy as np
from tqdm import tqdm
from mmdet.ops.nms import nms_wrapper
import json

def get_names(predicted_file):
    names = []
    for predict in predicted_file:
        names.append(predict['image_id'])
    return list(set(names))

if __name__ == "__main__":
    nms_op = nms_wrapper.soft_nms
    CLASSES = ['holothurian', 'echinus', 'scallop', 'starfish']
    model1 = mmcv.load("./results/cascade_rcnn_r50_testB.bbox.json")
    model2 = mmcv.load("./results/cascade_rcnn_r101_testB.bbox.json")
    names1 = get_names(model1)
    names2 = get_names(model2)
    final_names = names1

    result = {name: [[] for i in range(len(CLASSES))] for name in final_names}
    for predict in tqdm(model1):
        name = predict['image_id']
        if name not in final_names:
            continue
        cls = predict['category_id'] - 1
        if cls >= 4:
            continue
        score = predict['score']
        bbox = predict['bbox']
        bbox = bbox + [score]
        result[name][cls].append(np.array(bbox))

    for predict in tqdm(model2):
        name = predict['image_id']
        if name not in final_names:
            continue
        cls = predict['category_id'] - 1
        if cls >= 4:
            continue
        score = predict['score']
        bbox = predict['bbox']
        bbox = bbox + [score]
        result[name][cls].append(np.array(bbox))

    submit = []
    for name in tqdm(final_names):
        for i in range(len(CLASSES)):
            det = np.array(result[name][i])
            if len(det) == 0:
                continue
            det[:, 2] += det[:, 0]
            det[:, 3] += det[:, 1]

            det = det.astype(np.float32)

            if det.shape[0] == 0:
                continue
            cls_dets, _ = nms_op(det, iou_thr=0.7)

            for bbox in cls_dets:
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                res_line = {'image_id': name, 'category_id': int(i + 1), 'bbox': [float(x) for x in bbox[:4]],
                            'score': float(bbox[4])}
                submit.append(res_line)
    print(len(submit))
    print(len(final_names))
    out = "./toge_testB.json"
    with open(out, 'w') as fp:
        json.dump(submit, fp, indent=4, separators=(',', ': '))
    print('over!')
