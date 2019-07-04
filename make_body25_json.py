import json
import os
from PIL import Image
import copy

_COCO_MAX_ID = 900100581904
our_datapath = "C:\\Users\\Jeongmin\\Desktop\\[190307]Hyundai-BirdHands"

body25_keypoint_keys = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                        "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye",
                        "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]

vtouch_keypoint_keys = [u'rKnee', u'lEar', u'rAnkle', u'lElbow', u'rElbow', u'rHeel', u'lHeel', u'rWrist', u'rHip',
                        u'lWrist', u'lSmallToe', u'midHip', u'rEar', u'lShoulder', u'lEye', u'rBigToe', u'rShoulder',
                        u'lBigToe', u'rEye', u'lKnee', u'neck', u'rSmallToe', u'lHip', u'nose', u'lAnkle']

vtouch_keys = ['rightKnee', 'leftEar', 'rightAnkle', 'leftElbow', 'rightElbow', 'rightHeel', 'leftHeel', 'rightWrist', 'rightHip',
               'leftWrist', 'leftSmallToe', 'middleHip', 'rightEar', 'leftShoulder', 'leftEye', 'rightBigToe', 'rightShoulder',
               'leftBigToe', 'rightEye', 'leftKnee', 'neck', 'rightSmallToe', 'leftHip', 'nose', 'leftAnkle']

coco_keypoint_keys = [u'nose', u'left_eye', u'right_eye', u'left_ear', u'right_ear', u'left_shoulder',
                      u'right_shoulder', u'left_elbow', u'right_elbow', u'left_wrist', u'right_wrist', u'left_hip',
                      u'right_hip', u'left_knee', u'right_knee', u'left_ankle', u'right_ankle']

openpose_keypoint_keys = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                          "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye",
                          "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]


vtouch_to_openpose_mapper = {
    'nose': 'Nose',
    'rightKnee': 'RKnee',
    'leftEar': 'LEar',
    'rightAnkle': 'RAnkle',
    'leftElbow': 'LElbow',
    'rightElbow': 'RElbow',
    'rightHeel': 'RHeel',
    'leftHeel': 'LHeel',
    'rightWrist': 'RWrist',
    'rightHip': 'RHip',
    'leftWrist': 'LWrist',
    'leftSmallToe': 'LSmallToe',
    'middleHip': 'MidHip',
    'rightEar': 'REar',
    'leftShoulder': 'LShoulder',
    'leftEye': 'LEye',
    'rightBigToe': 'RBigToe',
    'rightShoulder': 'RShoulder',
    'leftBigToe': 'LBigToe',
    'rightEye': 'REye',
    'leftKnee': 'LKnee',
    'neck': 'Neck',
    'rightSmallToe': 'RSmallToe',
    'leftHip': 'LHip',
    'leftAnkle': 'LAnkle'
}

body25_to_vtouch_mapper = {
    'Nose': 'nose',
    'RKnee': 'rightKnee',
    'LEar': 'leftEar',
    'RAnkle': 'rightAnkle',
    'LElbow': 'leftElbow',
    'RElbow': 'rightElbow',
    'RHeel': 'rightHeel',
    'LHeel': 'leftHeel',
    'RWrist': 'rightWrist',
    'RHip': 'rightHip',
    'LWrist': 'leftWrist',
    'LSmallToe': 'leftSmallToe',
    'MidHip': 'middleHip',
    'REar': 'rightEar',
    'LShoulder': 'leftShoulder',
    'LEye': 'leftEye',
    'RBigToe': 'rightBigToe',
    'RShoulder': 'rightShoulder',
    'LBigToe': 'leftBigToe',
    'REye': 'rightEye',
    'LKnee': 'leftKnee',
    'Neck': 'neck',
    'RSmallToe': 'rightSmallToe',
    'LHip': 'leftHip',
    'LAnkle': 'leftAnkle'
}

coco_to_openpose_mapper = {
    'Nose': 'nose',
    'RKnee': 'right_knee',
    'LEar': 'left_ear',
    'RAnkle': 'right_ankle',
    'LElbow': 'left_elbow',
    'RElbow': 'right_elbow',
    'RHeel': None,
    'LHeel': None,
    'RWrist': 'right_wrist',
    'RHip': 'right_hip',
    'LWrist': 'left_wrist',
    'LSmallToe': None,
    'MidHip': None,
    'REar': 'right_ear',
    'LShoulder': 'left_shoulder',
    'LEye': 'left_eye',
    'RBigToe': None,
    'RShoulder': 'right_shoulder',
    'LBigToe': None,
    'REye': 'right_eye',
    'LKnee': 'left_knee',
    'Neck': None,
    'RSmallToe': None,
    'LHip': 'left_hip',
    'LAnkle': 'left_ankle'
}

POSE_PAIRS = [[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
            [8,9], [9,10], [10,11], [8,12], [12,13], [13,14],
            [1,0], [0,15], [15,17], [0,16], [16,18], [2,17], [5,18],
            [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]


def load_coco_json(_json_path):
    with open(_json_path, 'r') as json_file:
        loaded_json = json.load(json_file)
    return loaded_json


def union_joint_set(_coco_joint, _foot_joint):
    result = list()
    for body25_idx in openpose_keypoint_keys:
        # print(type(coco_to_openpose_mapper[body25_idx]))
        if body25_idx == "MidHip":
            left_idx = coco_keypoint_keys.index('left_hip')
            right_idx = coco_keypoint_keys.index('right_hip')

            if _coco_joint[left_idx*3 + 2] * _coco_joint[right_idx*3 + 2] == 0:
                for i in range(3):
                    result.append(0)
            else:
                for i in range(3):
                    if i == 2:
                        result.append(2)  # TODO: check visible is 1
                    else:
                        result.append((_coco_joint[left_idx*3 + i] + _coco_joint[right_idx*3 + i])/2)

        elif body25_idx == "Neck":
            left_idx = coco_keypoint_keys.index('left_shoulder')
            right_idx = coco_keypoint_keys.index('right_shoulder')

            if _coco_joint[left_idx*3 + 2] * _coco_joint[right_idx*3 + 2] == 0:
                for i in range(3):
                    result.append(0)
            else:
                for i in range(3):
                    if i == 2:
                        result.append(2)  # TODO: check visible is 1
                    else:
                        result.append((_coco_joint[left_idx * 3 + i] + _coco_joint[right_idx * 3 + i])/2)

        elif coco_to_openpose_mapper[body25_idx]:
            idx = coco_keypoint_keys.index(coco_to_openpose_mapper[body25_idx])
            for i in range(3):
                result.append(_coco_joint[idx*3 + i])

    result.extend(_foot_joint)
    return result


def convert_categories(_pose_catecories):
    _pose_catecories[0]['keypoints'] = openpose_keypoint_keys
    _pose_catecories[0]['skeleton'] = POSE_PAIRS

    return _pose_catecories 


def check_foot_data(_pose_id, _foot_anno):
    n = 0
    flag = False
    _foot_joint = None 
    for anno in _foot_anno:
        if _pose_id == anno['id']:
            flag = True
            n = n + 1
            _foot_joint = anno['keypoints']
    if not flag:
        _foot_joint = [0] * 18
    return _foot_joint, flag, n

# TODO:
def make_segment(_bbox):

    return [[_bbox[0], _bbox[2], _bbox[1], _bbox[2], _bbox[1], _bbox[3], _bbox[0], _bbox[3]]]


def calc_area(_bbox):

    return (_bbox[1] - _bbox[0]) * (_bbox[3] - _bbox[2])

def make_annotation_dict(_coco_keypoints, _segmentaion, _area, _id, _bbox, _img_id):
    result = dict()
    """
    result['segmentation'] = _segmentaion
    result['num_keypoints'] = 25  # TODO save int infomation [x, y, v] *25 point. v is visualible
    result['area'] = _area  # TODO save float information
    result['iscrowd'] = 0
    result['keypoints'] = _coco_keypoints
    result['image_id'] = _img_id  # TODO save image id maybe... image name(?)
    result['bbox'] = _bbox  # TODO save object bbox (x, y, width, height)
    result['category_id'] = 1  # TODO save int format category id
    result['id'] = _id  # TODO save int format id..
    """
    # result['segmentation'] = dict()
    result['segmentation'] = [[float(seg) for seg in _segmentaion[0]]]
    result['num_keypoints'] = 25  # TODO save int infomation [x, y, v] *25 point. v is visualible
    result['area'] = float(_area)  # TODO save float information
    result['iscrowd'] = 0
    result['keypoints'] = [points if i%3 == 2 else float(points) for i, points in enumerate(_coco_keypoints)] # _coco_keypoints
    result['image_id'] = _img_id  # TODO save image id maybe... image name(?)
    result['bbox'] = [float(box) for box in _bbox]  # TODO save object bbox (x, y, width, height)
    result['category_id'] = 1  # TODO save int format category id
    result['id'] = _id  # TODO save int format id..
    return result


def make_images_dict(_file_name, _width, _height, _id):
    # [u'license', u'file_name', u'coco_url', u'height', u'width', u'date_captured', u'flickr_url', u'id']
    result = dict()
    """
    result['license'] = None
    result['file_name'] = _file_name  # TODO save image file name
    result['coco_url'] = None  # TODO if use coco url in ildoonet, save url or save None
    result['height'] = _height  # TODO save int format image height
    result['width'] = _width  # TODO save int format image width
    result['date_captured'] = None  # remain None
    result['flickr_url'] = None
    result['id'] = _id  # TODO save id. This id same annotation's id
    """
    result['license'] = 3
    result['file_name'] = _file_name  # TODO save image file name
    result['coco_url'] = ""  # TODO if use coco url in ildoonet, save url or save None
    result['height'] = _height  # TODO save int format image height
    result['width'] = _width  # TODO save int format image width
    result['date_captured'] = ""  # remain None
    result['flickr_url'] = ""
    result['id'] = _id  # TODO save id. This id same annotation's id
    
    return result


# only use vtouch json file
def load_vtouch_json(_json_path):
    with open(_json_path, 'r') as json_file:
        loaded_json = json.load(json_file)

    return loaded_json[0]['pointer']


def find_bounding_box(_keypoint):
    x_list = list()
    y_list = list()
    # for point in _keypoint:
    #     x_list.append(point[0])
    #     y_list.append(point[1])
    nKeypoint = int(len(_keypoint)/3)
    for i in range(nKeypoint):
        if _keypoint[3*i + 2] == 0:
            continue
        x_list.append(_keypoint[3*i])
        y_list.append(_keypoint[3*i + 1])

    return [min(x_list), max(x_list),min(y_list), max(y_list)]


if __name__ == "__main__":

    modes = ["train", "val"]

    for mode in modes:

        FootDataPath = "./person_keypoints_" + mode + "2017_foot_v1.json"
        PoseDataPath = "./person_keypoints_" + mode + "2017.json"

        # FootDataPath = "/Users/jmseo/Downloads/person_keypoints_train2017_foot_v1.json"
        # PoseDataPath = "/Users/jmseo/Downloads/annotations/person_keypoints_train2017.json"

        FootJson = load_coco_json(FootDataPath)
        PoseJson = load_coco_json(PoseDataPath)
        SaveJson = copy.deepcopy(PoseJson)


        # TODO: Convert Pose pair
        SaveJson['categories'] = convert_categories(PoseJson['categories'])
        SaveJson['annotations'] = list()
        # TODO: Union joint iter
        
        for pose_anno in PoseJson['annotations']:
            if type(pose_anno['segmentation']) == type(dict()):
                print("Detected")
                if pose_anno['num_keypoints'] > 0:
                    print(pose_anno['image_id'])
                continue

            foot_joint, flag, check_sum = check_foot_data(pose_anno['id'], FootJson['annotations'])
            body25_skeleton = union_joint_set(pose_anno['keypoints'], foot_joint)
            # pose_anno['keypoints'] = body25_skeleton
            SaveJson['annotations'].append(pose_anno)
            SaveJson['annotations'][-1]['keypoints'] = body25_skeleton 

            # for debug
            if flag and check_sum > 1:
                print(foot_joint, flag, check_sum, pose_anno['id'])
        
        if mode == "train":
            cur_id = _COCO_MAX_ID + 1
            for video in os.listdir(our_datapath):
                json_root = os.path.join(our_datapath, video, video)
                infra_root = os.path.join(our_datapath, video, "infra")

                for i, json_name in enumerate(sorted(os.listdir(json_root))):

                    if not i%3:
                        continue

                    name, ext = os.path.splitext(json_name)
                    infra_name = name + ".jpg"

                    our_json_path = os.path.join(json_root, json_name)
                    print(our_json_path)
                    infra_path = os.path.join(infra_root, infra_name)
                    vtouch_json = load_vtouch_json(our_json_path)

                    cur_id = cur_id + 1
                    save_img_name = "%12d.jpg" % cur_id
                    save_img_path = os.path.join("./vtouch_train", save_img_name)

                    keypoint_body25_format = list()
                    num_inactive = 0
                    for joint in body25_keypoint_keys:
                        x, y = vtouch_json[body25_to_vtouch_mapper[joint]]['left'], \
                               vtouch_json[body25_to_vtouch_mapper[joint]]['top']
                        # v = 2
                        if vtouch_json[body25_to_vtouch_mapper[joint]]['inactive']:
                            keypoint_body25_format.extend([-10, -10, 0])
                            num_inactive = num_inactive + 1
                        else:
                            keypoint_body25_format.extend([x, y, 2])

                    if num_inactive >= 20:
                        # cur_id = cur_id - 1
                        continue

                    img = Image.open(infra_path)
                    img.save(save_img_path)

                    bounding_box = find_bounding_box(keypoint_body25_format)
                    segment = make_segment(bounding_box)
                    area = calc_area(bounding_box)

                    bbox = [bounding_box[0], bounding_box[2], int(bounding_box[1] - bounding_box[0]),
                            int(bounding_box[3] - bounding_box[2])]
                    annotation = make_annotation_dict(keypoint_body25_format, segment, area, cur_id, bbox, cur_id) # , anno_result)
                    # print(annotation)
                    SaveJson['annotations'].append(annotation)
                    # print(type(list(annotation.keys())[0]))
                    images = make_images_dict(save_img_name, img.size[0], img.size[1], cur_id) # , images_result)
                    # print(list(images.keys()))
                    SaveJson['images'].append(images)


        # save json
        with open(os.path.join('./', 'vtouch_coco_' + mode + '.json'), 'w') as output:
            # json.dump(coco_format_json, output, sort_keys=True, indent=4, ensure_ascii=False)
            # json.dump(PoseJson, output, ensure_ascii=False)
            json.dump(SaveJson, output, ensure_ascii=False)
