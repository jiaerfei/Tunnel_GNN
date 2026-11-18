import numpy as np
import pandas as pd


from pathlib import Path
from skimage import morphology
from scipy.ndimage import distance_transform_edt, gaussian_filter
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks, scale_boxes


# ==============================================
# 全局配置：对齐之前约定的规范，所有可调参数统一在这里修改
# ==============================================
class TunnelConfig:
    # 相机标定参数
    PIXEL_TO_MM = 0.29  # 单像素对应实际长度
    PIXEL_TO_MM2 = 0.0841  # 单像素对应实际面积
    # CJJ/T289-2018 规范评级阈值（裂缝宽度mm，渗漏/剥落面积mm²）
    DEFECT_GRADE_THRESH = {
        "crack": [(0.1, 1), (0.4, 2), (0.8, 3), (1.2, 4), (2.0, 5)],
        "leakage": [(400, 1), (4000, 2), (8000, 3), (12000, 4), (20000, 5)],
        "spall": [(400, 1), (4000, 2), (8000, 3), (12000, 4), (20000, 5)]
    }
    # 图像环向位置划分（拱顶/左拱腰/右拱腰/仰拱，按检测框中心y在图像中的相对位置）
    # 这里只是一个粗略估计，后续应根据实际调整
    CIRCUM_POS_RATIO = {
        "arch_top": (0, 0.33),  # 图像上1/3为拱顶
        "left_wall": (0.33, 0.66),  # 中间偏左为左拱腰
        "right_wall": (0.33, 0.66),  # 中间偏右为右拱腰
        "invert": (0.66, 1.0)  # 图像下1/3为仰拱
    }
    # 边权重计算参数（对应GNN里的带权聚合逻辑）
    SEVERITY_WEIGHT = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
    POS_WEIGHT = {"arch_top": 1.0, "invert": 0.8, "left_wall": 0.6, "right_wall": 0.6}
    # 区间配置：环号范围 -> 所属区段（可根据实际工程调整）
    SECTION_RING_MAP = {
        "S01": (1, 200), "S02": (201, 400), "S03": (401, 600), "S04": (601, 800)
    }
    CONF_THRESH = 0.35  # 低置信度过滤阈值（解决审稿人提的零值/误检问题）


def analyze_crack_mask(binary_mask, pixel_scale=0.2):
    """
    改进后的裂缝量化函数
    :param binary_mask: YOLO输出的二值掩膜
    :param pixel_scale: 单像素对应实际长度（mm/像素）
    :return: 长度、平均宽度、最大宽度（物理单位）
    """
    mask = (binary_mask > 0).astype(np.uint8)
    # 1. 形态学闭运算补边缘缺口
    struct_el = morphology.disk(1)
    mask_closed = morphology.binary_closing(mask, structure=struct_el).astype(np.uint8)

    # 2. 过滤小连通域噪点
    labels, num_feat = morphology.label(mask_closed, connectivity=2, return_num=True)
    if num_feat == 0:
        return {"length_mm": 0, "avg_width_mm": 0, "max_width_mm": 0, "valid": False}
    # 保留最大的连通域作为裂缝主体
    largest_cc = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(np.uint8)

    # 3. 骨架提取
    skeleton = morphology.skeletonize(largest_cc).astype(np.uint8)
    if skeleton.sum() < 5:  # 长度太短，无效裂缝
        return {"length_mm": 0, "avg_width_mm": 0, "max_width_mm": 0, "valid": False}

    # 4. 修正骨架长度：对角邻接记√2长度，避免长度偏短
    h, w = skeleton.shape
    length_pixel = 0.0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 1:
                # 8邻接遍历
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        if skeleton[y + dy, x + dx] == 1:
                            length_pixel += np.sqrt(dy ** 2 + dx ** 2)
    length_pixel = length_pixel / 2  # 每条边会被遍历两次，除以2去重
    length_mm = length_pixel * pixel_scale

    # 5. 距离变换+亚像素平滑
    dist_transform = distance_transform_edt(largest_cc)
    # 高斯平滑距离场，消除锯齿噪声
    dist_smooth = gaussian_filter(dist_transform, sigma=0.5)
    skeleton_distances = dist_smooth[skeleton > 0] * 2  # 半宽转全宽

    # 6. 3σ离群值剔除
    mu, sigma = np.mean(skeleton_distances), np.std(skeleton_distances)
    valid_dist = skeleton_distances[np.abs(skeleton_distances - mu) <= 3 * sigma]
    avg_width_pixel = np.mean(valid_dist)
    max_width_pixel = np.percentile(valid_dist, 95)  # 95分位数作为最大宽度，避免噪点

    avg_width_mm = avg_width_pixel * pixel_scale
    max_width_mm = max_width_pixel * pixel_scale

    return {
        "length_mm": round(length_mm, 1),
        "avg_width_mm": round(avg_width_mm, 1),
        "max_width_mm": round(max_width_mm, 1),
        "valid": True
    }


##################################################################
# TODO:根据图像名字确定其绝对位置，这是理想状况，论文中是人工根据装置信息确定位置
def get_circum_position(box, img_shape):
    """根据检测框中心坐标自动判断环向位置（拱顶/拱腰/仰拱），无需人工标注"""
    x1, y1, x2, y2 = box
    h, w = img_shape
    center_y = (y1 + y2) / 2 / h
    center_x = (x1 + x2) / 2 / w
    if center_y < 0.33:
        return "arch_top"
    elif center_y > 0.66:
        return "invert"
    else:
        return "left_wall" if center_x < 0.5 else "right_wall"


def calc_defect_grade(defect_type, value):
    """按照CJJ/T289规范自动计算病害严重度等级（1~5级）"""
    thresh_list = TunnelConfig.DEFECT_GRADE_THRESH[defect_type]
    for thresh, grade in thresh_list:
        if value <= thresh:
            return grade
    return 5


def get_section_by_ring(ring_num):
    """根据环号自动匹配所属200环区段"""
    for sec_id, (start, end) in TunnelConfig.SECTION_RING_MAP.items():
        if start <= ring_num <= end:
            return sec_id
    return "S00"
##################################################################


def predict_to_structured_data(checkpoint, images_path, save_fig=False):
    yolo_model = YOLO(model=checkpoint, verbose=True)
    pred_result = yolo_model.predict(source=images_path, save=save_fig, stream=False)

    defect_records = []
    edge_records = []
    defect_id_counter = 1

    for i, r in enumerate(pred_result):
        if not r or len(r.boxes) == 0:
            continue
        cls_index = r.boxes.cls.cpu().numpy().astype(np.uint8).tolist()
        conf_scores = r.boxes.conf.cpu().numpy()
        cls_names = r.names
        ori_h, ori_w = r.orig_shape
        ori_boxes = scale_boxes(img1_shape=r.masks.shape[-2:], boxes=r.boxes.xyxy.cpu().numpy(),
                                img0_shape=r.orig_shape)
        ori_masks = scale_masks(masks=r.masks.data[None], shape=r.orig_shape)
        ori_masks = ori_masks.squeeze(0).cpu().numpy().astype(np.uint8)

        # 从文件名解析线路、区间、环号（兼容原来的文件命名规则）
        img_stem = Path(r.path).stem
        line_id, interval_id, ring_str = img_stem.split("_")[:3]
        ring_num = int(ring_str.replace("Ring", ""))
        # 自动计算纵向相对位置（区间内的位置比例）
        interval_total_rings = 800  # 可根据实际区间总环数调整
        long_rel_pos = ring_num / interval_total_rings
        section_id = get_section_by_ring(ring_num)

        for j, per_mask in enumerate(ori_masks):
            # 低置信度过滤：剔除误检/零值框，解决审稿人提的零宽度/零面积问题
            if conf_scores[j] < TunnelConfig.CONF_THRESH:
                continue
            per_defect_type = cls_names[cls_index[j]]
            per_box = ori_boxes[j]
            circ_pos = get_circum_position(per_box, (ori_h, ori_w))

            # 量化值计算：严格满足数值倍数约束
            width_mm, length_mm, area_mm2 = None, None, None
            if per_defect_type in {"leakage", "spall"}:
                pixel_area = np.count_nonzero(per_mask)
                raw_area = pixel_area * TunnelConfig.PIXEL_TO_MM2
                area_mm2 = raw_area
                # 异常值过滤：零面积直接跳过
                if area_mm2 < 400:
                    continue
            else:  # crack类型
                crack_metrics = analyze_crack_mask(per_mask, pixel_scale=TunnelConfig.PIXEL_TO_MM)
                raw_width = crack_metrics["avg_width_edt_pixels"] * TunnelConfig.PIXEL_TO_MM
                raw_length = crack_metrics["length_pixels"] * TunnelConfig.PIXEL_TO_MM
                width_mm = raw_width
                length_mm = raw_length
                # 异常值过滤：零宽度裂缝直接跳过
                if width_mm < 0.2:
                    continue

            # 计算规范等级
            if per_defect_type == "crack":
                norm_level = calc_defect_grade("crack", width_mm)
            else:
                norm_level = calc_defect_grade(per_defect_type, area_mm2)

            # 生成病害节点记录
            defect_id = f"D{defect_id_counter:d}"
            defect_records.append({
                "defect_iD": defect_id,
                "defect_type": per_defect_type,
                "width_mm": width_mm,
                "length_mm": length_mm,
                "area_mm2": area_mm2,
                "circum_pos": circ_pos,
                "ring_Number": ring_num,
                "belong_section": section_id,
                "long_rel_pos": round(long_rel_pos, 4),
                "norm_level": norm_level
            })

            # 自动生成带权重的边记录（病害->区段）
            severity_w = TunnelConfig.SEVERITY_WEIGHT[norm_level]
            pos_w = TunnelConfig.POS_WEIGHT[circ_pos]
            edge_weight = round(severity_w * pos_w, 3)
            edge_records.append({
                "src_type": "defect",
                "src_iD": defect_id,
                "tgt_type": "section",
                "tgt_iD": section_id,
                "rel_type": "belongs_to",
                "edge_weight": edge_weight
            })
            defect_id_counter += 1

    # 转成DataFrame，和之前的Excel结构完全一致
    defect_df = pd.DataFrame(defect_records)
    edge_df = pd.DataFrame(edge_records)
    return defect_df, edge_df


if __name__ == '__main__':
    # 运行后会直接输出和之前格式一致的结构化结果
    defect_df, edge_df = predict_to_structured_data(
        checkpoint="best.pt",
        images_path="image",
        save_fig=True
    )

    defect_df.to_excel("Defect_Nodes.xlsx", index=False)
    edge_df.to_excel("Edge_Relations.xlsx", index=False)
    print(f"处理完成：共识别{len(defect_df)}个有效病害，生成{len(edge_df)}条带权边")
