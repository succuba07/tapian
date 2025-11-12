import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.font_manager as fm
import logging
import glob
import pandas as pd
import traceback

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================== 字体支持与双语言标签函数 ========================
def setup_chinese_font():
    try:
        chinese_fonts = []
        for font in fm.findSystemFonts():
            try:
                font_name = fm.FontProperties(fname=font).get_name()
                if any(char in font_name for char in ['宋体', '黑体', '微软雅黑', 'SimHei', 'KaiTi']):
                    chinese_fonts.append(font)
            except:
                continue
        if chinese_fonts:
            font_path = chinese_fonts[0]
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            logging.info(f"使用中文字体: {font_prop.get_name()}")
            return font_prop.get_name()
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            logging.info("未找到中文字体，使用英文标签")
            return None
    except Exception as e:
        logging.error(f"字体设置失败: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        return None

def get_bilingual_labels():
    """返回中英文双显示标签映射"""
    return {
        '拓片原始轮廓': '拓片原始轮廓 (Original Rubbing)',
        '摹本原始轮廓': '摹本原始轮廓 (Original Copy)', 
        '拓片轮廓（红色）': '拓片轮廓（红色）(Rubbing Contour - Red)',
        '摹本轮廓（绿色）': '摹本轮廓（绿色）(Copy Contour - Green)',
        '完整性得分': '完整性得分 (Completeness Score)',
        'Hu矩相似度': 'Hu矩相似度 (Hu Moment Similarity)',
        '参数': '参数 (Parameters)',
        '图像问题诊断报告': '图像问题诊断报告 (Image Issue Diagnosis Report)',
        '质量等级': '质量等级 (Quality Grade)',
        '质量得分': '质量得分 (Quality Score)',
        '初始轮廓': '初始轮廓 (Initial Contours)',
        '过滤后轮廓': '过滤后轮廓 (Filtered Contours)',
        'Rubbing Contours': '拓片轮廓 (Rubbing Contours)',
        'Copy Contours': '摹本轮廓 (Copy Contours)'
    }

# ======================== 新增工具函数（仅新增，不改动原有函数） ========================
def get_top_n_contours(contours, n):
    """按面积降序排序，返回前n个轮廓（适配动态匹配数量）"""
    if len(contours) <= n:
        return contours
    # 按轮廓面积降序排序
    sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    return sorted_contours[:n]

def calculate_greedy_matching_similarity(tuopian_contours, muben_contours):
    """贪心算法实现多轮廓最优匹配，避免顺序偏差"""
    if len(tuopian_contours) == 0 or len(muben_contours) == 0:
        logging.warning("拓片或摹本无有效轮廓，相似度为0")
        return 0.0, []
    
    # 计算所有轮廓对的相似度矩阵
    sim_matrix = []
    for t_cnt in tuopian_contours:
        t_moments = cv2.moments(t_cnt)
        if t_moments['m00'] == 0:
            sim_matrix.append([0.0]*len(muben_contours))
            continue
        t_hu = cv2.HuMoments(t_moments).flatten()
        row_sims = []
        for m_cnt in muben_contours:
            m_moments = cv2.moments(m_cnt)
            if m_moments['m00'] == 0:
                row_sims.append(0.0)
                continue
            m_hu = cv2.HuMoments(m_moments).flatten()
            dist = distance.euclidean(t_hu, m_hu)
            row_sims.append(1.0/(1.0+dist))  # 距离转相似度
        sim_matrix.append(row_sims)
    
    # 贪心匹配：每次选相似度最高的未匹配对
    used_t = set()  # 已匹配的拓片轮廓索引
    used_m = set()  # 已匹配的摹本轮廓索引
    total_sim = 0.0
    match_pairs = []  # 记录匹配对（拓片索引, 摹本索引, 相似度）
    
    # 按相似度降序排列所有可能的匹配对
    all_matches = []
    for t_idx in range(len(tuopian_contours)):
        for m_idx in range(len(muben_contours)):
            all_matches.append( (sim_matrix[t_idx][m_idx], t_idx, m_idx) )
    all_matches.sort(reverse=True, key=lambda x: x[0])
    
    # 选择最优匹配（无重复）
    for sim, t_idx, m_idx in all_matches:
        if t_idx not in used_t and m_idx not in used_m:
            total_sim += sim
            used_t.add(t_idx)
            used_m.add(m_idx)
            match_pairs.append( (t_idx, m_idx, sim) )
            logging.info(f"最优匹配：拓片轮廓{t_idx+1} ↔ 摹本轮廓{m_idx+1}（相似度: {sim:.3f}）")
    
    # 计算平均相似度
    avg_sim = total_sim / len(match_pairs) if match_pairs else 0.0
    logging.info(f"多轮廓最优匹配完成，平均相似度: {avg_sim:.3f}")
    return avg_sim, match_pairs

# ======================== 图像预处理函数（保持不变） ========================
def preprocess_image(image_path, output_dir=None, image_name="image", save_intermediate=True, is_muben=False):
    if is_muben:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取摹本图像: {image_path}")
        # 摹本二值化：白底黑字（文字0，背景255）
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        denoised_img = binary_img
        gray_img = img
        is_color = False
        logging.info(f"摹本预处理完成: 白底黑字二值化（文字0，背景255）")
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取拓片图像: {image_path}")
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray_img = img
            is_color = False
        denoised_img = cv2.medianBlur(gray_img, 3)
        _, binary_img = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logging.info(f"拓片预处理完成: 黑底白字二值化（文字255，背景0）")
    
    if save_intermediate and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if is_muben:
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_原始灰度.png"), img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_二值化（白底黑字）.png"), binary_img)
        else:
            if is_color:
                cv2.imwrite(os.path.join(output_dir, f"{image_name}_原始彩色.png"), img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_灰度图.png"), gray_img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_去噪后.png"), denoised_img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_二值化（黑底白字）.png"), binary_img)
        create_preprocessing_flowchart(img, gray_img, denoised_img, binary_img, output_dir, image_name, is_muben)
    
    return {
        'original': img,
        'gray': gray_img,
        'denoised': denoised_img,
        'binary': binary_img,
        'is_color': is_color
    }

def create_preprocessing_flowchart(original, gray, denoised, binary, output_dir, image_name, is_muben):
    if is_muben:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('原始灰度图（摹本）')
        axes[0].axis('off')
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title('二值化（白底黑字）')
        axes[1].axis('off')
        plt.suptitle(f'{image_name} - 摹本预处理流程')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        if len(original.shape) == 3:
            axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('原始彩色图像')
        else:
            axes[0, 0].imshow(original, cmap='gray')
            axes[0, 0].set_title('原始灰度图像')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('灰度图')
        axes[0, 1].axis('off')
        axes[1, 0].imshow(denoised, cmap='gray')
        axes[1, 0].set_title('中值滤波去噪后')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title('二值化结果（黑底白字）')
        axes[1, 1].axis('off')
        plt.suptitle(f'{image_name} - 拓片预处理流程')
    plt.tight_layout()
    flowchart_path = os.path.join(output_dir, f"{image_name}_预处理流程图.png")
    plt.savefig(flowchart_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"预处理流程图已保存: {flowchart_path}")

# ======================== 核心处理函数（保持拓片/摹本各自提取逻辑不变，仅修改匹配逻辑） ========================
def extract_tuopian_contours(binary_image, min_area=30, 
                            save_dir=None, image_name="拓片"):
    """提取拓片（黑底白字）的白色文字外轮廓（保持不变）"""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        initial_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(initial_contour_img, contours, -1, (0, 0, 255), 2)
        save_path = os.path.join(save_dir, f"{image_name}_初始轮廓.png")
        cv2.imwrite(save_path, initial_contour_img)
        logging.info(f"拓片初始轮廓已保存: {save_path}")
    
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if save_dir:
        filtered_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(filtered_contour_img, filtered_contours, -1, (0, 0, 255), 2)
        save_path = os.path.join(save_dir, f"{image_name}_过滤后轮廓.png")
        cv2.imwrite(save_path, filtered_contour_img)
        logging.info(f"拓片过滤后轮廓已保存: {save_path}")
    
    return filtered_contours

def extract_muben_contours(binary_image, min_area=30, 
                          save_dir=None, image_name="摹本"):
    """提取摹本（白底黑字）的黑色文字外轮廓（保持不变）"""
    # 生成黑色文字的掩码（文字255，背景0）
    text_mask = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # 在原始摹本二值图上绘制轮廓（白底黑字）
        initial_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(initial_contour_img, contours, -1, (0, 255, 0), 2)
        save_path = os.path.join(save_dir, f"{image_name}_初始轮廓.png")
        cv2.imwrite(save_path, initial_contour_img)
        logging.info(f"摹本初始轮廓已保存: {save_path}")
    
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if save_dir:
        filtered_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(filtered_contour_img, filtered_contours, -1, (0, 255, 0), 2)
        save_path = os.path.join(save_dir, f"{image_name}_过滤后轮廓.png")
        cv2.imwrite(save_path, filtered_contour_img)
        logging.info(f"摹本过滤后轮廓已保存: {save_path}")
    
    return filtered_contours

def detect_image_type(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    height, width = img.shape
    border_pixels = []
    border_pixels.extend(img[0, :])
    border_pixels.extend(img[-1, :])
    border_pixels.extend(img[:, 0])
    border_pixels.extend(img[:, -1])
    avg_border_brightness = np.mean(border_pixels)
    return 'white_bg_black_text' if avg_border_brightness > 127 else 'black_bg_white_text'

# ======================== 相似度计算函数（保持不变，新增贪心匹配函数独立存在） ========================
def calculate_hu_similarity(contours1, contours2):
    if len(contours1) == 0 or len(contours2) == 0:
        return 0
    moments1 = [cv2.moments(cnt) for cnt in contours1]
    moments2 = [cv2.moments(cnt) for cnt in contours2]
    hu_moments1 = [cv2.HuMoments(m) for m in moments1 if m['m00'] > 0]
    hu_moments2 = [cv2.HuMoments(m) for m in moments2 if m['m00'] > 0]
    if not hu_moments1 or not hu_moments2:
        return 0
    min_distances = []
    for hu1 in hu_moments1:
        distances = [distance.euclidean(hu1.flatten(), hu2.flatten()) for hu2 in hu_moments2]
        min_distances.append(min(distances))
    avg_min_distance = np.mean(min_distances) if min_distances else 1.0
    similarity = 1.0 / (1.0 + avg_min_distance)
    logging.info(f"Hu矩相似度计算完成: {similarity:.3f}")
    return similarity

# ======================== 高级噪声过滤函数（保持不变） ========================
def advanced_noise_filtering(contours, image_shape, 
                            min_area=10, max_area=500,
                            min_aspect_ratio=0.1, max_aspect_ratio=0.9,
                            circularity_threshold=0.3):
    filtered_contours = []
    image_area = image_shape[0] * image_shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = min(w, h) / max(w, h)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < circularity_threshold:
                continue
        filtered_contours.append(cnt)
    logging.info(f"高级噪声过滤: 原始数量={len(contours)}, 过滤后数量={len(filtered_contours)}")
    return filtered_contours

# ======================== 诊断与评估函数（保持不变） ========================
def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"创建目录: {directory}")

def diagnose_image_issues(tuopian_path, muben_path, output_dir):
    ensure_directory_exists(os.path.join(output_dir, "dummy.txt"))
    tuopian = cv2.imread(tuopian_path, cv2.IMREAD_GRAYSCALE)
    muben = cv2.imread(muben_path, cv2.IMREAD_GRAYSCALE)
    if tuopian is None or muben is None:
        raise ValueError("无法读取图像文件")
    _, tuopian_binary = cv2.threshold(tuopian, 127, 255, cv2.THRESH_BINARY)
    _, muben_binary = cv2.threshold(muben, 127, 255, cv2.THRESH_BINARY_INV)  # 摹本二值图反转用于诊断
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    labels = get_bilingual_labels()
    axes[0, 0].imshow(tuopian, cmap='gray')
    axes[0, 0].set_title(labels['拓片原始轮廓'])
    axes[0, 0].axis('off')
    axes[0, 1].imshow(muben, cmap='gray')
    axes[0, 1].set_title(labels['摹本原始轮廓'])
    axes[0, 1].axis('off')
    axes[0, 2].imshow(tuopian_binary, cmap='gray')
    axes[0, 2].set_title('Rubbing Binary')
    axes[0, 2].axis('off')
    axes[0, 3].imshow(muben_binary, cmap='gray')
    axes[0, 3].set_title('Copy Binary (Inverted for Diagnosis)')
    axes[0, 3].axis('off')
    contours_tuopian, _ = cv2.findContours(tuopian_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_muben, _ = cv2.findContours(muben_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tuopian_diagnostic = cv2.cvtColor(tuopian_binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(tuopian_diagnostic, contours_tuopian, -1, (0, 0, 255), 2)
    axes[1, 0].imshow(tuopian_diagnostic)
    axes[1, 0].set_title(f'{labels["Rubbing Contours"]} ({len(contours_tuopian)} 个)')
    axes[1, 0].axis('off')
    muben_diagnostic = cv2.cvtColor(muben_binary, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(contours_muben):
        cv2.drawContours(muben_diagnostic, [cnt], -1, (0, 255, 0), 2)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(muben_diagnostic, f'轮廓{i+1}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    axes[1, 1].imshow(muben_diagnostic)
    axes[1, 1].set_title(f'{labels["Copy Contours"]} ({len(contours_muben)} 个)')
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    info_text = (
        f"{labels['图像问题诊断报告']}\n"
        f"拓片尺寸: {tuopian.shape}\n"
        f"摹本尺寸: {muben.shape}\n"
        f"拓片轮廓数: {len(contours_tuopian)}\n"
        f"摹本轮廓数: {len(contours_muben)}\n"
        f"拓片非零像素数: {np.count_nonzero(tuopian_binary)}\n"
        f"摹本非零像素数: {np.count_nonzero(muben_binary)}\n"
        f"建议参数:\n"
        f"- 将min_area减小至10\n"
        f"- 检查摹本图像质量"
    )
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, va='center')
    axes[1, 3].axis('off')
    problem_text = (
        "识别到的问题:\n"
        "1. 噪声去除不充分\n"
        "2. 轮廓提取不完整\n"
        "3. 摹本轮廓显示异常\n"
        "\n解决方案:\n"
        "• 使用高级噪声过滤\n"
        "• 优化轮廓提取策略\n"
        "• 验证摹本图像完整性"
    )
    axes[1, 3].text(0.1, 0.5, problem_text, fontsize=12, va='center', color='red')
    plt.tight_layout()
    output_path = os.path.join(output_dir, labels['图像问题诊断报告'] + '.png')
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"诊断报告已保存: {output_path}")

# 新增：单独保存拓片匹配后轮廓图像的函数（保持不变）
def save_matched_contours_image(binary_image, matched_contours, output_dir, image_name="拓片匹配后轮廓"):
    """单独保存仅含匹配后轮廓的图像"""
    img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, matched_contours, -1, (0, 0, 255), 2)  # 红色绘制匹配轮廓
    save_path = os.path.join(output_dir, f"{image_name}.png")
    cv2.imwrite(save_path, img)
    logging.info(f"单独保存匹配后轮廓图像: {save_path}")

# ======================== 核心评估函数（仅修改匹配逻辑，其他不变） ========================
def assess_completeness_robust(tuopian_path, muben_path, output_dir=None, 
                              min_area=200, save_intermediate=True):
    if output_dir:
        ensure_directory_exists(os.path.join(output_dir, "dummy.txt"))
        if save_intermediate:
            intermediate_dir = os.path.join(output_dir, "中间结果")
            os.makedirs(intermediate_dir, exist_ok=True)
            logging.info(f"创建中间结果目录: {intermediate_dir}")
        else:
            intermediate_dir = None
    else:
        intermediate_dir = None
    
    # 1. 预处理拓片和摹本图像（保持不变）
    logging.info("预处理拓片图像...")
    tuopian_preprocessed = preprocess_image(
        tuopian_path, 
        intermediate_dir if save_intermediate else None, 
        "拓片", 
        save_intermediate=save_intermediate,
        is_muben=False
    )
    tuopian_binary = tuopian_preprocessed['binary']
    
    logging.info("预处理摹本图像...")
    muben_preprocessed = preprocess_image(
        muben_path, 
        intermediate_dir if save_intermediate else None, 
        "摹本", 
        save_intermediate=save_intermediate,
        is_muben=True
    )
    muben_binary = muben_preprocessed['binary']
    
    # 2. 提取初始轮廓（保持不变，调用原有专属函数）
    logging.info("正在提取拓片轮廓...")
    tuopian_contours = extract_tuopian_contours(
        tuopian_binary, 
        min_area=min_area,  
        save_dir=intermediate_dir if save_intermediate else None, 
        image_name="拓片"
    )
    logging.info("正在提取摹本轮廓...")
    muben_contours = extract_muben_contours(
        muben_binary, 
        min_area=min_area,  
        save_dir=intermediate_dir if save_intermediate else None, 
        image_name="摹本"
    )
    logging.info(f"拓片初始轮廓数: {len(tuopian_contours)}")
    logging.info(f"摹本初始轮廓数: {len(muben_contours)}")
    
    # 3. 核心修改：动态匹配轮廓数量 + 多轮廓最优匹配
    n = len(muben_contours)  # 以摹本轮廓数量为标准
    logging.info(f"以摹本轮廓数量为标准（n={n}），提取拓片前{n}大轮廓")
    tuopian_top_n_contours = get_top_n_contours(tuopian_contours, n)  # 拓片前n大轮廓
    logging.info(f"拓片前{n}大轮廓数: {len(tuopian_top_n_contours)}")
    
    # 贪心算法最优匹配（替换原有匹配逻辑）
    hu_similarity, match_pairs = calculate_greedy_matching_similarity(tuopian_top_n_contours, muben_contours)
    matched_tuopian_contours = tuopian_top_n_contours  # 匹配后轮廓即拓片前n大轮廓
    
    # 4. 降级处理（保持不变）
    if not matched_tuopian_contours:
        matched_tuopian_contours = tuopian_contours
        logging.warning("未匹配到任何符合条件的拓片轮廓，使用原始拓片轮廓评估")
    else:
        logging.info(f"匹配后拓片轮廓数（前{n}大）: {len(matched_tuopian_contours)}")
    
    # 5. 单独保存匹配后轮廓图像（保持不变）
    if output_dir and save_intermediate:
        save_matched_contours_image(
            tuopian_binary, 
            matched_tuopian_contours,
            intermediate_dir if save_intermediate else output_dir
        )
    
    # 6. 计算相似度（已用贪心匹配替换，此处沿用结果）
    try:
        # 若贪心匹配失败， fallback 到原有计算方式
        if hu_similarity == 0 and match_pairs == []:
            hu_similarity = calculate_hu_similarity(matched_tuopian_contours, muben_contours)
    except Exception as e:
        logging.error(f"Hu矩相似度计算失败: {e}")
        hu_similarity = 0
    
    # 7. 生成评估报告（新增轮廓数量匹配状态）
    report = {
        'completeness_score': hu_similarity,
        'hu_similarity': hu_similarity,
        'contour_counts': {
            'tuopian': len(matched_tuopian_contours),  # 拓片前n大轮廓数
            'muben': len(muben_contours),
            'tuopian_original': len(tuopian_contours),  # 原始拓片轮廓数
            'matched_rate': len(matched_tuopian_contours) / len(muben_contours) if len(muben_contours) > 0 else 0,
            'contour_count_match': len(matched_tuopian_contours) == len(muben_contours)  # 新增：轮廓数量匹配状态
        },
        'contour_areas': {  # 新增：轮廓面积统计
            'tuopian_top_n_areas': [round(cv2.contourArea(cnt)) for cnt in matched_tuopian_contours],
            'muben_areas': [round(cv2.contourArea(cnt)) for cnt in muben_contours]
        },
        'match_details': {  # 新增：匹配详情
            'match_pairs': match_pairs  # (拓片索引, 摹本索引, 相似度)
        },
        'parameters': {
            'min_area': min_area,
            'similarity_threshold': 0.5,  # 保留原有阈值字段
            'duplicate_removal': True
        }
    }
    
    # 8. 可视化结果（修改以支持标注序号和面积）
    if output_dir and save_intermediate:
        visualize_results(
            tuopian_binary, muben_binary,
            matched_tuopian_contours,  # 传入拓片前n大轮廓
            muben_contours,
            report, output_dir
        )
    
    return report

# ======================== 可视化函数（修改以支持标注序号和面积） ========================
def visualize_results(tuopian_binary, muben_binary, 
                    tuopian_contours, muben_contours,
                    report, output_dir):
    labels = get_bilingual_labels()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 拓片原始二值图（保持不变）
    axes[0, 0].imshow(tuopian_binary, cmap='gray')
    axes[0, 0].set_title(labels['拓片原始轮廓'])
    axes[0, 0].axis('off')
    
    # 2. 摹本原始二值图（保持不变）
    axes[0, 1].imshow(muben_binary, cmap='gray')
    axes[0, 1].set_title(labels['摹本原始轮廓'])
    axes[0, 1].axis('off')
    
    # 3. 拓片匹配后轮廓（标注序号和面积）
    tuopian_color = cv2.cvtColor(tuopian_binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(tuopian_color, tuopian_contours, -1, (0, 0, 255), 2)
    for i, cnt in enumerate(tuopian_contours):
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 标注序号和面积
            cv2.putText(tuopian_color, f'轮廓{i+1}\n面积:{round(area)}', 
                       (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    axes[1, 0].imshow(tuopian_color)
    axes[1, 0].set_title(f"拓片匹配后轮廓（红色，前{len(muben_contours)}大）")
    axes[1, 0].axis('off')
    
    # 4. 摹本轮廓（标注序号和面积）
    muben_color = cv2.cvtColor(muben_binary, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(muben_contours):
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 标注序号和面积
            cv2.putText(muben_color, f'轮廓{i+1}\n面积:{round(area)}', 
                       (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.drawContours(muben_color, muben_contours, -1, (0, 255, 0), 2)
    axes[1, 1].imshow(muben_color)
    axes[1, 1].set_title(labels['摹本轮廓（绿色）'])
    axes[1, 1].axis('off')
    
    # 底部信息标注（新增轮廓数量匹配状态和面积）
    contour_counts = report['contour_counts']
    info_text = (
        f"{labels['完整性得分']}: {report['completeness_score']:.3f}\n"
        f"拓片原始轮廓数: {contour_counts['tuopian_original']}\n"
        f"拓片匹配后轮廓数: {len(tuopian_contours)}（前{len(muben_contours)}大）\n"
        f"摹本轮廓数: {contour_counts['muben']}\n"
        f"轮廓数量匹配: {'是' if contour_counts['contour_count_match'] else '否'}\n"
        f"匹配成功率: {contour_counts['matched_rate']:.1%}\n"
        f"{labels['参数']}: min_area={report['parameters']['min_area']}, 相似度阈值={report['parameters']['similarity_threshold']}"
    )
    plt.figtext(0.5, 0.01, info_text, 
               ha='center', fontsize=10, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'completeness_analysis.png')
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"结果可视化已保存: {output_path}")

# ======================== 综合质量评估函数（修改报告保存逻辑） ========================
def assess_tuopian_quality(tuopian_path, muben_path, output_dir=None, save_intermediate=True):
    completeness_report = assess_completeness_robust(tuopian_path, muben_path, output_dir, save_intermediate=save_intermediate)
    completeness_score = completeness_report['completeness_score'] * 100
    quality_score = completeness_score
    report = {
        'quality_score': round(quality_score, 2),
        'completeness_score': round(completeness_score, 2),
        'completeness_details': completeness_report,
        'image_type': detect_image_type(tuopian_path)
    }
    if output_dir:
        save_quality_report(report, output_dir)
    return report

def save_quality_report(report, output_dir):
    report_path = os.path.join(output_dir, "tuopian_quality_report.txt")
    ensure_directory_exists(report_path)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("甲骨文拓片完整性评估报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"图像类型: {report['image_type']}\n")
        f.write(f"质量得分: {report['quality_score']:.2f}/100\n\n")
        f.write("详细得分:\n")
        f.write(f"  完整性得分: {report['completeness_score']:.2f}\n\n")
        f.write("评估标准说明:\n")
        f.write("• 完整性: 拓片前n大轮廓与摹本n个轮廓的最优匹配相似度，得分越高越完整\n")
        f.write("• 黑底白字特征: 背景黑色(0)，文字白色(255)\n")
        f.write("• 匹配规则: 以摹本轮廓数量n为标准，拓片取前n大外轮廓，贪心算法最优匹配\n")
        f.write("• 外轮廓提取: 全程使用RETR_EXTERNAL，不包含内部子轮廓\n\n")
        f.write("完整性评估详情:\n")
        details = report['completeness_details']
        f.write(f"  Hu矩相似度: {details['hu_similarity']:.3f}\n")
        f.write(f"  轮廓数量: 拓片原始={details['contour_counts']['tuopian_original']}, 拓片匹配后（前n大）={details['contour_counts']['tuopian']}, 摹本={details['contour_counts']['muben']}\n")
        f.write(f"  轮廓数量匹配: {'是' if details['contour_counts']['contour_count_match'] else '否'}\n")
        f.write(f"  拓片前n大轮廓面积: {details['contour_areas']['tuopian_top_n_areas']} 像素\n")
        f.write(f"  摹本轮廓面积: {details['contour_areas']['muben_areas']} 像素\n")
        f.write(f"  匹配成功率: {details['contour_counts']['matched_rate']:.1%}\n")
        f.write(f"  参数: min_area={details['parameters']['min_area']}, 相似度阈值={details['parameters']['similarity_threshold']}\n")
        # 新增匹配对详情
        f.write("\n匹配对详情（拓片索引-摹本索引-相似度）:\n")
        for t_idx, m_idx, sim in details['match_details']['match_pairs']:
            f"  拓片轮廓{t_idx+1} ↔ 摹本轮廓{m_idx+1}（相似度: {sim:.3f}）\n"
    logging.info(f"综合质量报告已保存: {report_path}")

# ======================== 质量评估可视化（保持不变） ========================
def visualize_quality_results(report, output_dir):
    labels = get_bilingual_labels()
    fig, ax = plt.subplots(figsize=(8, 6))
    scores = [report['completeness_score'], report['quality_score']]
    score_labels = ['完整性', '质量得分']
    colors = ['#ff9999', '#99ff99']
    bars = ax.bar(score_labels, scores, color=colors, alpha=0.7)
    ax.set_title('甲骨文拓片质量得分')
    ax.set_ylabel('得分 (0-100)')
    ax.set_ylim(0, 100)
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'quality_assessment_summary.png')
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"质量评估总结图已保存: {output_path}")

# ======================== 主程序（保持不变） ========================
def main():
    tuopian_path = "/work/home/succuba/BRISQUE/庭_xe5j0rm6ao/xe5j0rm6ao.png"
    muben_path = "/work/home/succuba/BRISQUE/test_tapian/庭_xe5j0rm6ao/xe5j0rm6ao.png"
    output_dir = "/work/home/succuba/BRISQUE/results_test_ting2"
    font_name = setup_chinese_font()
    print("开始甲骨文拓片质量综合评估...")
    print("=" * 50)
    try:
        print("正在诊断图像问题...")
        diagnose_image_issues(tuopian_path, muben_path, output_dir)
        print("进行质量综合评估...")
        quality_report = assess_tuopian_quality(tuopian_path, muben_path, output_dir, save_intermediate=True)
        visualize_quality_results(quality_report, output_dir)
        print("\n评估完成!")
        print("=" * 50)
        details = quality_report['completeness_details']
        print(f"图像类型: {quality_report['image_type']}")
        print(f"综合质量得分: {quality_report['quality_score']:.2f}/100")
        print(f"完整性得分: {quality_report['completeness_score']:.2f}")
        print(f"拓片原始轮廓数: {details['contour_counts']['tuopian_original']}")
        print(f"拓片匹配后轮廓数（前{details['contour_counts']['muben']}大）: {details['contour_counts']['tuopian']}")
        print(f"摹本轮廓数: {details['contour_counts']['muben']}")
        print(f"轮廓数量匹配: {'是' if details['contour_counts']['contour_count_match'] else '否'}")
        print(f"匹配成功率: {details['contour_counts']['matched_rate']:.1%}")
        print(f"\n所有结果已保存到: {output_dir}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        traceback.print_exc()

def batch_process_tuopian(tuopian_dir, moben_dir, output_dir):
    tuopian_files = []
    for root, _, files in os.walk(tuopian_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                rel_path = os.path.relpath(root, tuopian_dir)
                tuopian_files.append({
                    'path': os.path.join(root, file),
                    'rel_path': rel_path,
                    'filename': file
                })
    if not tuopian_files:
        print(f"在目录 {tuopian_dir} 中未找到拓片图像")
        return
    results = []
    processed_count = 0
    for tuopian_info in tuopian_files:
        tuopian_path = tuopian_info['path']
        rel_path = tuopian_info['rel_path']
        filename = tuopian_info['filename']
        moben_subdir = os.path.join(moben_dir, rel_path)
        moben_path = os.path.join(moben_subdir, filename)
        if not os.path.exists(moben_path):
            print(f"警告: 未找到摹本文件 {moben_path} (对应拓片: {tuopian_path})")
            continue
        print(f"处理拓片: {rel_path}/{filename}")
        try:
            file_output_dir = os.path.join(output_dir, rel_path, os.path.splitext(filename)[0])
            os.makedirs(file_output_dir, exist_ok=True)
            quality_report = assess_tuopian_quality(tuopian_path, moben_path, file_output_dir, save_intermediate=False)
            details = quality_report['completeness_details']
            results.append({
                'filename': filename,
                'subfolder': rel_path,
                'quality_score': quality_report['quality_score'],
                'completeness_score': quality_report['completeness_score'],
                'tuopian_original_count': details['contour_counts']['tuopian_original'],
                'tuopian_matched_count': details['contour_counts']['tuopian'],
                'muben_count': details['contour_counts']['muben'],
                'contour_count_match': details['contour_counts']['contour_count_match'],
                'matched_rate': details['contour_counts']['matched_rate']
            })
            processed_count += 1
            print(f"✅ 完成: 质量得分={quality_report['quality_score']:.2f}, 轮廓匹配={'成功' if details['contour_counts']['contour_count_match'] else '失败'}, 匹配成功率={details['contour_counts']['matched_rate']:.1%}")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            continue
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "batch_quality_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"批量处理结果已保存到: {csv_path}")
        print("\n批量处理统计摘要:")
        print(f"处理图像数量: {processed_count}/{len(tuopian_files)}")
        print(f"平均质量得分: {df['quality_score'].mean():.2f}")
        print(f"轮廓数量匹配率: {df['contour_count_match'].sum()}/{len(df)} ({df['contour_count_match'].mean()*100:.1f}%)")
        print(f"平均匹配成功率: {df['matched_rate'].mean():.1%}")
        if 'subfolder' in df.columns:
            print("\n按子文件夹统计:")
            grouped = df.groupby('subfolder').agg({
                'quality_score': 'mean',
                'contour_count_match': 'mean',
                'matched_rate': 'mean',
                'filename': 'count'
            }).rename(columns={'filename': 'count', 'contour_count_match': '轮廓匹配率'})
            grouped['轮廓匹配率'] = grouped['轮廓匹配率'] * 100
            print(grouped.round(2))
    else:
        print("没有成功处理的图像")

# ======================== 主程序入口（保持不变） ========================
if __name__ == "__main__":
    main()
    # 批量处理示例（取消注释以使用）
    # tuopian_dir = "/work/home/succuba/BRISQUE/results"
    # moben_dir = "/work/home/succuba/BRISQUE/data/moben"
    # output_dir = "/work/home/succuba/BRISQUE/batch_results"
    # batch_process_tuopian(tuopian_dir, moben_dir, output_dir)