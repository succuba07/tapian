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

# ======================== 字体支持函数 ========================

def setup_chinese_font():
    """
    设置中文字体支持
    """
    try:
        # 尝试查找系统中可用的中文字体
        chinese_fonts = []
        for font in fm.findSystemFonts():
            try:
                font_name = fm.FontProperties(fname=font).get_name()
                if any(char in font_name for char in ['宋体', '黑体', '微软雅黑', 'SimHei', 'KaiTi']):
                    chinese_fonts.append(font)
            except:
                continue
        
        if chinese_fonts:
            # 使用找到的第一个中文字体
            font_path = chinese_fonts[0]
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            logging.info(f"使用中文字体: {font_prop.get_name()}")
            return font_prop.get_name()
        else:
            # 使用默认字体并禁用中文
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            logging.info("未找到中文字体，使用英文标签")
            return None
    except Exception as e:
        logging.error(f"字体设置失败: {e}")
        # 使用英文标签
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        return None

def use_english_labels():
    """
    返回英文标签映射
    """
    return {
        '拓片原始轮廓': 'Original Rubbing',
        '摹本原始轮廓': 'Original Copy', 
        '拓片轮廓（红色）': 'Rubbing Contour (Red)',
        '摹本轮廓（绿色）': 'Copy Contour (Green)',
        '平滑前轮廓（有毛刺）': 'Before Smoothing (with noise)',
        '平滑后轮廓': 'After Smoothing',
        '毛刺平滑效果对比': 'Noise Smoothing Comparison',
        '完整性得分': 'Completeness Score',
        'Hu矩相似度': 'Hu Moment Similarity',
        '形状上下文相似度': 'Shape Context Similarity',
        '傅里叶相似度': 'Fourier Similarity',
        '参数': 'Parameters',
        '图像问题诊断报告': 'Image Issue Diagnosis Report',
        '清晰度得分': 'Clarity Score',
        '综合质量得分': 'Overall Quality Score',
        '质量等级': 'Quality Grade'
    }

# ======================== 核心处理函数 ========================

def enhanced_smooth_contours(contours, smoothing_factor=0.001):
    """
    增强的轮廓平滑算法
    """
    smoothed_contours = []
    for cnt in contours:
        # 计算轮廓周长
        perimeter = cv2.arcLength(cnt, True)
        
        # 使用Douglas-Peucker算法简化轮廓
        epsilon = smoothing_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 进一步平滑轮廓点
        if len(approx) > 2:
            points = approx.reshape(-1, 2).astype(np.float32)
            
            # 应用高斯滤波平滑轮廓点
            smoothed_points = cv2.GaussianBlur(points, (5, 5), 0.5)
            
            # 转换回轮廓格式
            smoothed_cnt = smoothed_points.reshape(-1, 1, 2).astype(np.int32)
            smoothed_contours.append(smoothed_cnt)
        else:
            smoothed_contours.append(approx)
    
    return smoothed_contours

def extract_contours_with_saving(binary_image, min_area=30, smoothing_factor=0.001, 
                                save_dir=None, image_name="contour", is_muben=False):
    """
    提取轮廓并保存中间处理图像
    """
    # 查找初始轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 保存初始轮廓图像
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        initial_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(initial_contour_img, contours, -1, (0, 0, 255), 2)
        save_path = os.path.join(save_dir, f"{image_name}_初始轮廓.png")
        cv2.imwrite(save_path, initial_contour_img)
        logging.info(f"初始轮廓已保存: {save_path}")
    
    # 过滤小面积轮廓
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # 初始化变量
    filtered_contour_img = None
    smoothed_contour_img = None
    
    # 保存过滤后轮廓图像
    if save_dir:
        filtered_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(filtered_contour_img, filtered_contours, -1, (0, 0, 255), 2)
        save_path = os.path.join(save_dir, f"{image_name}_过滤后轮廓.png")
        cv2.imwrite(save_path, filtered_contour_img)
        logging.info(f"过滤后轮廓已保存: {save_path}")
    
    # 应用毛刺平滑（仅对拓片进行）
    if is_muben:
        # 摹本不需要毛刺处理
        smoothed_contours = filtered_contours
    else:
        smoothed_contours = enhanced_smooth_contours(filtered_contours, smoothing_factor)
        
        # 保存毛刺平滑后的轮廓图像
        if save_dir:
            smoothed_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(smoothed_contour_img, smoothed_contours, -1, (0, 0, 255), 2)
            save_path = os.path.join(save_dir, f"{image_name}_平滑后轮廓.png")
            cv2.imwrite(save_path, smoothed_contour_img)
            logging.info(f"平滑后轮廓已保存: {save_path}")
    
    # 创建平滑效果对比图（仅对拓片）
    if save_dir and not is_muben and filtered_contour_img is not None and smoothed_contour_img is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 平滑前
        ax1.imshow(cv2.cvtColor(filtered_contour_img, cv2.COLOR_BGR2RGB))
        ax1.set_title('平滑前轮廓（有毛刺）')
        ax1.axis('off')
        
        # 平滑后
        ax2.imshow(cv2.cvtColor(smoothed_contour_img, cv2.COLOR_BGR2RGB))
        ax2.set_title('平滑后轮廓')
        ax2.axis('off')
        
        plt.suptitle(f'毛刺平滑效果对比 (smoothing_factor={smoothing_factor})')
        save_path = os.path.join(save_dir, f"{image_name}_平滑效果对比.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"平滑对比图已保存: {save_path}")
    
    return smoothed_contours

def filter_text_contours(contours, image_shape, max_area_ratio=0.5, min_aspect_ratio=0.1, edge_margin=10):
    """
    过滤非文字轮廓（如方形边框）
    """
    filtered_contours = []
    image_area = image_shape[0] * image_shape[1]  # 图像总面积
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 1. 过滤面积过大的轮廓（如图像边框）
        if area > image_area * max_area_ratio:
            logging.debug(f"过滤面积过大的轮廓: {area} > {image_area * max_area_ratio}")
            continue
        
        # 2. 过滤非文字形状（如方形边框）
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = min(w, h) / max(w, h)  # 长宽比（接近1表示方形）
        if aspect_ratio > 0.8:  # 方形轮廓可能为边框
            logging.debug(f"过滤方形轮廓: 长宽比={aspect_ratio:.2f}")
            continue
        
        # 3. 过滤位于图像边缘的轮廓（如边框）
        if (x < edge_margin or 
            y < edge_margin or 
            (x + w) > image_shape[1] - edge_margin or 
            (y + h) > image_shape[0] - edge_margin):
            logging.debug(f"过滤边缘轮廓: 位置({x},{y}) 尺寸({w},{h})")
            continue
        
        filtered_contours.append(cnt)
    
    logging.info(f"轮廓过滤: 原始数量={len(contours)}, 过滤后数量={len(filtered_contours)}")
    return filtered_contours

def remove_square_borders(contours, image_shape, max_area_ratio=0.5, aspect_ratio_threshold=0.8, edge_margin=10):
    """
    专门去除方形边框轮廓
    """
    filtered_contours = []
    image_area = image_shape[0] * image_shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 如果面积过大，则可能是边框
        if area > image_area * max_area_ratio:
            continue
            
        # 计算边界矩形
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if w > h else float(h) / w
        # 长宽比接近1，且位于图像边缘，则视为边框
        if aspect_ratio > aspect_ratio_threshold:
            # 检查是否在边缘
            if (x < edge_margin or y < edge_margin or 
                (x + w) > image_shape[1] - edge_margin or 
                (y + h) > image_shape[0] - edge_margin):
                continue
        filtered_contours.append(cnt)
    
    return filtered_contours

def detect_image_type(image_path):
    """
    检测图像是黑底白字还是白底黑字
    返回: 'black_bg_white_text' 或 'white_bg_black_text'
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # 计算图像边缘区域的像素分布
    height, width = img.shape
    border_pixels = []
    
    # 取样图像四个边缘的像素
    border_pixels.extend(img[0, :])  # 上边缘
    border_pixels.extend(img[-1, :])  # 下边缘
    border_pixels.extend(img[:, 0])   # 左边缘
    border_pixels.extend(img[:, -1])  # 右边缘
    
    avg_border_brightness = np.mean(border_pixels)
    
    # 如果边缘区域亮度高，则可能是白底黑字；反之则是黑底白字
    if avg_border_brightness > 127:
        return 'white_bg_black_text'
    else:
        return 'black_bg_white_text'

# ======================== 清晰度评估函数 ========================

def calculate_text_brightness(image_path):
    """
    计算单张图像文字区域的平均亮度得分（针对黑底白字拓片）
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"无法读取图像: {image_path}")
            return None
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用Otsu阈值处理分离文字和背景（黑底白字）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 提取文字区域（白色像素）
        text_pixels = gray[binary == 255]
        if len(text_pixels) == 0:
            logging.warning(f"图像 {os.path.basename(image_path)} 中未检测到文字区域")
            return 0.0
        
        # 计算文字区域平均灰度值（0-255范围）
        avg_brightness = np.mean(text_pixels)
        
        # 归一化到0-100分：亮度越高得分越高（黑底白字）
        brightness_score = (avg_brightness / 255) * 100
        return round(brightness_score, 2)
    
    except Exception as e:
        logging.error(f"处理图像 {image_path} 时出错: {str(e)}")
        return None

# ======================== 修复的高级噪声过滤函数 ========================

def advanced_noise_filtering(contours, image_shape, 
                            min_area=10, max_area=500,
                            min_aspect_ratio=0.1, max_aspect_ratio=0.9,
                            circularity_threshold=0.3,
                            exclude_regions=None):
    """
    修复版高级噪声过滤：针对单字甲骨文拓片优化
    """
    filtered_contours = []
    image_area = image_shape[0] * image_shape[1]
    
    # 调试信息：记录过滤统计
    filter_reasons = {
        'area_too_small': 0,
        'area_too_large': 0,
        'aspect_ratio': 0,
        'circularity': 0,
        'exclude_region': 0
    }
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 1. 面积范围过滤（既不能太小也不能太大）
        if area < min_area:
            filter_reasons['area_too_small'] += 1
            continue
        
        if area > max_area:
            filter_reasons['area_too_large'] += 1
            continue
        
        # 2. 长宽比过滤（排除过于方形或过于细长的轮廓）
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = min(w, h) / max(w, h)  # 长宽比（接近1表示方形）
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            filter_reasons['aspect_ratio'] += 1
            continue
        
        # 3. 圆形度过滤（排除不规则团状噪声）
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < circularity_threshold:  # 圆形度低表示不规则
                filter_reasons['circularity'] += 1
                continue
        
        # 4. 区域排除过滤 - 关键修复：单字位于中心，谨慎使用区域排除
        if exclude_regions:
            # 计算轮廓中心点
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 检查是否在排除区域内
                exclude = False
                for (x1, y1, x2, y2) in exclude_regions:
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        exclude = True
                        break
                
                if exclude:
                    filter_reasons['exclude_region'] += 1
                    continue
        
        filtered_contours.append(cnt)
    
    # 输出过滤统计信息
    total_filtered = len(contours) - len(filtered_contours)
    logging.info(f"高级噪声过滤统计: 总过滤{total_filtered}个轮廓")
    for reason, count in filter_reasons.items():
        if count > 0:
            logging.info(f"  {reason}: {count}")
    
    logging.info(f"高级噪声过滤: 原始数量={len(contours)}, 过滤后数量={len(filtered_contours)}")
    return filtered_contours

def calculate_shape_based_similarity(contours1, contours2, image_shape):
    """
    基于形状特征的相似度计算（修复版）
    """
    # 1. Hu矩相似度
    def hu_moment_similarity(contours_a, contours_b):
        if len(contours_a) == 0 or len(contours_b) == 0:
            return 0
        
        # 计算每个轮廓的Hu矩
        moments_a = [cv2.moments(cnt) for cnt in contours_a]
        moments_b = [cv2.moments(cnt) for cnt in contours_b]
        
        hu_moments_a = [cv2.HuMoments(m) for m in moments_a if m['m00'] > 0]
        hu_moments_b = [cv2.HuMoments(m) for m in moments_b if m['m00'] > 0]
        
        if not hu_moments_a or not hu_moments_b:
            return 0
        
        # 计算Hu矩之间的最小距离
        min_distances = []
        for hu_a in hu_moments_a:
            distances = [distance.euclidean(hu_a.flatten(), hu_b.flatten()) for hu_b in hu_moments_b]
            min_distances.append(min(distances))
        
        avg_min_distance = np.mean(min_distances) if min_distances else 1.0
        return 1.0 / (1.0 + avg_min_distance)
    
    # 计算各项相似度（添加错误处理）
    try:
        hu_similarity = hu_moment_similarity(contours1, contours2)
        
        logging.info(f"相似度计算完成: Hu={hu_similarity:.3f}")
        
        return {
            'hu_moment_similarity': hu_similarity,
            'shape_context_similarity': 0,
            'fourier_similarity': 0
        }
    except Exception as e:
        logging.error(f"形状相似度计算全面失败: {e}")
        # 返回默认值而不是崩溃
        return {
            'hu_moment_similarity': 0,
            'shape_context_similarity': 0,
            'fourier_similarity': 0
        }

# ======================== 诊断与评估函数 ========================

def ensure_directory_exists(path):
    """
    确保目录存在
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"创建目录: {directory}")

def diagnose_image_issues(tuopian_path, muben_path, output_dir):
    """
    诊断图像问题
    """
    # 确保输出目录存在
    ensure_directory_exists(os.path.join(output_dir, "dummy.txt"))
    
    # 读取图像
    tuopian = cv2.imread(tuopian_path, cv2.IMREAD_GRAYSCALE)
    muben = cv2.imread(muben_path, cv2.IMREAD_GRAYSCALE)
    
    if tuopian is None or muben is None:
        raise ValueError("无法读取图像文件")
    
    # 二值化处理
    _, tuopian_binary = cv2.threshold(tuopian, 127, 255, cv2.THRESH_BINARY)
    _, muben_binary = cv2.threshold(muben, 127, 255, cv2.THRESH_BINARY)
    
    # 创建诊断报告图像
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 使用英文标签
    labels = use_english_labels()
    
    # 原始图像分析
    axes[0, 0].imshow(tuopian, cmap='gray')
    axes[0, 0].set_title(labels['拓片原始轮廓'])
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(muben, cmap='gray')
    axes[0, 1].set_title(labels['摹本原始轮廓'])
    axes[0, 1].axis('off')
    
    # 二值化效果检查
    axes[0, 2].imshow(tuopian_binary, cmap='gray')
    axes[0, 2].set_title('Rubbing Binary')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(muben_binary, cmap='gray')
    axes[0, 3].set_title('Copy Binary')
    axes[0, 3].axis('off')
    
    # 轮廓提取诊断
    contours_tuopian, _ = cv2.findContours(tuopian_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_muben, _ = cv2.findContours(muben_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    tuopian_diagnostic = cv2.cvtColor(tuopian_binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(tuopian_diagnostic, contours_tuopian, -1, (0, 0, 255), 2)
    axes[1, 0].imshow(tuopian_diagnostic)
    axes[1, 0].set_title(f'Rubbing Contours ({len(contours_tuopian)} contours)')
    axes[1, 0].axis('off')
    
    muben_diagnostic = cv2.cvtColor(muben_binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(muben_diagnostic, contours_muben, -1, (0, 255, 0), 2)
    axes[1, 1].imshow(muben_diagnostic)
    axes[1, 1].set_title(f'Copy Contours ({len(contours_muben)} contours)')
    axes[1, 1].axis('off')
    
    # 图像统计信息
    axes[1, 2].axis('off')
    info_text = (
        f"Image Diagnosis Report\n"
        f"Rubbing size: {tuopian.shape}\n"
        f"Copy size: {muben.shape}\n"
        f"Rubbing contours: {len(contours_tuopian)}\n"
        f"Copy contours: {len(contours_muben)}\n"
        f"Rubbing non-zero pixels: {np.count_nonzero(tuopian_binary)}\n"
        f"Copy non-zero pixels: {np.count_nonzero(muben_binary)}\n"
        f"Suggested parameters:\n"
        f"- Increase smoothing factor to 0.015\n"
        f"- Reduce min area to 10\n"
        f"- Check copy image quality"
    )
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, va='center')
    
    axes[1, 3].axis('off')
    problem_text = (
        "Identified issues:\n"
        "1. Insufficient noise removal\n"
        "2. Incomplete contour extraction\n"
        "3. Abnormal copy contour display\n"
        "\nSolutions:\n"
        "• Use enhanced smoothing algorithm\n"
        "• Improve contour extraction strategy\n"
        "• Fix copy processing workflow"
    )
    axes[1, 3].text(0.1, 0.5, problem_text, fontsize=12, va='center', color='red')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, labels['图像问题诊断报告'] + '.png')
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"诊断报告已保存: {output_path}")

def assess_completeness_robust(tuopian_path, muben_path, output_dir=None, 
                              min_area=10, smoothing_factor=0.001,save_intermediate=True):
    """
    修复版的完整性评估：针对单字甲骨文拓片优化
    """
    # 确保输出目录存在
    if output_dir:
        ensure_directory_exists(os.path.join(output_dir, "dummy.txt"))
        # 创建中间结果目录
        if save_intermediate:
            intermediate_dir = os.path.join(output_dir, "中间结果")
            os.makedirs(intermediate_dir, exist_ok=True)
            logging.info(f"创建中间结果目录: {intermediate_dir}")
        else:
            intermediate_dir = None
    else:
        intermediate_dir = None
    
    # 读取图像
    tuopian = cv2.imread(tuopian_path, cv2.IMREAD_GRAYSCALE)
    muben = cv2.imread(muben_path, cv2.IMREAD_GRAYSCALE)

    tuopian_type = detect_image_type(tuopian_path)
    muben_type = detect_image_type(muben_path)
    
    if tuopian is None or muben is None:
        raise ValueError("无法读取图像文件")
    
    # 二值化处理
    # 根据图像类型自适应二值化
    if tuopian_type == 'black_bg_white_text':
        _, tuopian_binary = cv2.threshold(tuopian, 127, 255, cv2.THRESH_BINARY)
    else:
        # 白底黑字：反转二值化
        _, tuopian_binary = cv2.threshold(tuopian, 127, 255, cv2.THRESH_BINARY_INV)

    if muben_type == 'black_bg_white_text':
        _, muben_binary = cv2.threshold(muben, 127, 255, cv2.THRESH_BINARY)
    else:
        # 白底黑字：反转二值化
        _, muben_binary = cv2.threshold(muben, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 保存二值化图像
    if intermediate_dir and save_intermediate:
        save_path = os.path.join(intermediate_dir, "拓片二值化.png")
        ensure_directory_exists(save_path)
        cv2.imwrite(save_path, tuopian_binary)
        
        save_path = os.path.join(intermediate_dir, "摹本二值化.png")
        ensure_directory_exists(save_path)
        cv2.imwrite(save_path, muben_binary)
        logging.info("二值化图像已保存")
    
    logging.info("正在提取轮廓...")
    
    # 使用改进的轮廓提取并保存中间图像
    tuopian_contours = extract_contours_with_saving(
        tuopian_binary, 
        min_area=min_area, 
        smoothing_factor=smoothing_factor, 
        save_dir=intermediate_dir if save_intermediate else None, 
        image_name="拓片", 
        is_muben=False
    )
    
    muben_contours = extract_contours_with_saving(
        muben_binary, 
        min_area=min_area, 
        smoothing_factor=0.001,  # 摹本使用极轻度平滑
        save_dir=intermediate_dir if save_intermediate else None, 
        image_name="摹本", 
        is_muben=True
    )
    
    logging.info(f"拓片轮廓数: {len(tuopian_contours)}")
    logging.info(f"摹本轮廓数: {len(muben_contours)}")
    
    # 应用轮廓过滤（主要针对摹本）
    muben_contours = filter_text_contours(
        muben_contours, 
        muben_binary.shape, 
        max_area_ratio=0.5, 
        min_aspect_ratio=0.1, 
        edge_margin=10
    )
    
    logging.info(f"过滤后摹本轮廓数: {len(muben_contours)}")
    
    # 专门去除摹本中的方形边框
    muben_contours = remove_square_borders(
        muben_contours, 
        muben_binary.shape,
        max_area_ratio=0.5,
        aspect_ratio_threshold=0.8,
        edge_margin=10
    )
    logging.info(f"去除方形边框后摹本轮廓数: {len(muben_contours)}")
    
    # ================ 关键修复：针对单字拓片的优化过滤 ================
    height, width = tuopian_binary.shape
    image_area = height * width
    
    # 关键修复1：单字位于图像中心，不应排除中心区域
    # 改为排除真正的边缘噪声区域，而不是中心区域
    exclude_regions = [
        # 排除图像四个角落的小区域，而不是中心区域
        (0, 0, int(width * 0.1), int(height * 0.1)),  # 左上角
        (int(width * 0.9), 0, width, int(height * 0.1)),  # 右上角
        (0, int(height * 0.9), int(width * 0.1), height),  # 左下角
        (int(width * 0.9), int(height * 0.9), width, height)  # 右下角
    ]
    
    # 关键修复2：针对单字拓片的优化参数
    single_character_params = {
        'min_area': min_area,  # 使用较小的min_area
        'max_area': int(image_area * 0.8),  # 单字可能很大，增加最大面积限制
        'min_aspect_ratio': 0.05,  # 甲骨文形状多样，放宽长宽比限制
        'max_aspect_ratio': 0.95,
        'circularity_threshold': 0.1,  # 甲骨文不规则，大幅降低圆形度阈值
        'exclude_regions': exclude_regions  # 使用修复后的排除区域
    }
    
    # 应用修复后的高级噪声过滤
    tuopian_contours = advanced_noise_filtering(
        tuopian_contours, 
        tuopian_binary.shape,
        **single_character_params
    )
    logging.info(f"修复过滤后拓片轮廓数: {len(tuopian_contours)}")
    
    # 关键修复3：如果轮廓数仍然为0，使用备用方案
    if len(tuopian_contours) == 0:
        logging.warning("修复过滤后轮廓数仍为0，使用最宽松参数")
        
        # 重新提取轮廓，使用最宽松参数
        contours_tuopian, _ = cv2.findContours(tuopian_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tuopian_contours = [cnt for cnt in contours_tuopian if cv2.contourArea(cnt) > 5]  # 极低面积阈值
        
        # 使用最宽松的过滤参数
        fallback_params = {
            'min_area': 5,
            'max_area': int(image_area * 0.9),
            'min_aspect_ratio': 0.01,
            'max_aspect_ratio': 0.99,
            'circularity_threshold': 0.05,
            'exclude_regions': None  # 完全禁用区域排除
        }
        
        tuopian_contours = advanced_noise_filtering(
            tuopian_contours, 
            tuopian_binary.shape,
            **fallback_params
        )
        logging.info(f"最宽松参数过滤后拓片轮廓数: {len(tuopian_contours)}")
    
    # 计算形状相似度（添加错误处理）
    try:
        similarity_scores = calculate_shape_based_similarity(
            tuopian_contours, muben_contours, tuopian_binary.shape
        )
    except Exception as e:
        logging.error(f"形状相似度计算失败: {e}")
        similarity_scores = {
            'hu_moment_similarity': 0,
            'shape_context_similarity': 0,
            'fourier_similarity': 0
        }
    
    # 计算综合完整性得分
    weights = {
        'hu_moment_similarity': 1.0,
        'shape_context_similarity': 0,
        'fourier_similarity': 0
    }
    
    completeness_score = sum(
        similarity_scores[metric] * weight 
        for metric, weight in weights.items()
    )
    
    # 生成报告
    report = {
        'completeness_score': completeness_score,
        'similarity_breakdown': similarity_scores,
        'contour_counts': {
            'tuopian': len(tuopian_contours),
            'muben': len(muben_contours)
        },
        'parameters': {
            'min_area': min_area,
            'smoothing_factor': smoothing_factor
        }
    }
    
    # 可视化结果
    if output_dir and save_intermediate:
        visualize_results(
            tuopian_binary, muben_binary,
            tuopian_contours, muben_contours,
            report, output_dir
        )
    
    return report

def visualize_results(tuopian_binary, muben_binary, 
                    tuopian_contours, muben_contours,
                    report, output_dir):
    """
    可视化结果
    """
    # 使用英文标签
    labels = use_english_labels()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始图像
    axes[0, 0].imshow(tuopian_binary, cmap='gray')
    axes[0, 0].set_title(labels['拓片原始轮廓'])
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(muben_binary, cmap='gray')
    axes[0, 1].set_title(labels['摹本原始轮廓'])
    axes[0, 1].axis('off')
    
    # 轮廓叠加显示
    # 拓片轮廓（红色）
    tuopian_color = cv2.cvtColor(tuopian_binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(tuopian_color, tuopian_contours, -1, (0, 0, 255), 2)
    axes[1, 0].imshow(tuopian_color)
    axes[1, 0].set_title(labels['拓片轮廓（红色）'])
    axes[1, 0].axis('off')
    
    # 摹本轮廓（绿色）
    muben_color = cv2.cvtColor(muben_binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(muben_color, muben_contours, -1, (0, 255, 0), 2)
    axes[1, 1].imshow(muben_color)
    axes[1, 1].set_title(labels['摹本轮廓（绿色）'])
    axes[1, 1].axis('off')
    
    # 添加文本信息
    info_text = (
        f"{labels['完整性得分']}: {report['completeness_score']:.3f}\n"
        f"{labels['Hu矩相似度']}: {report['similarity_breakdown']['hu_moment_similarity']:.3f}\n"
        f"{labels['形状上下文相似度']}: {report['similarity_breakdown']['shape_context_similarity']:.3f}\n"
        f"{labels['傅里叶相似度']}: {report['similarity_breakdown']['fourier_similarity']:.3f}\n"
        f"{labels['参数']}: min_area={report['parameters']['min_area']}, smoothing_factor={report['parameters']['smoothing_factor']}"
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

# ======================== 综合质量评估函数 ========================

def assess_tuopian_quality(tuopian_path, muben_path, output_dir=None, save_intermediate=True):
    """
    甲骨文拓片质量综合评估（完整性70% + 清晰度30%）
    """
    # # 1. 清晰度评估
    # clarity_score = calculate_text_brightness(tuopian_path)
    # if clarity_score is None:
    #     clarity_score = 0
    # logging.info(f"清晰度评估: {clarity_score:.2f}")
    
    # 2. 完整性评估
    completeness_report = assess_completeness_robust(tuopian_path, muben_path, output_dir,save_intermediate=save_intermediate)
    completeness_score = completeness_report['completeness_score'] * 100  # 转换为0-100分
    
    # # 3. 综合质量得分（权重：完整性70%，清晰度30%）
    # weights = {
    #     'completeness': 0.7,
    #     'clarity': 0.3
    
    quality_score = completeness_score
    
    # # 4. 质量等级评定
    # if quality_score >= 90:
    #     quality_grade = "优秀"
    # elif quality_score >= 80:
    #     quality_grade = "良好" 
    # elif quality_score >= 60:
    #     quality_grade = "一般"
    # else:
    #     quality_grade = "较差"
    
    # 5. 生成综合报告
    report = {
        'quality_score': round(quality_score, 2),
        # 'quality_grade': quality_grade,
        'completeness_score': round(completeness_score, 2),
        # 'clarity_score': round(clarity_score, 2),
        'completeness_details': completeness_report,
        'image_type': '黑底白字拓片'
        # 'weights': weights
    }
    
    # 6. 保存报告
    if output_dir:
        save_quality_report(report, output_dir)
    
    return report

def save_quality_report(report, output_dir):
    """
    保存综合质量报告
    """
    report_path = os.path.join(output_dir, "tuopian_quality_report.txt")
    ensure_directory_exists(report_path)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("甲骨文拓片质量综合评估报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"图像类型: {report['image_type']}\n")
        f.write(f"质量得分: {report['quality_score']:.2f}/100\n\n")
        f.write("详细得分:\n")
        f.write(f"  完整性得分: {report['completeness_score']:.2f}\n\n")
        
        f.write("评估标准说明:\n")
        f.write("• 清晰度: 文字区域亮度越高，与背景对比越明显，得分越高\n")
        f.write("• 完整性: 拓片文字与标准摹本的形状相似度，得分越高越完整\n")
        f.write("• 黑底白字特征: 背景黑色(0)，文字白色(255)\n\n")
        
        # 添加完整性评估详情
        f.write("完整性评估详情:\n")
        details = report['completeness_details']
        f.write(f"  Hu矩相似度: {details['similarity_breakdown']['hu_moment_similarity']:.3f}\n")
        f.write(f"  轮廓数量: 拓片={details['contour_counts']['tuopian']}, 摹本={details['contour_counts']['muben']}\n")
    
    logging.info(f"综合质量报告已保存: {report_path}")

def visualize_quality_results(report, output_dir):
    """
    可视化综合质量评估结果
    """
    # 使用英文标签
    labels = use_english_labels()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 得分对比图
    scores = [report['completeness_score'], report['quality_score']]
    score_labels = ['完整性', '质量得分']
    colors = ['#ff9999', '#99ff99']
    
    bars = ax.bar(score_labels, scores, color=colors, alpha=0.7)
    ax.set_title('甲骨文拓片质量得分')
    ax.set_ylabel('得分 (0-100)')
    ax.set_ylim(0, 100)
    
    # 在柱状图上添加数值标签
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

# ======================== 主程序 ========================

def main():
    """
    主程序：甲骨文拓片质量综合评估
    """
    # 配置路径
    tuopian_path = "/work/home/succuba/BRISQUE/improved_results/text_skeleton.png"
    muben_path = "/work/home/succuba/BRISQUE/data/moben/0tmf9tz3ge.png"
    output_dir = "/work/home/succuba/BRISQUE/results_all"
    
    # 设置字体
    font_name = setup_chinese_font()
    
    print("开始甲骨文拓片质量综合评估...")
    print("=" * 50)
    
    try:
        # 首先诊断问题
        print("正在诊断图像问题...")
        diagnose_image_issues(tuopian_path, muben_path, output_dir)
        
        # 然后进行综合评估
        print("进行质量综合评估...")
        quality_report = assess_tuopian_quality(tuopian_path, muben_path, output_dir,save_intermediate=True)
        
        # 可视化质量结果
        visualize_quality_results(quality_report, output_dir)
        
        # 打印结果
        print("\n评估完成!")
        print("=" * 50)
        print(f"图像类型: {quality_report['image_type']}")
        print(f"综合质量得分: {quality_report['quality_score']:.2f}/100 - {quality_report['quality_grade']}")
        print(f"完整性得分: {quality_report['completeness_score']:.2f} (权重70%)")
        print(f"清晰度得分: {quality_report['clarity_score']:.2f} (权重30%)")
        print(f"\n所有结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def batch_process_tuopian(tuopian_dir, moben_dir, output_dir):
    """
    批量处理拓片图像 - 改进版：要求子文件夹和文件名完全匹配
    """
    # 获取所有拓片图像（包括子文件夹）
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
        
        # 构建对应的摹本路径
        moben_subdir = os.path.join(moben_dir, rel_path)
        moben_path = os.path.join(moben_subdir, filename)
        
        if not os.path.exists(moben_path):
            print(f"警告: 未找到摹本文件 {moben_path} (对应拓片: {tuopian_path})")
            continue
        
        print(f"处理拓片: {rel_path}/{filename}")
        
        try:
            # 创建子目录保存结果
            file_output_dir = os.path.join(output_dir, rel_path, os.path.splitext(filename)[0])
            os.makedirs(file_output_dir, exist_ok=True)
            
            # 进行质量评估
            quality_report = assess_tuopian_quality(tuopian_path, moben_path, file_output_dir,save_intermediate=False)
            
            results.append({
                'filename': filename,
                'subfolder': rel_path,
                'quality_score': quality_report['quality_score'],
                'completeness_score': quality_report['completeness_score'],
                # 'clarity_score': quality_report['clarity_score'],
                # 'quality_grade': quality_report['quality_grade']
            })
            
            processed_count += 1
            print(f"✅ 完成: 质量得分={quality_report['quality_score']:.2f}")
            
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            continue
    
    # 保存批量处理结果
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "batch_quality_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"批量处理结果已保存到: {csv_path}")
        
        # 打印统计摘要
        print("\n批量处理统计摘要:")
        print(f"处理图像数量: {processed_count}/{len(tuopian_files)}")
        print(f"平均质量得分: {df['quality_score'].mean():.2f}")
        # print(f"质量等级分布:")
        # print(df['quality_grade'].value_counts())
        
        # 按子文件夹分组统计
        if 'subfolder' in df.columns:
            print("\n按子文件夹统计:")
            grouped = df.groupby('subfolder')['quality_score'].agg(['mean', 'count'])
            print(grouped)
    else:
        print("没有成功处理的图像")

if __name__ == "__main__":
    # 单张图像评估
    #main()
    
    # 批量处理示例（取消注释以使用）
    tuopian_dir = "/work/home/succuba/BRISQUE/results"
    muben_dir = "/work/home/succuba/BRISQUE/data/moben"
    output_dir = "/work/home/succuba/BRISQUE/batch_results"
    batch_process_tuopian(tuopian_dir, muben_dir, output_dir)