# -*- coding: utf-8 -*-
"""
甲骨文拓片骨架提取系统 - 增强版（集成智能噪声分离）
核心升级：
1. 智能噪声分离：在清晰度评估前去除外源性设备噪声，保留内源性拓片特征
2. 多维度清晰度评估：梯度域（抗噪声）+频域（尺度不变）+局部自适应（解决对比度差异）
3. 详细诊断报告：提供噪声处理分析和改进建议
"""

import numpy as np
import cv2
import os
from datetime import datetime
import glob
import pandas as pd
import logging
import matplotlib.pyplot as plt
from scipy import ndimage

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IntelligentNoiseProcessor:
    """智能噪声处理器 - 专门区分和处理外源性噪声"""
    
    def __init__(self):
        # 外源性噪声特征参数（可根据实际数据调整）
        self.exogenous_params = {
            'max_size': 10,           # 最大噪声点尺寸（像素）
            'intensity_threshold': 160,  # 噪声点亮度阈值
            'min_isolation': 0.5,    # 最小孤立性阈值
            'morph_kernel_size': 2   # 形态学操作核大小
        }
        
    def remove_exogenous_noise(self, image_path):
        """
        去除外源性噪声但保留内源性特征
        返回: (去噪后的图像, 噪声掩码)
        """
        try:
            # 读取图像
            if isinstance(image_path, str):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = image_path.copy()
                
            if img is None:
                return None, None
            
            # 检测外源性噪声
            noise_mask = self._detect_exogenous_noise(img)
            
            # 应用中值滤波，但仅限于噪声区域
            denoised = self._selective_median_filter(img, noise_mask)
            
            return denoised, noise_mask
            
        except Exception as e:
            logging.error(f"噪声处理失败: {str(e)}")
            return None, None
        except KeyError as e:
            logging.error(f"参数配置错误: {str(e)}，请检查exogenous_params字典中的键名")
            return None, None
        except AttributeError as e:
            logging.error(f"OpenCV函数调用错误: {str(e)}，请检查OpenCV版本和函数名")
            return None, None
        except Exception as e:
            logging.error(f"噪声处理失败: {str(e)}")
            return None, None
    
    def _detect_exogenous_noise(self, gray_image):
        """检测外源性噪声特征"""
        height, width = gray_image.shape
        noise_mask = np.zeros_like(gray_image, dtype=np.uint8)
        
        # 基于连通组件分析
        _, binary = cv2.threshold(gray_image, self.exogenous_params['intensity_threshold'], 
                                255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):  # 跳过背景
            # 检查组件特征
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 外源性噪声判断条件
            if (area <= self.exogenous_params['max_size'] ** 2 and  # 小面积
                max(width, height) <= self.exogenous_params['max_size'] and  # 小尺寸
                self._is_isolated_component(labels == i, labels)):  # 孤立性
                
                # 标记为外源性噪声
                noise_mask[labels == i] = 255
        
        # 形态学操作增强检测
        kernel = np.ones((self.exogenous_params['morph_kernel_size'], 
                         self.exogenous_params['morph_kernel_size']), np.uint8)
        noise_mask = cv2.morphologyEx(noise_mask, cv2.MORPH_CLOSE, kernel)
        
        return noise_mask
    
    def _is_isolated_component(self, component_mask, all_labels):
        """检查组件是否孤立（不与主要结构连接）"""
        # 膨胀组件检查邻接关系
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(component_mask.astype(np.uint8), kernel)
        
        # 检查膨胀后是否接触其他大组件
        dilated_labels = dilated * all_labels
        unique_neighbors = np.unique(dilated_labels[dilated > 0])
        
        # 如果只接触自己或背景，则是孤立的
        return len(unique_neighbors) <= 2
    
    def _selective_median_filter(self, image, noise_mask, kernel_size=3):
        """选择性中值滤波 - 仅在噪声区域应用"""
        # 对整个图像进行轻度滤波
        lightly_filtered = cv2.medianBlur(image, kernel_size)
        
        # 只在检测到的噪声区域应用滤波结果
        result = image.copy()
        result[noise_mask == 255] = lightly_filtered[noise_mask == 255]
        
        return result

class EnhancedClarityEvaluator:
    """增强的清晰度评估器 - 集成智能噪声去除功能"""
    
    def __init__(self):
        # 权重配置（可根据实际数据调整）
        self.weights = {
            'gradient': 0.5,    # 梯度特征权重（抗噪声）
            'frequency': 0.25,   # 频域特征权重（尺度不变）
            'local': 0.25        # 局部特征权重（适应对比度差异）
        }
        # 新增：集成智能噪声处理器
        self.noise_processor = IntelligentNoiseProcessor()
        
    def calculate_clarity_score(self, image_path):
        """
        综合清晰度评估（先去除外源性噪声）
        返回: 综合得分(0-100)，失败返回None
        """
        try:
            # 读取图像并去除外源性噪声
            denoised_img, noise_mask = self.noise_processor.remove_exogenous_noise(image_path)
            if denoised_img is None:
                logging.error(f"图像去噪失败: {image_path}")
                return None
            
            # 在去噪后的图像上并行计算三种特征
            gradient_score = self._gradient_based_sharpness(denoised_img)
            frequency_score = self._frequency_domain_sharpness(denoised_img)
            local_score = self._adaptive_local_sharpness(denoised_img)
            
            # 归一化处理
            scores = {
                'gradient': self._normalize_score(gradient_score, (0, 100)),
                'frequency': self._normalize_score(frequency_score, (0, 100)),
                'local': self._normalize_score(local_score, (0, 5000))
            }
            
            # 加权综合得分
            total_score = sum(scores[method] * weight 
                             for method, weight in self.weights.items())
            
            return round(total_score, 2)
            
        except Exception as e:
            logging.error(f"综合清晰度评估失败: {str(e)}")
            return None
    
    def calculate_detailed_assessment(self, image_path):
        """
        详细的清晰度评估（返回完整分析报告，包含噪声处理信息）
        """
        try:
            # 读取图像并去除外源性噪声
            denoised_img, noise_mask = self.noise_processor.remove_exogenous_noise(image_path)
            if denoised_img is None:
                return None
            
            # 计算噪声去除比例（用于诊断报告）
            noise_ratio = np.sum(noise_mask) / (noise_mask.size * 255) if noise_mask is not None else 0
            
            # 在去噪后的图像上计算清晰度
            gradient_score = self._gradient_based_sharpness(denoised_img)
            frequency_score = self._frequency_domain_sharpness(denoised_img)
            local_score = self._adaptive_local_sharpness(denoised_img)
            
            # 归一化
            scores = {
                'gradient': self._normalize_score(gradient_score, (0, 100)),
                'frequency': self._normalize_score(frequency_score, (0, 100)),
                'local': self._normalize_score(local_score, (0, 5000))
            }
            
            # 加权综合
            total_score = sum(scores[method] * weight 
                             for method, weight in self.weights.items())
            
            # 问题诊断（基于去噪后图像）
            diagnosis = self._diagnose_issues(scores)
            
            # 添加噪声处理信息到诊断结果
            diagnosis.append(f"外源性噪声去除比例: {noise_ratio:.1%}")
            
            return {
                'total_score': round(total_score, 2),
                'detailed_scores': scores,
                'diagnosis': diagnosis,
                'recommendation': self._generate_recommendation(total_score, diagnosis),
                'noise_removed_ratio': noise_ratio,
                'assessment_note': '评估基于去噪后图像，外源性噪声已去除'
            }
            
        except Exception as e:
            logging.error(f"详细清晰度评估失败: {str(e)}")
            return None
    
    def _gradient_based_sharpness(self, image):
        """
        基于梯度域的清晰度评估（抗噪声干扰）
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 多方向梯度计算
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 梯度幅值（边缘强度）
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 高梯度像素比例（真正的文字边缘）
        high_gradient_ratio = np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 90)) / gradient_magnitude.size
        
        return high_gradient_ratio * 100
    
    def _frequency_domain_sharpness(self, image):
        """
        频域清晰度评估（尺度不变性）
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 傅里叶变换
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # 频域幅度谱
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        # 高频能量占比（与图像尺度无关）
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        
        # 创建环形掩码提取高频成分
        high_freq_mask = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                dist = np.sqrt((r - crow)**2 + (c - ccol)**2)
                if dist > min(rows, cols) * 0.3:  # 高频区域
                    high_freq_mask[r, c] = 1
        
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
        total_energy = np.sum(magnitude_spectrum)
        
        return (high_freq_energy / total_energy) * 100 if total_energy > 0 else 0
    
    def _adaptive_local_sharpness(self, image, block_size=32):
        """
        自适应局部清晰度评估（解决对比度差异）
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        height, width = gray.shape
        
        sharpness_scores = []
        
        # 分块处理
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = gray[i:min(i+block_size, height), j:min(j+block_size, width)]
                
                if block.size > 100:  # 确保块足够大
                    # 计算块的局部清晰度（使用Laplacian方差）
                    laplacian_var = cv2.Laplacian(block, cv2.CV_64F).var()
                    if not np.isnan(laplacian_var):
                        sharpness_scores.append(laplacian_var)
        
        # 使用中位数避免极端值影响
        return np.median(sharpness_scores) if sharpness_scores else 0
    
    def _normalize_score(self, value, value_range):
        """归一化分数到0-100范围"""
        min_val, max_val = value_range
        if max_val - min_val == 0:
            return 0
        normalized = (value - min_val) / (max_val - min_val) * 100
        return max(0, min(100, normalized))
    
    def _diagnose_issues(self, scores):
        """问题诊断（基于去噪后图像）"""
        issues = []
        
        if scores['gradient'] < 30:
            issues.append("噪声干扰较严重（去噪后评估）")
        if scores['frequency'] < 25:
            issues.append("图像可能过度模糊或缩放不一致")
        if scores['local'] < 20:
            issues.append("局部对比度差异明显")
        
        return issues if issues else ["图像质量良好"]
    
    def _generate_recommendation(self, total_score, issues):
        """生成改进建议（基于去噪后评估）"""
        if total_score >= 80:
            return "图像质量优秀，无需进一步处理"
        elif total_score >= 60:
            return "建议轻度对比度增强"
        else:
            return "需要综合处理：对比度增强+尺寸标准化"

class OracleBoneSkeletonExtractor:
    """甲骨文骨架提取器 - 专门用于生成test_skeleton图"""
    
    def __init__(self, min_area=70, smoothing=True):
        self.min_area = min_area
        self.smoothing = smoothing
        
    def extract_skeleton(self, image_path):
        """从单张图像提取骨架"""
        try:
            # 读取图像
            if isinstance(image_path, str):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = image_path
                
            if image is None:
                return None
            
            # 中值滤波预处理
            denoised = cv2.medianBlur(image, 3)
            
            # 自适应二值化
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作增强连通性
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 提取轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建轮廓掩码
            contour_mask = np.zeros_like(binary)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= self.min_area]
            cv2.drawContours(contour_mask, valid_contours, -1, 255, -1)
            
            # 平滑处理
            if self.smoothing:
                contour_mask = self._smooth_contour(contour_mask)
            
            return contour_mask
            
        except Exception as e:
            logging.error(f"处理图像时出错: {e}")
            return None
    
    def _smooth_contour(self, contour_mask):
        """平滑轮廓掩码"""
        # 形态学闭操作填充小孔
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 形态学开操作去除毛刺
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel_open)
        
        return smoothed

class EnhancedBatchProcessor:
    """增强的批量处理器 - 支持综合清晰度评估"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.clarity_evaluator = EnhancedClarityEvaluator()
        self.skeleton_extractor = OracleBoneSkeletonExtractor(min_area=70, smoothing=True)
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        
    def process_clarity_evaluation(self):
        """批量处理增强版清晰度评估（集成噪声去除）"""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"输入目录不存在: {self.input_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        results = []
        
        print(f"开始增强版清晰度评估: {self.input_dir}")
        print("=" * 60)
        print("评估策略: 去除外源性设备噪声，保留内源性拓片特征")
        print("=" * 60)
        
        # 遍历所有子目录
        for root, dirs, files in os.walk(self.input_dir):
            # 计算相对路径
            rel_path = os.path.relpath(root, self.input_dir)
            
            # 处理当前目录的文件
            for file in files:
                if self._is_image_file(file):
                    input_path = os.path.join(root, file)
                    stats['total'] += 1
                    
                    # 使用新方法计算清晰度得分（集成噪声去除）
                    clarity_result = self.clarity_evaluator.calculate_detailed_assessment(input_path)
                    
                    if clarity_result is not None:
                        # 记录详细结果（包含噪声处理信息）
                        results.append({
                            'filename': file,
                            'subfolder': rel_path,
                            'total_score': clarity_result['total_score'],
                            'gradient_score': clarity_result['detailed_scores']['gradient'],
                            'frequency_score': clarity_result['detailed_scores']['frequency'],
                            'local_score': clarity_result['detailed_scores']['local'],
                            'noise_removed_ratio': clarity_result['noise_removed_ratio'],
                            'diagnosis': ';'.join(clarity_result['diagnosis']),
                            'recommendation': clarity_result['recommendation']
                        })
                        stats['success'] += 1
                        print(f"✅ 成功: {file} | 噪声去除: {clarity_result['noise_removed_ratio']:.1%} | 清晰度: {clarity_result['total_score']:.2f}")
                    else:
                        stats['failed'] += 1
                        stats['failed_files'].append(input_path)
                        print(f"❌ 失败: {file}")
        
        # 保存增强版清晰度得分结果
        if results:
            self._save_enhanced_clarity_results(results)
        
        # 生成增强版报告
        self._generate_enhanced_clarity_report(stats, results)
        return stats
    
    def process_skeleton_extraction(self):
        """批量处理骨架提取"""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"输入目录不存在: {self.input_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        print(f"开始骨架提取: {self.input_dir}")
        print("=" * 50)
        
        # 遍历所有子目录
        for root, dirs, files in os.walk(self.input_dir):
            # 计算相对路径
            rel_path = os.path.relpath(root, self.input_dir)
            
            # 创建对应的输出子目录
            if rel_path != '.':
                output_subdir = os.path.join(self.output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
            else:
                output_subdir = self.output_dir
            
            # 处理当前目录的文件
            for file in files:
                if self._is_image_file(file):
                    input_path = os.path.join(root, file)
                    stats['total'] += 1
                    
                    # 处理图像
                    skeleton = self.skeleton_extractor.extract_skeleton(input_path)
                    
                    if skeleton is not None:
                        # 生成输出文件名（保持原名，添加_skeleton后缀）
                        output_filename = self._get_output_filename(file)
                        output_path = os.path.join(output_subdir, output_filename)
                        
                        # 保存骨架图
                        cv2.imwrite(output_path, skeleton)
                        stats['success'] += 1
                        print(f"✅ 成功: {file} -> {output_filename}")
                    else:
                        stats['failed'] += 1
                        stats['failed_files'].append(input_path)
                        print(f"❌ 失败: {file}")
        
        # 生成报告
        self._generate_skeleton_report(stats)
        return stats
    
    def _is_image_file(self, filename):
        """检查是否为图像文件"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.supported_formats
    
    def _get_output_filename(self, filename):
        """生成输出文件名"""
        name, ext = os.path.splitext(filename)
        return f"{name}.png"
    
    def _save_enhanced_clarity_results(self, results):
        """保存增强版清晰度结果"""
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, "enhanced_clarity_scores.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"增强版清晰度得分已保存到: {csv_path}")
        
        # 打印统计摘要
        if not df.empty:
            print("\n增强版清晰度评估统计摘要:")
            print(f"平均综合得分: {df['total_score'].mean():.2f}")
            print(f"平均梯度得分: {df['gradient_score'].mean():.2f}")
            print(f"平均频域得分: {df['frequency_score'].mean():.2f}")
            print(f"平均局部得分: {df['local_score'].mean():.2f}")
            print(f"平均噪声去除比例: {df['noise_removed_ratio'].mean():.1%}")
            
            # 常见问题统计
            diagnosis_counts = df['diagnosis'].value_counts()
            print("\n常见问题统计:")
            for diagnosis, count in diagnosis_counts.items():
                print(f"  {diagnosis}: {count}次")
    
    def _generate_enhanced_clarity_report(self, stats, results):
        """生成增强版清晰度评估报告"""
        report_path = os.path.join(self.output_dir, "enhanced_clarity_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("甲骨文拓片增强版清晰度评估报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {self.input_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            f.write("评估方法: 多维度综合评估（梯度域+频域+局部自适应）\n")
            f.write("噪声处理: 智能分离外源性噪声，保留内源性特征\n")
            f.write("设计目标: 解决噪声干扰、缩放不一致、局部对比度差异\n\n")
            
            f.write("处理统计:\n")
            f.write(f"总文件数: {stats['total']}\n")
            f.write(f"成功评估: {stats['success']}\n")
            f.write(f"评估失败: {stats['failed']}\n")
            f.write(f"成功率: {stats['success']/stats['total']*100:.1f}%\n\n")
            
            if results:
                df = pd.DataFrame(results)
                f.write("得分统计:\n")
                f.write(f"综合得分范围: {df['total_score'].min():.2f} - {df['total_score'].max():.2f}\n")
                f.write(f"综合得分平均值: {df['total_score'].mean():.2f}\n")
                f.write(f"梯度得分平均值: {df['gradient_score'].mean():.2f}\n")
                f.write(f"频域得分平均值: {df['frequency_score'].mean():.2f}\n")
                f.write(f"局部得分平均值: {df['local_score'].mean():.2f}\n")
                f.write(f"噪声去除比例平均值: {df['noise_removed_ratio'].mean():.1%}\n\n")
                
                # 质量分布
                f.write("质量分布 (基于去噪后评估):\n")
                excellent = len(df[df['total_score'] >= 80])
                good = len(df[(df['total_score'] >= 60) & (df['total_score'] < 80)])
                poor = len(df[df['total_score'] < 60])
                f.write(f"优秀(≥80): {excellent}张 ({excellent/len(df)*100:.1f}%)\n")
                f.write(f"良好(60-79): {good}张 ({good/len(df)*100:.1f}%)\n")
                f.write(f"需改进(<60): {poor}张 ({poor/len(df)*100:.1f}%)\n\n")
                
                # 常见问题分析
                f.write("常见问题分析:\n")
                diagnosis_counts = df['diagnosis'].value_counts()
                for diagnosis, count in diagnosis_counts.items():
                    f.write(f"{diagnosis}: {count}次\n")
                
                # 改进建议统计
                f.write("\n改进建议统计:\n")
                recommendation_counts = df['recommendation'].value_counts()
                for recommendation, count in recommendation_counts.items():
                    f.write(f"{recommendation}: {count}次\n")
            
            if stats['failed_files']:
                f.write("\n失败文件列表:\n")
                for file in stats['failed_files']:
                    f.write(f"- {file}\n")
        
        print(f"\n📊 增强版清晰度报告已保存: {report_path}")
    
    def _generate_skeleton_report(self, stats):
        """生成骨架提取报告"""
        report_path = os.path.join(self.output_dir, "skeleton_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("甲骨文拓片骨架提取报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {self.input_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            f.write("处理统计:\n")
            f.write(f"总文件数: {stats['total']}\n")
            f.write(f"成功提取: {stats['success']}\n")
            f.write(f"提取失败: {stats['failed']}\n")
            f.write(f"成功率: {stats['success']/stats['total']*100:.1f}%\n")
            
            if stats['failed_files']:
                f.write("\n失败文件列表:\n")
                for file in stats['failed_files']:
                    f.write(f"- {file}\n")
        
        print(f"\n📊 骨架提取报告已保存: {report_path}")

def main():
    """主函数"""
    print("甲骨文拓片处理系统 - 增强版")
    print("=" * 50)
    print("新增功能: 多维度清晰度评估（抗噪声、尺度不变、局部自适应）")
    print("=" * 50)
    
    # 配置路径 - 修改为您需要的路径
    input_directory = "/work/home/succuba/BRISQUE/data/tapian"  # 输入目录
    output_directory = "/work/home/succuba/BRISQUE/enhanced_results"  # 输出目录
    
    # 检查输入目录
    if not os.path.exists(input_directory):
        print(f"错误: 输入目录不存在: {input_directory}")
        print("请修改 input_directory 变量")
        return
    
    # 创建增强版处理器
    processor = EnhancedBatchProcessor(
        input_dir=input_directory,
        output_dir=output_directory
    )
    
    try:
        # 第一步：增强版清晰度评估
        print("\n" + "=" * 60)
        print("开始增强版清晰度评估")
        print("=" * 60)
        clarity_stats = processor.process_clarity_evaluation()
        
        print("\n" + "=" * 60)
        print("增强版清晰度评估完成!")
        print(f"成功评估: {clarity_stats['success']}/{clarity_stats['total']} 个文件")
        print(f"成功率: {clarity_stats['success']/clarity_stats['total']*100:.1f}%")
        
        # 第二步：骨架提取
        print("\n" + "=" * 60)
        print("开始骨架提取")
        print("=" * 60)
        skeleton_stats = processor.process_skeleton_extraction()
        
        print("\n" + "=" * 60)
        print("骨架提取完成!")
        print(f"成功提取: {skeleton_stats['success']}/{skeleton_stats['total']} 个文件")
        print(f"成功率: {skeleton_stats['success']/skeleton_stats['total']*100:.1f}%")
        print(f"结果保存在: {output_directory}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 单文件测试函数
def test_single_file():
    """测试单文件处理 - 增强版（保存去噪效果图）"""
    test_image = "/work/home/succuba/BRISQUE/data/tapian/1/1.png"  # 测试文件
    
    if not os.path.exists(test_image):
        print("❌ 测试文件不存在")
        return
    
    print("🧪 开始单文件测试（保存去噪效果图）...")
    print("=" * 60)
    
    # 1. 首先单独处理噪声去除并保存结果
    print("🔍 进行外源性噪声去除...")
    noise_processor = IntelligentNoiseProcessor()
    denoised_img, noise_mask = noise_processor.remove_exogenous_noise(test_image)
    
    if denoised_img is not None and noise_mask is not None:
        # 创建专门的输出目录用于保存去噪结果
        output_dir = "/work/home/succuba/BRISQUE/tmp/oracle_bone_debug"
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名（不含扩展名）
        file_name = os.path.splitext(os.path.basename(test_image))[0]
        
        # 保存去噪后的图像
        denoised_path = f"{output_dir}/{file_name}_denoised.png"
        cv2.imwrite(denoised_path, denoised_img)
        
        # 保存噪声掩码（可视化噪声区域）
        noise_mask_path = f"{output_dir}/{file_name}_noise_mask.png"
        cv2.imwrite(noise_mask_path, noise_mask)
        
        print(f"✅ 去噪图像已保存: {denoised_path}")
        print(f"✅ 噪声掩码已保存: {noise_mask_path}")
        print(f"📁 所有调试文件保存在: {output_dir}")
    else:
        print("❌ 噪声处理失败，无法保存去噪效果图")
        return  # 如果噪声处理失败，提前退出
    
    print("\n" + "=" * 60)
    print("📊 进行清晰度评估...")
    print("=" * 60)
    
    # 2. 继续进行原有的清晰度评估（使用去噪后的图像）
    evaluator = EnhancedClarityEvaluator()
    clarity_result = evaluator.calculate_detailed_assessment(test_image)
    
    if clarity_result:
        print(f"✅ 清晰度评估成功!")
        print(f"   综合得分: {clarity_result['total_score']:.2f}")
        print(f"   梯度得分: {clarity_result['detailed_scores']['gradient']:.2f}")
        print(f"   频域得分: {clarity_result['detailed_scores']['frequency']:.2f}")
        print(f"   局部得分: {clarity_result['detailed_scores']['local']:.2f}")
        print(f"   噪声去除比例: {clarity_result['noise_removed_ratio']:.1%}")
        print(f"   诊断结果: {', '.join(clarity_result['diagnosis'])}")
        print(f"   改进建议: {clarity_result['recommendation']}")
    else:
        print("❌ 清晰度评估失败")
    
    print("\n" + "=" * 60)
    print("🦴 进行骨架提取...")
    print("=" * 60)
    
    # 3. 进行骨架提取
    extractor = OracleBoneSkeletonExtractor()
    skeleton = extractor.extract_skeleton(test_image)
    
    if skeleton is not None:
        skeleton_path = f"{output_dir}/{file_name}_skeleton.png"
        cv2.imwrite(skeleton_path, skeleton)
        print(f"✅ 骨架提取完成! 结果保存到: {skeleton_path}")
        
        # 可选：显示处理前后的对比信息
        print("\n📋 处理摘要:")
        print(f"   原始图像: {test_image}")
        print(f"   去噪图像: {denoised_path}")
        print(f"   噪声掩码: {noise_mask_path}")
        print(f"   骨架图像: {skeleton_path}")
    else:
        print("❌ 骨架提取失败")
    
    print("\n" + "=" * 60)
    print("🎉 单文件测试完成! 所有结果已保存到调试目录")

if __name__ == "__main__":
    # 运行批量处理
    # main()
    
    # 如果要测试单文件，取消下面的注释
    test_single_file()