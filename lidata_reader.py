# -*- coding: utf-8 -*-
"""
Module đơn giản để đọc dữ liệu LiData và chuyển thành DataFrame
"""

import pandas as pd
import numpy as np
import gvlib
from pathlib import Path
import locale


def read_lidata_file(lidata_path):
    """
    Đọc file LiData và chuyển đổi thành DataFrame pandas
    
    Parameters:
        lidata_path (str or Path): Đường dẫn đến file LiData
    
    Returns:
        pandas.DataFrame: DataFrame chứa dữ liệu điểm từ file LiData
    """
    # Chuyển đường dẫn thành chuỗi
    inputfile = str(lidata_path)
    
    try:
        # Khởi tạo đối tượng đọc LiData
        reader = gvlib.LidataReader(inputfile)
        if not reader.open():
            raise IOError(f"Không thể mở file {inputfile}")
        
        # Lấy toàn bộ dữ liệu
        # đọc toàn bộ dữ liệu từ file LiData vào bộ nhớ
        reader.read()
        '''
            return về một đối tượng LiData chứa dữ liệu điểm, bao gồm các thuộc tính như tọa độ (xyz), phân loại (classification), cường độ (intensity), màu sắc (rgb) và nhiều thuộc tính khác.
            dạng lưới 3D với các điểm có tọa độ (x, y, z) và các thuộc tính khác như phân loại, cường độ và màu sắc.
                - lidata.xyz: Tọa độ 3D của các điểm trong không gian (x, y, z).
                - lidata.classification: Phân loại của các điểm (ví dụ: mặt đất, cây cối, tòa nhà, v.v.).
                - lidata.intensity: Cường độ của các điểm (thường là độ sáng hoặc độ mạnh của tín hiệu phản hồi từ laser).
                - lidata.rgb: Màu sắc của các điểm (thường là giá trị RGB).
        '''
        lidata = reader.tile()
        
        # Tạo dictionary để lưu trữ dữ liệu
        point_data = {}
        
        # Lấy giá trị tọa độ
        pos = lidata.xyz
        x_values = pos[:, 0]
        y_values = pos[:, 1]
        z_values = pos[:, 2]
        
        # Lấy giá trị phân loại và cường độ
        classification = lidata.classification
        intensity = lidata.intensity
        
        # Lấy giá trị màu sắc RGB
        rgb = lidata.rgb
        red = rgb[:, 0]
        green = rgb[:, 1]
        blue = rgb[:, 2]
        
        # Lấy giá trị min từ dữ liệu để chuẩn hóa (tương tự như code của bạn)
        x_min = np.min(x_values)
        y_min = np.min(y_values)
        z_min = np.min(z_values)
        
        # Tính giá trị chuẩn hóa
        x_normalized = x_values - x_min
        y_normalized = y_values - y_min
        z_normalized = z_values - z_min
        
        # Lưu dữ liệu vào dictionary
        point_data['X'] = x_normalized
        point_data['Y'] = y_normalized
        point_data['Z'] = z_normalized
        point_data['Intensity'] = intensity
        point_data['R'] = red
        point_data['G'] = green
        point_data['B'] = blue
        point_data['Classification'] = classification
        
        # Tạo DataFrame từ dictionary
        lidar_df = pd.DataFrame(point_data)
        
        locale.setlocale(locale.LC_ALL, '')  # Thiết lập locale theo hệ thống
        print(f"Đã đọc file {inputfile}")
        print(f"Số điểm: {locale.format_string('%d', len(lidar_df), grouping=True)}")
        print(f"Các cột có sẵn: {', '.join(lidar_df.columns)}")
        
        return lidar_df
        
    except Exception as e:
        print(f"Lỗi khi đọc file {inputfile}: {str(e)}")
        raise


def read_multiple_lidata_files(path):
    """
    Đọc một file LiData cụ thể hoặc tất cả file LiData từ một thư mục.
    
    Parameters:
        path (str or Path): Đường dẫn đến một file LiData hoặc một thư mục chứa các file LiData
    
    Returns:
        pandas.DataFrame: DataFrame từ file LiData hoặc kết hợp từ tất cả file LiData
    """
    # Chuyển đường dẫn thành đối tượng Path
    path_obj = Path(path)
    
    # Kiểm tra xem path có phải là file hay thư mục
    if path_obj.is_file():
        # Nếu là file, kiểm tra xem có phải là file LiData không
        if path_obj.suffix.lower() == '.lidata':
            print(f"Đang xử lý file LiData: {path_obj}")
            return read_lidata_file(str(path_obj))
        else:
            raise ValueError(f"File {path_obj} không phải là file LiData.")
    elif path_obj.is_dir():
        # Nếu là thư mục, tìm tất cả các file LiData trong thư mục và thư mục con
        lidata_files = list(path_obj.rglob("*.lidata"))
        if not lidata_files:
            raise FileNotFoundError(f"Không tìm thấy file LiData nào trong thư mục {path_obj}.")
        
        # Khởi tạo DataFrame tổng hợp
        combined_df = None
        
        # Xử lý tất cả các file LiData
        print(f"Tìm thấy {len(lidata_files)} file LiData. Bắt đầu xử lý...")
        
        for i, lidata_file in enumerate(lidata_files):
            print(f"Đang xử lý file {i+1}/{len(lidata_files)}: {lidata_file}")
            
            # Chuyển đổi LiData sang DataFrame
            current_df = read_lidata_file(str(lidata_file))
            
            # Thêm vào DataFrame tổng hợp
            if combined_df is None:
                combined_df = current_df
            else:
                combined_df = pd.concat([combined_df, current_df], ignore_index=True)
        
        print(f"Đã xử lý xong {len(lidata_files)} file LiData.")
        print(f"Tổng số điểm: {len(combined_df)}")
        
        return combined_df
    else:
        raise FileNotFoundError(f"Đường dẫn {path} không tồn tại.")