# -*- coding: utf-8 -*-
"""
Module tách file LiData thành train và test theo tỷ lệ 80-20
Đọc file LiData và xuất ra file LAS
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gvlib
import laspy
import locale
import sys
# Thêm đường dẫn thư mục cha vào PATH để có thể import module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
# Import hàm đọc từ module có sẵn
from lidata_reader import read_lidata_file


def create_las_from_dataframe(df, output_path):
    """
    Tạo file LAS mới từ DataFrame
    
    Parameters:
        df (DataFrame): DataFrame chứa dữ liệu điểm
        output_path (Path): Đường dẫn file LAS đầu ra
    
    Returns:
        Path: Đường dẫn file đã tạo
    """
    # Tạo thư mục nếu chưa tồn tại
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Tạo header cho file LAS mới
    header = laspy.LasHeader(point_format=2, version="1.2")
    
    # Tạo đối tượng LAS
    las = laspy.LasData(header)
    
    # Thiết lập tọa độ XYZ
    las.x = df['X'].values
    las.y = df['Y'].values
    las.z = df['Z'].values
    
    # Thiết lập classification nếu có
    if 'Classification' in df.columns:
        las.classification = df['Classification'].values.astype(np.uint8)
    
    # Thiết lập intensity nếu có
    if 'Intensity' in df.columns:
        las.intensity = df['Intensity'].values.astype(np.uint16)
    
    # Thiết lập RGB nếu có
    if all(col in df.columns for col in ['R', 'G', 'B']):
        las.red = df['R'].values.astype(np.uint16)
        las.green = df['G'].values.astype(np.uint16)
        las.blue = df['B'].values.astype(np.uint16)
    
    # Ghi file LAS
    las.write(str(output_path))
    
    print(f"Đã tạo file LAS: {output_path}")
    print(f"Số điểm: {len(df):,}")
    
    return output_path

def split_lidata_to_las(input_file, output_dir=None, train_name=None, test_name=None, sample_size=10):
    """
    Tách file LiData thành train và test dưới dạng file LAS
    Duyệt từng cụm 5 điểm: 4 điểm cho vào Train, 1 điểm cho vào Test

    Parameters:
        input_file (str or Path): Đường dẫn đến file LiData cần tách
        output_dir (str or Path, optional): Thư mục đầu ra
        train_name (str, optional): Tên file train
        test_name (str, optional): Tên file test
        sample_size (int, optional): Số lượng mẫu để hiển thị

    Returns:
        tuple: (train_path, test_path) - Đường dẫn đến các file LAS đã tạo
    """
    # Chuyển đổi đường dẫn
    input_path = Path(input_file)

    # Kiểm tra file tồn tại
    if not input_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {input_path}")

    if not input_path.suffix.lower() == '.lidata':
        raise ValueError(f"File không phải định dạng LiData: {input_path}")

    # Xác định thư mục đầu ra
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Xác định tên file đầu ra (với đuôi .las)
    base_name = input_path.stem

    if train_name is None:
        train_name = f"{base_name}_train.las"
    else:
        if not train_name.endswith('.las'):
            train_name += '.las'

    if test_name is None:
        test_name = f"{base_name}_test.las"
    else:
        if not test_name.endswith('.las'):
            test_name += '.las'

    train_path = output_dir / train_name
    test_path = output_dir / test_name

    # Đọc dữ liệu từ file LiData
    print(f"\nĐang đọc file LiData: {input_path}")
    lidar_df = read_lidata_file(input_path)

    # In ra thông tin DataFrame gốc
    print("\n=== THÔNG TIN DATAFRAME GỐC ===")
    print(f"Số lượng bản ghi: {len(lidar_df):,}")
    print(f"Các cột: {list(lidar_df.columns)}")
    print("\nMẫu dữ liệu gốc (đầu file):")
    print(lidar_df.head(sample_size).to_string())

    # Tách dữ liệu theo cụm 5 điểm
    n_points = len(lidar_df)
    n_clusters = n_points // 5
    n_remainder = n_points % 5

    print(f"\nThông tin tách dữ liệu:")
    print(f"Tổng số điểm: {n_points:,}")
    print(f"Số cụm 5 điểm: {n_clusters:,}")
    print(f"Số điểm dư: {n_remainder}")

    # Tạo mảng indices cho train và test
    train_indices = []
    test_indices = []

    # Xử lý các cụm 5 điểm
    for i in range(n_clusters):
        cluster_start = i * 5
        # 4 điểm đầu cho train, 1 điểm cuối cho test
        train_indices.extend(range(cluster_start, cluster_start + 4))
        test_indices.append(cluster_start + 4)

    # Xử lý các điểm dư (nếu có) - đưa vào train
    if n_remainder > 0:
        remainder_start = n_clusters * 5
        train_indices.extend(range(remainder_start, remainder_start + n_remainder))

    # In ra một số mẫu của chỉ số được chọn
    max_sample = min(20, len(train_indices), len(test_indices))
    print(f"\nMẫu chỉ số train (20 chỉ số đầu): {train_indices[:max_sample]}")
    print(f"Mẫu chỉ số test (20 chỉ số đầu): {test_indices[:max_sample]}")

    # Tạo DataFrame train và test
    train_df = lidar_df.iloc[train_indices].reset_index(drop=True)
    test_df = lidar_df.iloc[test_indices].reset_index(drop=True)

    print(f"\nKết quả tách:")
    print(f"Số điểm train: {len(train_df):,} ({len(train_df) / n_points * 100:.1f}%)")
    print(f"Số điểm test: {len(test_df):,} ({len(test_df) / n_points * 100:.1f}%)")

    # In ra mẫu dữ liệu từ train và test
    print("\n=== MẪU DỮ LIỆU SAU KHI TÁCH ===")
    print("\nMẫu dữ liệu TRAIN (đầu file):")
    print(train_df.head(sample_size).to_string())

    print("\nMẫu dữ liệu TEST (đầu file):")
    print(test_df.head(sample_size).to_string())

    # In ra chi tiết cho cụm đầu tiên để kiểm tra
    print("\n=== KIỂM TRA CỤM ĐẦU TIÊN ===")
    print("5 điểm đầu tiên trong file gốc:")
    print(lidar_df.iloc[:5].to_string())
    print("\n4 điểm đầu tiên trong TRAIN (từ cụm đầu tiên):")
    print(train_df.iloc[:4].to_string())
    print("\nĐiểm đầu tiên trong TEST (từ cụm đầu tiên):")
    print(test_df.iloc[0].to_string())

    # Hiển thị phân phối lớp trong train và test
    if 'Classification' in lidar_df.columns:
        print("\nPhân phối lớp trong tập train:")
        train_class_dist = train_df['Classification'].value_counts().sort_index()
        for cls, count in train_class_dist.items():
            print(f"  Lớp {cls}: {count:,} điểm ({count / len(train_df) * 100:.2f}%)")

        print("\nPhân phối lớp trong tập test:")
        test_class_dist = test_df['Classification'].value_counts().sort_index()
        for cls, count in test_class_dist.items():
            print(f"  Lớp {cls}: {count:,} điểm ({count / len(test_df) * 100:.2f}%)")

    # Tạo file LAS
    print(f"\nĐang tạo file train LAS: {train_path}")
    try:
        train_path = create_las_from_dataframe(train_df, train_path)
    except Exception as e:
        print(f"Lỗi khi tạo file train: {e}")
        # Fallback to CSV
        train_path = train_path.with_suffix('.csv')
        train_df.to_csv(train_path, index=False)
        print(f"Đã lưu dưới dạng CSV: {train_path}")

    print(f"\nĐang tạo file test LAS: {test_path}")
    try:
        test_path = create_las_from_dataframe(test_df, test_path)
    except Exception as e:
        print(f"Lỗi khi tạo file test: {e}")
        # Fallback to CSV
        test_path = test_path.with_suffix('.csv')
        test_df.to_csv(test_path, index=False)
        print(f"Đã lưu dưới dạng CSV: {test_path}")

    print("\nHoàn tất tách file!")

    return train_path, test_path
def main():
    """
    Hàm main để chạy từ command line
    """
    print("=== CHƯƠNG TRÌNH TÁCH FILE LIDATA THÀNH LAS ===\n")
    
    # Liệt kê các file LiData
    current_dir = Path(".")
    data_dir = Path("./data")
    
    lidata_files = []
    for pattern in ["*.lidata", "*.LiData"]:
        if current_dir.exists():
            lidata_files.extend(list(current_dir.glob(pattern)))
        if data_dir.exists():
            lidata_files.extend(list(data_dir.glob(pattern)))
    
    # Loại bỏ file trùng
    lidata_files = list(set(lidata_files))
    
    if not lidata_files:
        print("\nKhông tìm thấy file LiData nào!")
        input_file = input("Nhập đường dẫn đến file LiData: ")
        if not Path(input_file).exists():
            print(f"File {input_file} không tồn tại!")
            return
    else:
        print("\nCác file LiData tìm thấy:")
        for i, file in enumerate(lidata_files, 1):
            print(f"  {i}. {file}")
        
        while True:
            choice = input(f"\nChọn file (1-{len(lidata_files)}) hoặc nhập đường dẫn khác: ")
            
            if choice.isdigit() and 1 <= int(choice) <= len(lidata_files):
                input_file = lidata_files[int(choice) - 1]
                break
            elif Path(choice).exists():
                input_file = choice
                break
            else:
                print("Lựa chọn không hợp lệ!")
    
    # Hỏi về thư mục đầu ra
    use_custom_output = input("\nSử dụng thư mục đầu ra tùy chỉnh? (y/n) [n]: ")
    output_dir = None
    
    if use_custom_output.lower() == 'y':
        output_dir = input("Nhập thư mục đầu ra: ")
        if not output_dir:
            output_dir = None
    
    # Hỏi về tên file
    use_custom_names = input("Sử dụng tên file tùy chỉnh? (y/n) [n]: ")
    train_name = None
    test_name = None
    
    if use_custom_names.lower() == 'y':
        train_name = input("Tên file train (để trống = mặc định): ")
        test_name = input("Tên file test (để trống = mặc định): ")
        if not train_name:
            train_name = None
        if not test_name:
            test_name = None
    
    # Thực hiện tách file
    try:
        train_path, test_path = split_lidata_to_las(
            input_file,
            output_dir=output_dir,
            train_name=train_name,
            test_name=test_name
        )
        
        print(f"\n✓ Thành công!")
        print(f"File train: {train_path}")
        print(f"File test: {test_path}")
        
    except Exception as e:
        print(f"\n✗ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()