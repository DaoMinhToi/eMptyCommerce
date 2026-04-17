"""
Mock Data - Xử lý Implicit Feedback (Phản hồi ẩn) trong Hệ thống Gợi ý

Implicit Feedback: Là những hành động người dùng thực hiện mà không rõ ràng cấp 1-5 sao
được ghi lại một cách không trực tiếp (ẩn):
- Click xem sản phẩm
- Thêm vào giỏ hàng
- Mua sản phẩm
- Share sản phẩm
- v.v...

Nhiệm vụ: Chuyển đổi các hành động ẩn thành rating hiển thị (explicit rating) 
để sử dụng trong mô hình Collaborative Filtering
"""

import pandas as pd
import numpy as np


def convert_action_to_rating(action):
    """
    Chuyển đổi loại hành động (action) thành mức đánh giá (rating).
    
    Mapping:
    - "Có mua hàng" -> 5 sao (hành động mạnh nhất, cho thấy người dùng yêu thích)
    - "Thêm vào giỏ" -> 3 sao (hành động trung bình, cho thấy người dùng quan tâm)
    - "Click xem" -> 1 sao (hành động yếu nhất, chỉ tỏ ra tò mò)
    
    Args:
        action (str): Loại hành động người dùng
    
    Returns:
        int: Mức đánh giá từ 1-5 sao
    """
    # Tạo dictionary mapping giữa action và rating
    action_rating_map = {
        "Có mua hàng": 5,    # Hành động quyết định nhất
        "Thêm vào giỏ": 3,    # Hành động quan tâm
        "Click xem": 1        # Hành động khám phá
    }
    
    # Trả về rating tương ứng, nếu action không tồn tại thì return 0
    return action_rating_map.get(action, 0)


def main():
    """
    Hàm chính - Tạo mock data và minh chứng xử lý Implicit Feedback
    """
    
    print("=" * 100)
    print("📊 MINH CHỨNG XỬ LÝ IMPLICIT FEEDBACK (PHẢN HỒI ẨN)")
    print("=" * 100)
    
    # ============ BƯỚC 1: TẠO DỮ LIỆU GIẢ LẬP ============
    print("\n🔧 Bước 1: Tạo dữ liệu giả lập (Mock Data) từ các hành động người dùng")
    print("-" * 100)
    
    # Tạo dữ liệu giả lập
    mock_data = {
        'user_id': [101, 102, 103, 104, 105],                              # ID khách hàng
        'product_id': [1001, 1002, 1001, 1003, 1002],                     # ID sản phẩm
        'action': [
            "Có mua hàng",      # Khách 101 mua sản phẩm 1001
            "Click xem",        # Khách 102 chỉ click xem sản phẩm 1002
            "Thêm vào giỏ",     # Khách 103 thêm sản phẩm 1001 vào giỏ
            "Có mua hàng",      # Khách 104 mua sản phẩm 1003
            "Click xem"         # Khách 105 click xem sản phẩm 1002
        ]
    }
    
    # Tạo DataFrame từ dữ liệu giả lập
    df_implicit = pd.DataFrame(mock_data)
    
    print("\n📋 DỮ LIỆU IMPLICIT FEEDBACK (Phản hồi ẩn - chỉ có hành động):")
    print(df_implicit.to_string(index=False))
    
    print(f"\n📌 Thông tin dữ liệu:")
    print(f"   - Tổng số hành động: {len(df_implicit)}")
    print(f"   - Số lượng khách hàng duy nhất: {df_implicit['user_id'].nunique()}")
    print(f"   - Số lượng sản phẩm duy nhất: {df_implicit['product_id'].nunique()}")
    print(f"   - Các loại hành động: {df_implicit['action'].unique().tolist()}")
    
    # ============ BƯỚC 2: CHUYỂN ĐỔI ACTION THÀNH RATING ============
    print("\n\n🔄 Bước 2: Chuyển đổi Implicit Feedback → Explicit Rating")
    print("-" * 100)
    
    print("\n📐 Mapping (Quy tắc chuyển đổi):")
    print("   ├─ 'Có mua hàng' → 5 sao  (Người dùng thực sự yêu thích sản phẩm)")
    print("   ├─ 'Thêm vào giỏ' → 3 sao (Người dùng quan tâm, có ý định mua)")
    print("   └─ 'Click xem' → 1 sao   (Người dùng chỉ tò mò, chưa quyết định)")
    
    # Áp dụng hàm convert_action_to_rating lên cột action
    df_implicit['rating_sao'] = df_implicit['action'].apply(convert_action_to_rating)
    
    # ============ BƯỚC 3: IN KẾT QUẢ ============
    print("\n\n✅ DỮ LIỆU EXPLICIT RATING (Phản hồi rõ ràng - đã quy đổi):")
    print(df_implicit.to_string(index=False))
    
    # ============ BƯỚC 4: THỐNG KÊ ============
    print("\n\n📊 THỐNG KÊ CÁC LOẠI HÀNH ĐỘNG:")
    print("-" * 100)
    
    action_stats = df_implicit.groupby('action').agg({
        'rating_sao': ['count', 'mean'],
        'user_id': 'nunique'
    }).round(2)
    
    print("\nBảng thống kê chi tiết:")
    print(action_stats)
    
    # Tính số lượng hành động từng loại
    print("\n📈 Phân bố hành động:")
    action_counts = df_implicit['action'].value_counts()
    for action, count in action_counts.items():
        rating = convert_action_to_rating(action)
        print(f"   • '{action}': {count} hành động → Rating {rating}⭐")
    
    # ============ BƯỚC 5: GHI CHÚ ============
    print("\n\n💡 GHI CHÚ CÓ GIÁO DỤC:")
    print("-" * 100)
    print("""
    Implicit Feedback là gì?
    ├─ Là những hành động mà người dùng thực hiện KHÔNG CÙNG MỤC ĐÍCH đánh giá
    ├─ Ví dụ: Click xem, thêm giỏ, mua sản phẩm, v.v...
    └─ Hệ thống phải tự "đoán" mức độ thích của người dùng từ các hành động này

    Ưu điểm:
    ├─ Dễ thu thập (tự động ghi lại từ hệ thống)
    ├─ Không cần người dùng xếp hạng rõ ràng
    └─ Dữ liệu phong phú và liên tục

    Nhược điểm:
    ├─ Khó xác định chính xác mức độ yêu thích
    ├─ Có thể có nhiễu (người mua nhầm, click lỡ, v.v...)
    └─ Cần phải transform thành explicit rating để sử dụng

    Ứng dụng trong Collaborative Filtering:
    └─ Rating được tạo từ implicit feedback sẽ được dùng thay cho explicit rating
       để huấn luyện mô hình SVD và dự đoán rating cho sản phẩm chưa xem
    """)
    
    print("\n" + "=" * 100)
    print("✅ MINH CHỨNG XỬ LÝ IMPLICIT FEEDBACK HOÀN TẤT")
    print("=" * 100)
    
    # Trả về DataFrame để có thể sử dụng sau
    return df_implicit


if __name__ == "__main__":
    df_result = main()
    
    # Lưu kết quả vào file CSV (tùy chọn)
    # df_result.to_csv('implicit_feedback_converted.csv', index=False)
    # print("\n💾 Dữ liệu đã được lưu vào: implicit_feedback_converted.csv")
