"""
Giao diện Streamlit cho Hệ thống Gợi ý Sản phẩm Hybrid
Minh họa Cold-Start Problem & Hybrid Recommendation System
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from recommender import HybridRecommender


# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(
    page_title="eMpTyCommerce - Gợi ý thông minh",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lấy đường dẫn của thư mục app hiện tại
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, 'data')


# ==================== CACHE RESOURCE ====================
# Khởi tạo HybridRecommender 1 lần duy nhất (tránh load lại mô hình)
@st.cache_resource
def load_recommender_model():
    """
    Load mô hình HybridRecommender một lần duy nhất.
    Streamlit sẽ cache kết quả để tránh phải huấn luyện lại model mỗi lần reload.
    """
    # Thay đổi thư mục làm việc để đảm bảo load dữ liệu đúng
    original_cwd = os.getcwd()
    os.chdir(APP_DIR)
    try:
        model = HybridRecommender()
    finally:
        os.chdir(original_cwd)
    return model


@st.cache_resource
def load_book_data():
    """
    Load dữ liệu sách từ clean_book_data.csv.
    Cache để tránh đọc file liên tục.
    """
    try:
        return pd.read_csv(os.path.join(DATA_DIR, 'clean_book_data.csv'))
    except FileNotFoundError:
        st.error(f"❌ Không tìm thấy file {os.path.join(DATA_DIR, 'clean_book_data.csv')}")
        return None


@st.cache_resource
def load_reviews_data():
    """
    Load dữ liệu đánh giá từ clean_reviews.csv.
    Cache để tránh đọc file liên tục.
    """
    try:
        return pd.read_csv(os.path.join(DATA_DIR, 'clean_reviews.csv'))
    except FileNotFoundError:
        st.error(f"❌ Không tìm thấy file {os.path.join(DATA_DIR, 'clean_reviews.csv')}")
        return None


# ==================== KHỞI TẠO DỮ LIỆU ====================
with st.spinner("⏳ Đang tải mô hình và dữ liệu..."):
    recommender = load_recommender_model()
    book_data = load_book_data()
    reviews_data = load_reviews_data()

# Kiểm tra dữ liệu
if book_data is None or reviews_data is None:
    st.error("❌ Lỗi: Không thể tải dữ liệu. Vui lòng kiểm tra file trong thư mục data/")
    st.stop()

# Danh sách khách hàng duy nhất
unique_customers = sorted(reviews_data['customer_id'].unique().tolist())
customer_dict = {cid: f"Customer {cid}" for cid in unique_customers}

# Danh sách sách với product_id
book_dict = {row['product_id']: row['title'] for _, row in book_data.iterrows()}


# ==================== SIDEBAR - THANH ĐIỀU HƯỚNG ====================
with st.sidebar:
    st.title("📚 eMpTyCommerce")
    st.markdown("---")
    
    # Thông tin luận văn
    st.subheader("📋 Thông tin Luận Văn")
    st.info("""
    **Đề tài:** Hệ thống gợi ý sản phẩm thương mại điện tử bằng Tiếng Việt
    
    **Phương pháp:** Hybrid Model (Content-Based + Collaborative Filtering)
    
    **Mô hình:** TF-IDF + SVD
    """)
    
    st.markdown("---")
    st.subheader("👤 Thông tin Sinh viên")
    st.write("**Tên sinh viên:** [Đào Minh Tới]")
    st.write("**Giáo viên hướng dẫn:** [ThS. BÙI THỊ DIỄM TRINH]")
    
    st.markdown("---")
    
    # Chọn loại khách hàng
    st.subheader("🎯 Chọn Kịch Bản")
    customer_type = st.radio(
        "Loại khách hàng:",
        ["👥 Khách hàng cũ", "🆕 Khách hàng mới (Cold-Start)"]
    )
    
    if customer_type == "👥 Khách hàng cũ":
        selected_customer = st.selectbox(
            "🔑 Đăng nhập với ID Khách hàng:",
            unique_customers,
            format_func=lambda x: customer_dict[x]
        )
    else:
        selected_customer = None
    
    st.markdown("---")
    
    # Thống kê dữ liệu
    st.subheader("📊 Thống kê dữ liệu")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💼 Sách", len(book_data))
        st.metric("⭐ Đánh giá", len(reviews_data))
    with col2:
        st.metric("👥 Khách hàng", reviews_data['customer_id'].nunique())
        st.metric("⭐ Trung bình", f"{reviews_data['rating'].mean():.2f}")


# ==================== MAIN CONTENT ====================
st.title("🎯 Hệ thống Gợi ý Sản phẩm Thương mại Điện tử")
st.markdown(
    "**Giải pháp:** Hybrid Model kết hợp Content-Based & Collaborative Filtering "
    "để xử lý Cold-Start Problem"
)
st.markdown("---")


# ============ KỊCH BẢN A: KHÁCH HÀNG MỚI (COLD-START) ============
if customer_type == "🆕 Khách hàng mới (Cold-Start)":
    st.header("🆕 Kịch bản: Khách hàng mới (Cold-Start)")
    
    # Thông báo
    st.info(
        "🔄 **Hệ thống nhận diện:** Đây là người dùng mới (chưa có lịch sử đánh giá)\n\n"
        "→ Tự động chuyển sang mô hình **Content-Based Filtering 100%**\n\n"
        "→ Gợi ý dựa trên **sách bạn đang quan tâm**"
    )
    
    # Chọn sách tham chiếu
    st.subheader("📖 Bước 1: Chọn cuốn sách bạn đang quan tâm")
    selected_book_id = st.selectbox(
        "Chọn quyển sách:",
        book_data['product_id'].tolist(),
        format_func=lambda x: book_dict.get(x, f"Sách {x}"),
        key="cold_start_book"
    )
    
    # Lấy thông tin sách được chọn
    selected_book = book_data[book_data['product_id'] == selected_book_id].iloc[0]
    
    st.subheader("📚 Sách bạn đang xem:")
    col1, col2, col3 = st.columns([2, 3, 5])
    
    with col1:
        # Hiển thị hình ảnh bìa sách
        if pd.notna(selected_book['cover_link']) and selected_book['cover_link'] != '':
            st.image(selected_book['cover_link'], use_container_width=True)
        else:
            st.image("https://picsum.photos/200/300?random=1", use_container_width=True)
    
    with col2:
        st.write(f"**Tên sách:** {selected_book['title']}")
        st.write(f"**Thể loại:** {selected_book['category']}")
    
    with col3:
        st.write("")  # Khoảng trống
    
    # Nút lấy gợi ý
    if st.button("🔍 Lấy gợi ý sách tương tự", key="btn_cold_start"):
        with st.spinner("⏳ Đang tính toán gợi ý dựa trên Content-Based Filtering..."):
            try:
                recommendations = recommender.get_content_based_recommendations(
                    selected_book_id,
                    top_n=10
                )
                
                if recommendations.empty:
                    st.warning("⚠️ Không tìm thấy sách tương tự")
                else:
                    st.success(f"✅ Tìm thấy {len(recommendations)} cuốn sách tương tự!")
                    
                    st.subheader("📚 Bước 2: Sách được gợi ý dựa trên nội dung tương tự")
                    
                    # Hiển thị thẻ sản phẩm (5 cột một hàng)
                    cols = st.columns(5)
                    for idx, (_, rec) in enumerate(recommendations.iterrows()):
                        with cols[idx % 5]:
                            # Hình ảnh
                            if pd.notna(rec['cover_link']) and rec['cover_link'] != '':
                                st.image(rec['cover_link'], use_container_width=True)
                            else:
                                st.image(f"https://picsum.photos/200/300?random={idx}", use_container_width=True)
                            
                            # Thông tin
                            st.write(f"**{rec['title'][:20]}...**")
                            st.write(f"*{rec['category'][:15]}*")
                            
                            # Điểm số
                            score = rec.get('similarity_score', 0)
                            st.metric("Độ tương đồng", f"{score:.2%}")
                    
                    # ========== PHẦN THƯỜNG ĐƯỢC MUA CÙNG NHAU ==========
                    st.markdown("---")
                    st.subheader("🛒 Khách hàng mua sách này cũng thường mua:")
                    
                    with st.spinner("⏳ Đang tìm sách thường được mua cùng nhau..."):
                        try:
                            frequently_bought = recommender.get_frequently_bought_together(
                                selected_book_id,
                                top_n=5
                            )
                            
                            if frequently_bought.empty:
                                st.info("ℹ️ Chưa đủ dữ liệu để xác định sách thường được mua cùng với cuốn sách này. "
                                       "Hãy thử chọn cuốn sách phổ biến hơn!")
                            else:
                                st.success(f"✅ Tìm thấy {len(frequently_bought)} cuốn sách thường được mua cùng!")
                                
                                # Hiển thị thẻ sản phẩm (5 cột một hàng)
                                cols = st.columns(5)
                                for idx, (_, item) in enumerate(frequently_bought.iterrows()):
                                    with cols[idx % 5]:
                                        # Hình ảnh
                                        if pd.notna(item['cover_link']) and item['cover_link'] != '':
                                            st.image(item['cover_link'], use_container_width=True)
                                        else:
                                            st.image(f"https://picsum.photos/200/300?random={idx+100}", use_container_width=True)
                                        
                                        # Thông tin
                                        st.write(f"**{item['title'][:20]}...**")
                                        st.write(f"*{item['category'][:15]}*")
                        
                        except Exception as e:
                            st.info(f"ℹ️ Không thể tải dữ liệu 'Thường được mua cùng': {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Chi tiết lỗi: {repr(e)}")


# ============ KỊCH BẢN B: KHÁCH HÀNG CŨ (WARM-START) ============
else:
    st.header("👥 Kịch bản: Khách hàng cũ (Warm-Start)")
    
    # Thông báo
    st.info(
        f"✅ **Hệ thống nhận diện:** Khách hàng {selected_customer} có lịch sử đánh giá\n\n"
        "→ **Kết hợp 2 mô hình đánh giá:**\n"
        "  - Collaborative Filtering (SVD): 60% - Dự đoán rating dựa trên khách hàng tương tự\n"
        "  - Content-Based Filtering: 40% - Dựa trên sách khách đã yêu thích (rating ≥ 4)\n\n"
        "→ Chỉ gợi ý sách **chưa từng xem**"
    )
    
    # Nút lấy gợi ý
    if st.button("🔍 Lấy gợi ý sách dành riêng cho bạn", key="btn_warm_start"):
        # Lấy thông tin khách hàng
        customer_reviews = reviews_data[reviews_data['customer_id'] == selected_customer]
        rated_books = customer_reviews['product_id'].tolist()
        avg_rating = customer_reviews['rating'].mean()
        
        # Hiển thị thông tin khách hàng
        st.subheader(f"👤 Thông tin Khách hàng {selected_customer}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Sách đã đánh giá", len(rated_books))
        with col2:
            st.metric("⭐ Mức đánh giá trung bình", f"{avg_rating:.2f}")
        with col3:
            st.metric("🎯 Sở thích", "Sách yêu thích" if avg_rating >= 4 else "Khác biệt")
        
        st.markdown("---")
        
        with st.spinner("⏳ Đang tính toán gợi ý Hybrid (SVD + Content-Based)..."):
            try:
                recommendations = recommender.get_hybrid_recommendations(
                    selected_customer,
                    product_id_viewed=None,
                    top_n=10,
                    content_weight=0.4,
                    collab_weight=0.6
                )
                
                if recommendations.empty:
                    st.warning("⚠️ Không có gợi ý - Khách hàng có thể đã đánh giá tất cả sách")
                else:
                    st.success(f"✅ Tìm thấy {len(recommendations)} cuốn sách phù hợp!")
                    
                    st.subheader("📚 Sách được gợi ý (Hybrid Model)")
                    
                    # Hiển thị thẻ sản phẩm (5 cột một hàng)
                    cols = st.columns(5)
                    for idx, (_, rec) in enumerate(recommendations.iterrows()):
                        with cols[idx % 5]:
                            # Hình ảnh
                            if pd.notna(rec['cover_link']) and rec['cover_link'] != '':
                                st.image(rec['cover_link'], use_container_width=True)
                            else:
                                st.image(f"https://picsum.photos/200/300?random={idx}", use_container_width=True)
                            
                            # Thông tin
                            st.write(f"**{rec['title'][:20]}...**")
                            st.write(f"*{rec['category'][:15]}*")
                            
                            # Điểm số
                            hybrid_score = rec.get('hybrid_score', 0)
                            st.metric("Điểm Hybrid", f"{hybrid_score:.2f}")
                            
                            # Breakdown các score (nếu có)
                            svd_score = rec.get('svd_score', 0)
                            content_score = rec.get('content_score', 0)
                            with st.expander("📊 Chi tiết"):
                                st.write(f"- **Collaborative (SVD):** {svd_score:.2f}")
                                st.write(f"- **Content-Based:** {content_score:.2f}")
            
            except Exception as e:
                st.error(f"❌ Chi tiết lỗi: {repr(e)}")
    
    # Hiển thị lịch sử đánh giá
    with st.expander("📋 Lịch sử đánh giá của bạn"):
        customer_reviews = reviews_data[reviews_data['customer_id'] == selected_customer]
        customer_reviews_display = customer_reviews.copy()
        customer_reviews_display['product_title'] = customer_reviews_display['product_id'].map(book_dict)
        customer_reviews_display = customer_reviews_display[['product_id', 'product_title', 'rating']]
        customer_reviews_display.columns = ['ID Sách', 'Tên Sách', 'Đánh giá']
        st.dataframe(customer_reviews_display, use_container_width=True)


# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p><small>💡 Hybrid Recommendation System | Cold-Start Problem Solution</small></p>
    <p><small>Powered by: Scikit-Learn (TF-IDF) + Surprise (SVD) + Streamlit</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
