import os
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Cấu hình trang
st.set_page_config(
    page_title="Dashboard Phân Tích Cảm Xúc Facebook",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    if os.path.exists("Facebook_predictions.csv"):
        df = pd.read_csv("Facebook_predictions.csv")
    elif os.path.exists("Facebook_data_txt.csv"):
        df = pd.read_csv("Facebook_data_txt.csv")
        # Làm sạch & dự đoán bằng Naive Bayes nếu chưa có cột predicted_sentiment
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r"http\S+|www\S+|https\S+", "", text)
            text = re.sub(r"@\w+|#\w+", "", text)
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r"\d+", "", text)
            return re.sub(r"\s+", " ", text).strip()

        clean = df["text"].apply(clean_text)
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        X = tfidf.fit_transform(clean)
        model = MultinomialNB(alpha=1.0).fit(X, df["sentiment"])
        df["predicted_sentiment"] = model.predict(X)
    else:
        st.error("Không tìm thấy file Facebook_predictions.csv hoặc Facebook_data_txt.csv trong thư mục!")
        st.stop()
    return df

df = load_data()

# Tiêu đề Dashboard
st.title("📊 Dashboard Phân Tích Cảm Xúc Bài Đăng Facebook")
st.markdown("Trực quan hóa kết quả phân loại cảm xúc (**Positive, Neutral, Negative**) và mức độ tương tác.")

# ==========================================
# YÊU CẦU d): BỘ LỌC DỮ LIỆU (SIDEBAR)
# ==========================================
st.sidebar.header("🔍 Bộ lọc dữ liệu")

categories = sorted(df["category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Chọn danh mục bài đăng (Category):",
    options=categories,
    default=categories
)

sentiments = ["positive", "neutral", "negative"]
selected_sentiments = st.sidebar.multiselect(
    "Chọn nhóm cảm xúc (Sentiment):",
    options=sentiments,
    default=sentiments
)

filtered_df = df[
    (df["category"].isin(selected_categories)) &
    (df["sentiment"].isin(selected_sentiments))
].copy()

if filtered_df.empty:
    st.warning("Không có bài đăng nào khớp với bộ lọc hiện tại. Vui lòng chọn lại!")
    st.stop()

# ==========================================
# YÊU CẦU a): THỐNG KÊ TỔNG SỐ & TỶ LỆ CẢM XÚC
# ==========================================
st.subheader("1. Thống kê tổng quan")
total_posts = len(filtered_df)
pos_count = (filtered_df["sentiment"] == "positive").sum()
neu_count = (filtered_df["sentiment"] == "neutral").sum()
neg_count = (filtered_df["sentiment"] == "negative").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tổng số bài đăng", f"{total_posts:,}")
col2.metric("Positive (Tích cực)", f"{pos_count:,}", f"{pos_count / total_posts * 100:.2f}%")
col3.metric("Neutral (Trung tính)", f"{neu_count:,}", f"{neu_count / total_posts * 100:.2f}%")
col4.metric("Negative (Tiêu cực)", f"{neg_count:,}", f"{neg_count / total_posts * 100:.2f}%")

st.divider()

# ==========================================
# YÊU CẦU b) & c): BIỂU ĐỒ PHÂN BỐ & TƯƠNG TÁC
# ==========================================
color_map = {"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"}
chart_col1, chart_col2 = st.columns(2)

# Yêu cầu b): Phân bố cảm xúc
with chart_col1:
    st.subheader("2. Phân bố nhóm cảm xúc")
    sent_counts = filtered_df["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Cảm xúc", "Số lượng"]

    fig_pie = px.pie(
        sent_counts,
        names="Cảm xúc",
        values="Số lượng",
        color="Cảm xúc",
        color_discrete_map=color_map,
        hole=0.4,
        title="Tỷ lệ các nhóm cảm xúc"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Yêu cầu c): Likes, Comments, Shares theo nhóm cảm xúc
with chart_col2:
    st.subheader("3. Tương tác trung bình theo cảm xúc")
    eng_mean = (
        filtered_df.groupby("sentiment")[["likes", "comments", "shares"]]
        .mean()
        .round(2)
        .reset_index()
    )
    eng_melt = eng_mean.melt(id_vars="sentiment", var_name="Loại tương tác", value_name="Trung bình")

    fig_bar = px.bar(
        eng_melt,
        x="sentiment",
        y="Trung bình",
        color="Loại tương tác",
        barmode="group",
        text_auto=True,
        title="Số lượt Likes, Comments, Shares trung bình theo cảm xúc",
        labels={"sentiment": "Nhóm cảm xúc"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Bảng thống kê tổng lượt tương tác (bổ sung chi tiết cho yêu cầu c)
with st.expander("📋 Xem bảng thống kê tổng và trung bình Likes, Comments, Shares"):
    eng_summary = filtered_df.groupby("sentiment")[["likes", "comments", "shares"]].agg(["sum", "mean"]).round(2)
    st.dataframe(eng_summary, use_container_width=True)

st.divider()

# ==========================================
# YÊU CẦU e): DANH SÁCH BÀI ĐĂNG & KẾT QUẢ DỰ ĐOÁN
# ==========================================
st.subheader("4. Danh sách bài đăng và kết quả dự đoán cảm xúc")
display_cols = ["post_id", "timestamp", "category", "text", "sentiment", "predicted_sentiment", "likes", "comments", "shares"]
available_cols = [c for c in display_cols if c in filtered_df.columns]

st.dataframe(
    filtered_df[available_cols].head(200),
    use_container_width=True,
    height=400
)
st.caption(f"Đang hiển thị tối đa 200 / {total_posts:,} bài đăng khớp bộ lọc.")