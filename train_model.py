"""
Module: train_model.py
Description: End-to-end Machine Learning pipeline for Video Game Sales prediction
Author: Senior Machine Learning Engineer
Dataset: vgsales_clean.csv (or vgsales.csv)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor

# Thiết lập seed cố định cho tính tái lập (Reproducibility)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_and_preprocess_data(file_path="vgsales_clean.csv"):
    """
    Tải và tiền xử lý sơ bộ dữ liệu:
    - Làm sạch missing values (nếu có)
    - Xử lý Feature Engineering cơ bản:
        + Name_Length: Độ dài tên game (các game nổi tiếng thường có tên ngắn hoặc phụ đề đặc trưng)
        + Is_Sequel: Kiểm tra game có phải phần tiếp theo (chứa sốLa Mã hoặc chữ số 2, 3, 4...)
        + Top Publishers: Gom các Publisher nhỏ (< 15 game) thành nhóm 'Other' để chống overfitting và giảm chiều dữ liệu.
    """
    print("=" * 70)
    print("BƯỚC 1: TẢI VÀ KHÁM PHÁ DỮ LIỆU (DATA LOADING & PREPROCESSING)")
    print("=" * 70)
    
    if not os.path.exists(file_path):
        # Fallback tìm trong thư mục hiện tại hoặc thư mục con
        candidates = ["vgsales_clean.csv", "vgsales.csv", "module4-day1-data/vgsales_clean.csv", "../vgsales_clean.csv"]
        for c in candidates:
            if os.path.exists(c):
                file_path = c
                break

    print(f"[*] Đang đọc dữ liệu từ: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[*] Kích thước dữ liệu gốc: {df.shape[0]:,} dòng, {df.shape[1]} cột")

    # Loại bỏ missing values nếu còn sót lại
    df = df.dropna(subset=["Year", "Platform", "Genre", "Publisher", "Global_Sales"]).copy()
    df["Year"] = df["Year"].astype(int)

    # Lọc các bản ghi hợp lý (ví dụ: Year <= 2020)
    df = df[df["Year"] <= 2020].copy()

    # Feature Engineering
    # 1. Chiều dài tên game
    df["Name_Length"] = df["Name"].apply(lambda x: len(str(x)))
    
    # 2. Kiểm tra phần tiếp theo (Sequel indicator)
    sequel_pattern = r"(?:\b2\b|\b3\b|\b4\b|\b5\b|\b6\b|\b7\b|\b8\b|\b9\b|\bII\b|\bIII\b|\bIV\b|\bV\b|\bVI\b|\bVII\b|\bVIII\b|\bIX\b|\bX\b)"
    df["Is_Sequel"] = df["Name"].str.contains(sequel_pattern, regex=True, case=False).astype(int)

    # 3. Giảm cardinality của Publisher: Gom các nhà phát hành có ít hơn 15 game thành 'Other'
    top_publishers = df["Publisher"].value_counts()[lambda x: x >= 15].index
    df["Publisher_Cleaned"] = df["Publisher"].apply(lambda p: p if p in top_publishers else "Other")
    
    print(f"[*] Số lượng Publisher sau khi gộp nhóm nhỏ: {df['Publisher_Cleaned'].nunique()} (thay vì {df['Publisher'].nunique()})")
    print(f"[*] Số lượng Platform: {df['Platform'].nunique()}")
    print(f"[*] Số lượng Genre: {df['Genre'].nunique()}")
    print(f"[*] Doanh số toàn cầu trung bình: {df['Global_Sales'].mean():.2f}M | Trung vị: {df['Global_Sales'].median():.2f}M | Lớn nhất: {df['Global_Sales'].max():.2f}M")
    
    return df


def build_pipeline():
    """
    Xây dựng Pipeline xử lý chuẩn của Scikit-Learn:
    - Biến Categorical (Platform, Genre, Publisher_Cleaned): OneHotEncoder(handle_unknown='ignore')
    - Biến Numerical (Year, Name_Length, Is_Sequel): StandardScaler()
    -> Đảm bảo không xảy ra Data Leakage khi huấn luyện và kiểm thử.
    """
    cat_features = ["Platform", "Genre", "Publisher_Cleaned"]
    num_features = ["Year", "Name_Length", "Is_Sequel"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
    )

    return preprocessor, cat_features, num_features


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Huấn luyện và so sánh đa mô hình (Benchmark):
    Vì phân phối Global_Sales bị lệch phải nghiêm trọng (Heavy Right Skew), 
    chúng ta sử dụng log1p (log(1 + y)) khi fit mô hình và expm1 khi dự đoán.
    Điều này giúp mô hình không bị chi phối bởi các outlier siêu phẩm và hội tụ tốt hơn.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 2: HUẤN LUYỆN VÀ SO SÁNH CÁC MÔ HÌNH (MODEL BENCHMARKING)")
    print("=" * 70)

    # Chuyển đổi target sang miền Log (Log Transformation)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Danh sách các mô hình ứng viên
    models = {
        "Baseline (Dummy Mean)": DummyRegressor(strategy="mean"),
        "Ridge Regression (L2 Linear)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(max_iter=150, max_depth=8, random_state=RANDOM_STATE),
    }

    results = []
    trained_pipelines = {}

    for name, model in models.items():
        print(f"\n[*] Đang huấn luyện: {name}...")
        
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        # Huấn luyện trên không gian log-transformed
        pipeline.fit(X_train, y_train_log)
        trained_pipelines[name] = pipeline

        # Dự đoán trên tập Train và Test (chuyển ngược lại bằng expm1)
        y_train_pred_log = pipeline.predict(X_train)
        y_train_pred = np.expm1(y_train_pred_log)
        y_train_pred = np.clip(y_train_pred, a_min=0, a_max=None)  # Doanh số không âm

        y_test_pred_log = pipeline.predict(X_test)
        y_test_pred = np.expm1(y_test_pred_log)
        y_test_pred = np.clip(y_test_pred, a_min=0, a_max=None)

        # Tính toán các chỉ số đánh giá (Evaluation Metrics)
        # 1. R2 Score (trên miền log để đánh giá độ khớp xu hướng tăng trưởng)
        r2_log = r2_score(y_test_log, y_test_pred_log)
        # 2. MAE & RMSE (trên miền doanh số thực tế triệu bản)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        # 3. RMSLE (Root Mean Squared Logarithmic Error)
        rmsle = np.sqrt(mean_squared_error(y_test_log, y_test_pred_log))

        results.append({
            "Model": name,
            "R2 (Log Scale)": r2_log,
            "RMSLE": rmsle,
            "MAE (Triệu bản)": mae,
            "RMSE (Triệu bản)": rmse,
        })
        
        print(f"    - R2 (Log Scale): {r2_log:.4f}")
        print(f"    - RMSLE:          {rmsle:.4f}")
        print(f"    - MAE:            {mae:.4f}M đơn vị")
        print(f"    - RMSE:           {rmse:.4f}M đơn vị")

    results_df = pd.DataFrame(results).sort_values("RMSLE")
    
    print("\n" + "=" * 70)
    print("BẢNG TỔNG HỢP SO SÁNH HIỆU SUẤT MÔ HÌNH (MODEL LEADERBOARD)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Chọn mô hình tốt nhất dựa trên RMSLE thấp nhất
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\n[✓] Mô hình chiến thắng: '{best_model_name}'")

    return best_model_name, best_pipeline, results_df, trained_pipelines


def analyze_feature_importance(best_pipeline, preprocessor, cat_features, num_features, output_img="feature_importance.png"):
    """
    Trích xuất và trực quan hóa Feature Importance để giải thích mô hình (Explainable AI / Interpretability).
    """
    print("\n" + "=" * 70)
    print("BƯỚC 3: PHÂN TÍCH TẦM QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG (FEATURE IMPORTANCE)")
    print("=" * 70)

    regressor = best_pipeline.named_steps["regressor"]
    
    # Lấy tên các feature sau One-Hot Encoding
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_features))
    all_feature_names = cat_feature_names + num_features

    # Lấy importance tùy loại mô hình
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    else:
        print("[!] Mô hình được chọn không có feature_importances_ trực tiếp.")
        return

    feat_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    print("[*] Top 15 đặc trưng có sức ảnh hưởng lớn nhất đến doanh số game:")
    print(feat_df.head(15).to_string(index=False))

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df.head(15), x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
    plt.title("Top 15 Most Important Features in Game Sales Prediction", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu biểu đồ tầm quan trọng đặc trưng tại: {output_img}")


def save_artifacts(best_pipeline, model_filename="best_game_sales_model.joblib"):
    """
    Lưu mô hình đã huấn luyện hoàn chỉnh để phục vụ suy luận (Inference / Production Deployment).
    """
    joblib.dump(best_pipeline, model_filename)
    print(f"\n[✓] Đã xuất gói mô hình sẵn sàng Production: '{model_filename}' ({os.path.getsize(model_filename) / 1024:.1f} KB)")


def predict_new_game(model_pipeline, game_dict):
    """
    Hàm suy luận (Inference helper) cho game mới:
    Input: dictionary chứa {Name, Platform, Genre, Publisher, Year}
    Output: Doanh số dự đoán (triệu bản)
    """
    df_single = pd.DataFrame([game_dict])
    
    # Feature engineering tương ứng
    df_single["Name_Length"] = len(str(game_dict.get("Name", "")))
    sequel_pattern = r"(?:\b2\b|\b3\b|\b4\b|\b5\b|\b6\b|\b7\b|\b8\b|\b9\b|\bII\b|\bIII\b|\bIV\b|\bV\b|\bVI\b|\bVII\b|\bVIII\b|\bIX\b|\bX\b)"
    df_single["Is_Sequel"] = int(bool(pd.Series([game_dict.get("Name", "")]).str.contains(sequel_pattern, regex=True).iloc[0]))
    df_single["Publisher_Cleaned"] = game_dict.get("Publisher", "Other")

    # Dự đoán (nhớ chuyển ngược từ log1p bằng expm1)
    pred_log = model_pipeline.predict(df_single)
    pred_sales = float(np.expm1(pred_log)[0])
    return max(0.0, pred_sales)


def run_demo_predictions(best_pipeline):
    """
    Chạy thử nghiệm dự đoán thực tế cho một số tựa game giả định.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 4: THỬ NGHIỆM SUY LUẬN THỰC TẾ (INFERENCE SIMULATION)")
    print("=" * 70)

    sample_games = [
        {
            "Name": "The Legend of Zelda: Tears of the Kingdom 2",
            "Platform": "Wii",
            "Genre": "Action",
            "Publisher": "Nintendo",
            "Year": 2016
        },
        {
            "Name": "Call of Duty: Modern Warfare 4",
            "Platform": "PS4",
            "Genre": "Shooter",
            "Publisher": "Activision",
            "Year": 2015
        },
        {
            "Name": "Indie Mystery Puzzle Adventure",
            "Platform": "PC",
            "Genre": "Puzzle",
            "Publisher": "Unknown Indie Dev",
            "Year": 2014
        },
        {
            "Name": "FIFA 17 Championship",
            "Platform": "PS4",
            "Genre": "Sports",
            "Publisher": "Electronic Arts",
            "Year": 2016
        }
    ]

    for game in sample_games:
        predicted_sales = predict_new_game(best_pipeline, game)
        print(f"[*] Game: {game['Name']:<40} | Hệ: {game['Platform']:<4} | Thể loại: {game['Genre']:<10} | Hãng: {game['Publisher']:<15} -> Doanh số dự đoán: {predicted_sales:.2f} triệu bản")


def main():
    # 1. Tải và tiền xử lý
    data_file = "vgsales_clean.csv" if os.path.exists("vgsales_clean.csv") else "vgsales.csv"
    df = load_and_preprocess_data(data_file)

    # 2. Định nghĩa Features (X) và Target (y)
    feature_cols = ["Platform", "Genre", "Publisher_Cleaned", "Year", "Name_Length", "Is_Sequel"]
    X = df[feature_cols]
    y = df["Global_Sales"]

    # 3. Phân chia Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"[*] Phân chia tập dữ liệu: Train = {len(X_train):,} mẫu | Test = {len(X_test):,} mẫu")

    # 4. Tạo Preprocessor
    preprocessor, cat_features, num_features = build_pipeline()

    # 5. Huấn luyện và Đánh giá
    best_name, best_pipeline, results_df, all_pipelines = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # 6. Phân tích Feature Importance
    # Ưu tiên lấy Random Forest hoặc HistGradientBoosting để trích xuất importance
    rf_pipeline = all_pipelines.get("Random Forest Regressor", best_pipeline)
    analyze_feature_importance(rf_pipeline, preprocessor, cat_features, num_features)

    # 7. Lưu mô hình (Joblib)
    save_artifacts(best_pipeline)

    # 8. Chạy Demo suy luận
    run_demo_predictions(best_pipeline)

    print("\n" + "=" * 70)
    print("HOÀN TẤT QUY TRÌNH MACHINE LEARNING PIPELINE THÀNH CÔNG!")
    print("=" * 70)


if __name__ == "__main__":
    main()
