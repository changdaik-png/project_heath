import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# 1. 전처리 모듈로부터 데이터 로드
from preprocessing import preprocess_khp_data

def train_and_evaluate_models():
    print("==================================================")
    print("[1] 한국의료패널(KHP) 데이터셋 로드 및 전처리 시작...")
    df = preprocess_khp_data()
    print(f"로드 완료! 데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
    
    # [오류 해결] 종속변수 y (delta_Y)가 결측치(NaN)인 행을 학습 데이터셋에서 미리 드롭합니다.
    df_cleaned = df.dropna(subset=['delta_Y']).copy()
    print(f" - 종속변수(delta_Y) 결측치 행 제거 완료: {df_cleaned.shape[0]}행 남음 (이전: {df.shape[0]}행)")
    
    # 2. 독립변수(X) 및 종속변수(y) 정의
    # 독립변수: 2017년 기준의 나이(A5), 소득(TOT_INC), 만성질환 여부(CHRONIC_YN), 성별(A2)
    # 종속변수: 2022년 의료비 - 2017년 의료비 (delta_Y)
    feature_mapping = {
        'A5': 'Age',
        'TOT_INC': 'Income',
        'CHRONIC_YN': 'Chronic_Disease',
        'A2': 'Gender'
    }
    
    # 해당 컬럼들이 데이터프레임에 존재하는지 확인하여 안전하게 데이터 추출
    available_features = [col for col in feature_mapping.keys() if col in df_cleaned.columns]
    
    if len(available_features) < len(feature_mapping):
        print("[경고] 일부 지정된 컬럼을 찾을 수 없어, 존재하는 컬럼만 사용합니다.")
        
    X = df_cleaned[available_features].copy()
    # 친절한 이름으로 컬럼명 변경
    X = X.rename(columns={col: feature_mapping[col] for col in available_features})
    
    y = df_cleaned['delta_Y']
    
    print("\n--- 학습에 사용되는 피처 요약 ---")
    print(X.info())
    print("\n종속변수(delta_Y) 요약:")
    print(y.describe())
    
    # 3. 학습용(70%) / 테스트용(30%) 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"\n[2] 데이터 분할 완료: Train={X_train.shape[0]}건, Test={X_test.shape[0]}건")
    
    # 4. 결측치 처리를 포함한 scikit-learn Pipeline 구축
    # 결측치는 평균(mean)으로 보완합니다.
    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', LinearRegression())
    ])
    
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', RandomForestRegressor(
            n_estimators=200, 
            max_depth=4, 
            min_samples_leaf=50, 
            random_state=42, 
            n_jobs=-1
        ))
    ])
    
    # 5. 모델 학습
    print("\n[3] 모형 학습 진행 중...")
    lr_pipeline.fit(X_train, y_train)
    print(" - 다중 선형 회귀(Linear Regression) 학습 완료.")
    rf_pipeline.fit(X_train, y_train)
    print(" - 랜덤 포레스트 회귀(Random Forest Regressor) 학습 완료.")
    
    # 6. 예측 및 성능 평가
    y_pred_lr = lr_pipeline.predict(X_test)
    y_pred_rf = rf_pipeline.predict(X_test)
    
    # 평가지표 산출 (RMSE, MAE)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    
    print("\n================== [4] 모델 성능 평가 결과 ==================")
    print(f"■ 다중 선형 회귀 (Linear Regression):")
    print(f"  - RMSE: {rmse_lr:,.2f} 원")
    print(f"  - MAE : {mae_lr:,.2f} 원")
    print(f"■ 랜덤 포레스트 회귀 (Random Forest Regressor):")
    print(f"  - RMSE: {rmse_rf:,.2f} 원")
    print(f"  - MAE : {mae_rf:,.2f} 원")
    print("=============================================================")
    
    # 7. 랜덤 포레스트 변수 중요도(Feature Importance) 출력
    rf_model = rf_pipeline.named_steps['regressor']
    importances = rf_model.feature_importances_
    feature_names = X.columns
    
    # 중요도 역정렬
    indices = np.argsort(importances)[::-1]
    
    print("\n■ 랜덤 포레스트 변수 중요도 (Feature Importance) 상위 3개:")
    for idx in range(min(3, len(indices))):
        name = feature_names[indices[idx]]
        imp = importances[indices[idx]]
        print(f"  {idx+1}위. {name:<16} : {imp*100:.2f}%")
    
    # 8. 최종 랜덤 포레스트 파이프라인 저장
    model_filename = 'model.pkl'
    joblib.dump(rf_pipeline, model_filename)
    print(f"\n[5] 학습 완료된 랜덤 포레스트 파이프라인을 '{model_filename}' 파일로 저장하였습니다!")
    print("==================================================")

if __name__ == "__main__":
    train_and_evaluate_models()
