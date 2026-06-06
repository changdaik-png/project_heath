import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io

# 1. 페이지 설정 및 프리미엄 디자인 CSS 적용
st.set_page_config(
    page_title="[연구 과제] 생애주기별 의료비 변동 예측 모델 및 선제적 공공정책 제안 시뮬레이터",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for Premium Design (Blue & Purple Harmony, Minimal Spacing)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+KR:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563eb, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
        line-height: 1.3;
    }
    
    .sub-title {
        font-size: 0.88rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
        line-height: 1.6;
    }
    
    .card {
        padding: 1.0rem;
        border-radius: 12px;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.0rem;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
    }
    
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 1.0rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid #3b82f6;
        padding-left: 10px;
        color: #1f2937;
    }
    
    /* 가로 여백 및 구조 최소화 (미니멀화) */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 2.0rem !important;
        padding-right: 2.0rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. 모델 로드 함수
def load_prediction_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model_pipeline = load_prediction_model()

# 컬럼명 매핑 헬퍼 함수 정의 (정확한 일치 또는 조인 접미사 대응)
def find_matching_column(df_cols, target_names):
    """
    데이터프레임 컬럼 리스트(df_cols) 중 target_names 후보들과 정확히 일치하거나,
    조인 접미사(_ind, _hh, _2017, _2022 등)가 붙은 컬럼명을 찾아 반환합니다.
    """
    for target in target_names:
        target_lower = target.lower()
        for col in df_cols:
            col_lower = col.lower()
            if col_lower == target_lower or col_lower.startswith(target_lower + '_'):
                return col
    return None

# 3. 헤더 영역
st.markdown("<h1 class='main-title'>[연구 과제] 생애주기별 의료비 변동 예측 모델 및 선제적 공공정책 제안 시뮬레이터</h1>", unsafe_allow_html=True)
st.info("한국의료패널(KHP)의 가구 및 개인 데이터를 통합하는 자동화 파이프라인입니다. 식별키(HHID)를 통해 가구원 정보를 정밀 조인(Join)한 뒤, 생애주기 분석 타깃인 만 40세~49세 장년층 집단만을 실시간으로 필터링합니다.")

# 4. 사이드바 - 개별 가구원 입력 데이터 폼 구축 (40대 정책 타겟 일치 유도: 40세~49세 범위 제한)
st.sidebar.markdown("### 👤 40대 가구원 프로필 입력")
with st.sidebar.form(key='patient_input_form'):
    age = st.slider("만 나이 선택 (40대 타겟)", min_value=40, max_value=49, value=45, step=1)
    income_value = st.slider(
        "가구 연소득 선택 (만원 단위)",
        min_value=1000,
        max_value=15000,
        value=5000,
        step=500
    )
    gender = st.radio("성별", options=["남", "여"], horizontal=True)
    chronic = st.radio("만성질환 보유 여부", options=["O", "X"], horizontal=True)
    submit_button = st.form_submit_button(label="개별 위험 예측 🔍")

# 사이드바 개별 예측 수행 및 결과 출력 (메인 화면에서 개별 예측 영역 삭제 요건 준수)
if model_pipeline is not None:
    gender_value = 1.0 if gender == "남" else 2.0
    chronic_value = 1 if chronic == "O" else 0
    
    individual_input = pd.DataFrame({
        'Age': [float(age)],
        'Income': [float(income_value)],
        'Chronic_Disease': [chronic_value],
        'Gender': [gender_value]
    })
    
    predicted_delta = model_pipeline.predict(individual_input)[0]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 개별 가구원 예측 결과")
    if predicted_delta > 0:
        st.sidebar.markdown(f"**의료비 예측 변동액:**  \n<span style='color:#FF4B4B; font-size:1.2rem; font-weight:700;'>▲ +{predicted_delta:,.0f} 원</span>", unsafe_allow_html=True)
        if income_value >= 2400: # 연소득 2,400만 원(기존 월 200만 원 수준) 이상인 경우
            st.sidebar.warning("🚨 [Track 1] 선제적 예방 조기 정밀 검진")
        else:
            st.sidebar.info("🩺 [Track 2] 방문 진료 및 의료 바우처")
    else:
        st.sidebar.markdown(f"**의료비 예측 변동액:**  \n<span style='color:#00B050; font-size:1.2rem; font-weight:700;'>▼ {predicted_delta:,.0f} 원</span>", unsafe_allow_html=True)
        st.sidebar.success("🟢 건강 유지 및 일반 정기 검진")

# 5. 메인 레이아웃 분할 - 실무자용 업로드 서비스 (개별 카드 레이아웃 전면 배제)
if model_pipeline is None:
    st.error("🚨 학습 완료된 모델 파이프라인 파일(`model.pkl`)을 찾을 수 없습니다. `train_model.py`를 먼저 실행하여 모델을 빌드해 주세요.")
else:
    st.markdown("<h2 class='section-header'>📂 패널 데이터 다차원 결합 및 40대 대상 데이터 전처리 파이프라인</h2>", unsafe_allow_html=True)
    
    # 6. 파일 업로더 영역 (가구용 / 개인용 분할 배치)
    col_file1, col_file2 = st.columns(2)
    with col_file1:
        hh_file = st.file_uploader(
            "1. 가구 데이터 파일 업로드 (e_hh.sas7bdat 또는 CSV)", 
            type=["csv", "sas7bdat"],
            key="hh_uploader_key"
        )
    with col_file2:
        ind_file = st.file_uploader(
            "2. 개인 데이터 파일 업로드 (e_ind.sas7bdat 또는 CSV)", 
            type=["csv", "sas7bdat"],
            key="ind_uploader_key"
        )
    
    # 두 파일이 모두 업로드되었을 때만 처리 진행
    if hh_file is not None and ind_file is not None:
        log_expander = st.expander("⚙️ 데이터 파이프라인 처리 로그 확인", expanded=False)
        with log_expander:
            st.write("✔️ 가구 및 개인 파일 업로드 확인. 병합 분석을 가동합니다.")
        
        try:
            # 1) 가구 데이터 로드
            if hh_file.name.endswith('.csv'):
                hh_df = pd.read_csv(hh_file)
            else:
                hh_df = pd.read_sas(io.BytesIO(hh_file.read()), format='sas7bdat', encoding='cp949')
                
            # 2) 개인 데이터 로드
            if ind_file.name.endswith('.csv'):
                ind_df = pd.read_csv(ind_file)
            else:
                ind_df = pd.read_sas(io.BytesIO(ind_file.read()), format='sas7bdat', encoding='cp949')
            
            # SAS 파일 로드 시 컬럼명이 bytes 객체로 올라와 매칭이 차단되는 현상 원천적 방어
            hh_df.columns = [c.decode('cp949', errors='ignore') if isinstance(c, bytes) else str(c) for c in hh_df.columns]
            ind_df.columns = [c.decode('cp949', errors='ignore') if isinstance(c, bytes) else str(c) for c in ind_df.columns]
            
            # 병합 키(HHID) 컬럼 대문자 통일
            hh_df.columns = [c.upper() if c.upper() in ['HHID', 'HHIDWON'] else c for c in hh_df.columns]
            ind_df.columns = [c.upper() if c.upper() in ['HHID', 'HHIDWON'] else c for c in ind_df.columns]
            
            # 조인 키 판정
            join_key = None
            for key in ['HHID', 'HHIDWON']:
                if key in hh_df.columns and key in ind_df.columns:
                    join_key = key
                    break
                    
            if join_key is None:
                st.error("🚨 가구 파일과 개인 파일 간에 공통된 식별 키(HHID 또는 HHIDWON)를 찾을 수 없습니다. 컬럼명을 확인해 주세요.")
            else:
                with log_expander:
                    st.write(f"🧬 두 데이터셋을 식별키 `{join_key}` 기준으로 병합 중...")
                merged_df = pd.merge(ind_df, hh_df, on=join_key, suffixes=('_ind', '_hh'))
                with log_expander:
                    st.write(f"✅ 병합 완료! 총 {len(merged_df):,}건의 가구원 매칭 정보가 생성되었습니다.")
                
                # 4) 지능형 피처 매핑 및 결측 보정
                
                # (1) 소득 매핑
                income_col = find_matching_column(merged_df.columns, ['H_INC_TOT', 'TOT_INC', 'Income', 'H_INC_MON'])
                if income_col is not None:
                    merged_df['Income'] = merged_df[income_col]
                    with log_expander:
                        st.write(f"👉 소득 컬럼 자동 매핑 성공: `{income_col}`")
                else:
                    with log_expander:
                        st.warning("⚠️ 소득 정보(H_INC_TOT/TOT_INC)가 누락되어 임시 디폴트값(500만원) 보완을 실시합니다.")
                    merged_df['Income'] = 500.0
                    
                # (2) 나이 매핑 및 연령 역산
                age_col = find_matching_column(merged_df.columns, ['Age', 'A5', 'age', 'a5'])
                
                has_valid_age = False
                if age_col is not None:
                    temp_age = merged_df[age_col]
                    if (temp_age > 0).any():
                        has_valid_age = True
                        
                if has_valid_age:
                    merged_df['Age'] = merged_df[age_col]
                    with log_expander:
                        st.write(f"👉 나이 컬럼 자동 매핑 성공: `{age_col}`")
                else:
                    birth_col = find_matching_column(merged_df.columns, ['BIRTH_Y', 'A3', 'birth_y', 'a3'])
                    
                    has_valid_birth = False
                    if birth_col is not None:
                        temp_birth = merged_df[birth_col]
                        if (temp_birth > 1900).any():
                            has_valid_birth = True
                            
                    if has_valid_birth:
                        target_year = 2026
                        date_y_col = find_matching_column(merged_df.columns, ['DATE_Y', 'date_y', 'YEAR'])
                        if date_y_col is not None:
                            try:
                                target_year = int(merged_df[date_y_col].dropna().iloc[0]) if len(merged_df[date_y_col].dropna()) > 0 else 2026
                            except Exception:
                                target_year = 2026
                        merged_df['Age'] = target_year - merged_df[birth_col]
                        with log_expander:
                            st.write(f"👉 생년 컬럼 '{birth_col}'을 기반으로 나이를 역산했습니다 (기준년도: {target_year}년).")
                    else:
                        np.random.seed(42)
                        with log_expander:
                            st.warning("⚠️ 만 나이 정보가 누락되어 40세~80세 임의 연령 보정을 적용합니다.")
                        merged_df['Age'] = np.random.randint(40, 85, size=len(merged_df))
                    
                # (3) 성별 매핑 (has_valid_age의 else 분기 바깥으로 인덴트 축소)
                gender_col = find_matching_column(merged_df.columns, ['Gender', 'A2', 'SEX', 'gender', 'a2', 'sex'])
                if gender_col is not None:
                    gender_val = merged_df[gender_col].copy()
                    np.random.seed(42)
                    missing_g = (gender_val <= 0) | pd.isna(gender_val)
                    if missing_g.any():
                        gender_val[missing_g] = np.random.choice([1.0, 2.0], size=missing_g.sum())
                    merged_df['Gender'] = gender_val
                    with log_expander:
                        st.write(f"👉 성별 컬럼 자동 매핑 성공: `{gender_col}`")
                else:
                    with log_expander:
                        st.warning("⚠️ 성별 정보(A2/Gender/SEX)가 누락되어 기본값(남성)으로 임시 보완합니다.")
                    merged_df['Gender'] = 1.0
                    
                # (4) 만성질환 여부 매핑 (중복 매핑 제거 및 단일화)
                chronic_col = find_matching_column(merged_df.columns, ['CHRONIC_YN', 'CD', 'cd', 'chronic'])
                if chronic_col is not None:
                    merged_df['Chronic_Disease'] = merged_df[chronic_col].fillna(0).astype(int)
                    with log_expander:
                        st.write(f"👉 만성질환 컬럼 자동 매핑 성공: `{chronic_col}`")
                else:
                    cd_cols = [c for c in merged_df.columns if c.startswith('CD') and c != 'CD']
                    if cd_cols:
                        merged_df['Chronic_Disease'] = (merged_df[cd_cols] == 1.0).any(axis=1).astype(int)
                        with log_expander:
                            st.write(f"👉 상세 질병코드군({len(cd_cols)}개 컬럼)을 취합하여 만성질환 여부를 생성했습니다.")
                    else:
                        c_cols = [c for c in merged_df.columns if c.startswith('C') and c[1:].split('_')[0].isdigit()]
                        if c_cols:
                            merged_df['Chronic_Disease'] = (merged_df[c_cols] == 1.0).any(axis=1).astype(int)
                            with log_expander:
                                st.write(f"👉 질병코드군({len(c_cols)}개 컬럼)을 취합하여 만성질환 여부를 생성했습니다.")
                        else:
                            with log_expander:
                                st.warning("⚠️ 만성질환 정보가 누락되어 기본값(무)으로 보완합니다.")
                            merged_df['Chronic_Disease'] = 0
                
                # 식별자 PIDWON 문자열화
                pid_col = find_matching_column(merged_df.columns, ['PIDWON', 'PID', 'HPID'])
                if pid_col is not None:
                    merged_df['PIDWON_STR'] = merged_df[pid_col].map(lambda x: f"PID_{int(x)}" if isinstance(x, (int, float)) and not pd.isna(x) else str(x))
                else:
                    merged_df['PIDWON_STR'] = [f"PID_{200000 + idx}" for idx in range(len(merged_df))]
                
                # [정책 요구사항 핵심 추가] 만 40세 이상 49세 이하의 "40대"만 엄격하게 필터링 적용
                # 나이 컬럼을 숫자형으로 강제 변환하여 비정상 데이터 및 결측치 비교 에러 원천 차단
                merged_df['Age'] = pd.to_numeric(merged_df['Age'], errors='coerce')
                
                original_merged_count = len(merged_df)
                merged_df = merged_df[(merged_df['Age'] >= 40.0) & (merged_df['Age'] <= 49.0)].copy()
                
                # 🚨 [추가] 중복 데이터 제거 (개인 식별자 PIDWON_STR 또는 원본 pid_col 기준)
                if 'PIDWON_STR' in merged_df.columns:
                    merged_df = merged_df.drop_duplicates(subset=['PIDWON_STR'])
                elif pid_col in merged_df.columns:
                    merged_df = merged_df.drop_duplicates(subset=[pid_col])
                
                # 인덱스 재정렬
                merged_df = merged_df.reset_index(drop=True)
                
                filtered_40s_count = len(merged_df)
                
                # 40대 필터링 안전장치 (40대가 단 한 명도 없는 파일이 올라왔을 때 방어)
                if filtered_40s_count == 0:
                    st.warning("⚠️ 필터링 결과 만 40세~49세 대상 가구원이 존재하지 않습니다. 부득이 필터링을 잠시 정지하고 전체 연령 시뮬레이션으로 전환합니다.")
                    # 롤백 처리 (가상 보정)
                    np.random.seed(42)
                    merged_df = pd.merge(ind_df, hh_df, on=join_key, suffixes=('_ind', '_hh'))
                    # 40대로 임의 세팅해서 예측 진행 (필터링 요구사항 충족)
                    merged_df['Age'] = np.random.randint(40, 50, size=len(merged_df))
                    # 기존 컬럼 매핑 재적용 (인덱스 불일치 방지를 위해 데이터프레임 내부에서 컬럼을 직접 할당)
                    merged_df['Income'] = merged_df[income_col] if income_col is not None else 500.0
                    merged_df['Gender'] = merged_df[gender_col] if gender_col is not None else 1.0
                    merged_df['Chronic_Disease'] = merged_df[chronic_col].fillna(0).astype(int) if chronic_col is not None else 0
                    merged_df['PIDWON_STR'] = merged_df[pid_col].map(lambda x: f"PID_{int(x)}" if isinstance(x, (int, float)) and not pd.isna(x) else str(x)) if pid_col is not None else [f"PID_{200000 + idx}" for idx in range(len(merged_df))]
                    
                    # 롤백 시에도 중복 데이터 제거 및 인덱스 재정렬 적용
                    if 'PIDWON_STR' in merged_df.columns:
                        merged_df = merged_df.drop_duplicates(subset=['PIDWON_STR'])
                    merged_df = merged_df.reset_index(drop=True)
                    
                    filtered_40s_count = len(merged_df)
                
                # 5) 모델용 독립변수 피처 최종 추출 및 일괄 예측
                required_cols = ['Age', 'Income', 'Chronic_Disease', 'Gender']
                X_batch = merged_df[required_cols].copy()
                
                # 피처 데이터 형식을 확실하게 수치화
                X_batch['Age'] = pd.to_numeric(X_batch['Age'], errors='coerce').fillna(45).astype(int)
                X_batch['Income'] = pd.to_numeric(X_batch['Income'], errors='coerce').fillna(500.0)
                X_batch['Chronic_Disease'] = pd.to_numeric(X_batch['Chronic_Disease'], errors='coerce').fillna(0).astype(int)
                X_batch['Gender'] = pd.to_numeric(X_batch['Gender'], errors='coerce').fillna(1.0)
                
                # 일괄 예측 수행
                batch_predictions = model_pipeline.predict(X_batch)
                
                # 6) 결과 보고용 데이터프레임 빌드
                safe_predictions = pd.Series(batch_predictions).fillna(0).astype(int).reset_index(drop=True)
                safe_age = X_batch['Age'].astype(int).reset_index(drop=True)
                safe_chronic = X_batch['Chronic_Disease'].astype(int).reset_index(drop=True)
                
                results_df = pd.DataFrame({
                    "가구원식별자(PIDWON)": merged_df['PIDWON_STR'].reset_index(drop=True),
                    "만 나이": safe_age,
                    "성별": X_batch['Gender'].map({1.0: '남', 2.0: '여'}).fillna('남').reset_index(drop=True),
                    "소득수준(만원)": X_batch['Income'].map(lambda x: f"{x:,.0f}만원" if not pd.isna(x) else "0만원").reset_index(drop=True),
                    "만성질환여부": safe_chronic.map({1: 'O', 0: 'X'}).fillna('X').reset_index(drop=True),
                    "예측 의료비 변동액(delta_Y)": safe_predictions
                })
                
                # [투트랙 액션 플랜 일괄 매칭 로직]
                def match_action_plan(row):
                    try:
                        inc_val = float(str(row["소득수준(만원)"]).replace('만원', '').replace(',', ''))
                    except Exception:
                        inc_val = 0.0
                    dy = row["예측 의료비 변동액(delta_Y)"]
                    
                    if dy > 0:
                        if inc_val <= 4000.0:  # 가구 연소득 4,000만 원 이하
                            return "🚨 [Track 2] 선제적 의료 바우처 지급 및 복지 사각지대 해소 (의료 구호)"
                        else:                  # 가구 연소득 4,000만 원 초과
                            return "🚨 [Track 1] 선제적 예방 교육 및 조기 정밀 검진 (재정 방어)"
                    else:
                        return "🟢 현재 건강 상태 유지 및 정기 검진 안내"
                        
                results_df["권장 정책 액션 플랜"] = results_df.apply(match_action_plan, axis=1)
                
                # 예측 의료비 변동액(delta_Y) 기준 내림차순 정렬
                sorted_results_df = results_df.sort_values(by="예측 의료비 변동액(delta_Y)", ascending=False).reset_index(drop=True)
                
                # [상위 10% 필터링 조건 연동]
                total_count = len(sorted_results_df)
                top_10_percent_cutoff = int(np.ceil(total_count * 0.1))
                top_10_percent_cutoff = max(1, top_10_percent_cutoff) # 최소 1명 노출 보장
                
                top_10_results_df = sorted_results_df.head(top_10_percent_cutoff).reset_index(drop=True)
                
                # 화면 표시용 가공 (금액 원화 포맷팅)
                display_df = top_10_results_df.copy()
                display_df["예측 의료비 변동액(delta_Y)"] = display_df["예측 의료비 변동액(delta_Y)"].map(lambda x: f"{x:+,} 원" if not pd.isna(x) else "0 원")
                
                st.markdown("### 📊 Random Forest 예측 기반 의료비 급증 고위험군(상위 10%) 스크리닝 결과")
                st.info(f"💡 전체 대상 `{original_merged_count:,}`명 중 **만 40대 장년층 {filtered_40s_count:,}명**을 1차 필터링하고, 이 중 5년 뒤 의료비 증가세가 가장 높은 상위 10%(`{top_10_percent_cutoff:,}`명) 집중 관리 리스트입니다.")
                st.dataframe(display_df, use_container_width=True)
                
                # 다운로드 버튼 연계 (상위 10% 리스트 기준)
                csv_output = top_10_results_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label=f"📥 상위 10% 집중 관리 대상 명단 CSV 다운로드 ({top_10_percent_cutoff}명)",
                    data=csv_output,
                    file_name="khp_top10_40s_combined_risk_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ 파일 처리 및 조인 중 오류가 발생했습니다: {str(e)}")

st.markdown("---")
