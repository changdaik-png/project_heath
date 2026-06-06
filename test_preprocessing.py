import pandas as pd
import pytest

# 아직 구현되지 않은 preprocessing 모듈에서 전처리 함수를 가져옵니다.
# 실제 preprocessing.py가 없으므로 테스트 실행 시 ModuleNotFoundError로 실패하게 됩니다.
from preprocessing import merge_and_calculate_delta

def test_delta_y_calculation():
    """
    가상의 2017년 데이터와 2022년 데이터를 PIDWON 기준으로 병합(Merge)했을 때,
    2022년 의료비에서 2017년 의료비를 뺀 delta_Y가 정확하게 계산되는지 검증하는 테스트입니다.
    
    - 2017년 의료비 컬럼: I_MEDICALEXP1 (개인 연간 총 의료비 본인부담금)
    - 2022년 의료비 컬럼: EROOP(응급) + INOOP(입원) + OUOOP_1(외래) 의 합산액
    """
    # 1. 2017년 가상 데이터 생성 (KHP 1기 기준)
    data_2017 = pd.DataFrame({
        'PIDWON': [101, 102, 103],
        'I_MEDICALEXP1': [100000, 200000, 300000]
    })

    # 2. 2022년 가상 데이터 생성 (KHP 2기 기준)
    # 103번 개인은 탈퇴, 104번 개인은 신규 진입했다고 가정합니다.
    data_2022 = pd.DataFrame({
        'PIDWON': [101, 102, 104],
        'EROOP': [10000, 20000, 5000],
        'INOOP': [50000, 100000, 0],
        'OUOOP_1': [90000, 180000, 45000]
    })

    # 3. 전처리 함수 실행 (병합 및 delta_Y 계산)
    # PID 101: 2022년 의료비(150,000) - 2017년 의료비(100,000) = delta_Y(50,000)
    # PID 102: 2022년 의료비(300,000) - 2017년 의료비(200,000) = delta_Y(100,000)
    result_df = merge_and_calculate_delta(data_2017, data_2022)

    # 4. 검증 (Assertion)
    # Inner Join 기준으로 공통으로 존재하는 PIDWON만 남아야 합니다.
    assert len(result_df) == 2
    
    # PIDWON을 인덱스로 변환하여 개별 delta_Y 값 검증
    result_df = result_df.set_index('PIDWON')
    
    assert result_df.loc[101, 'delta_Y'] == 50000
    assert result_df.loc[102, 'delta_Y'] == 100000
