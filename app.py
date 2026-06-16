import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 애플리케이션 대시보드의 기본 테마 및 전체 레이아웃 설정
st.set_page_config(
    page_title="머신러닝 회귀 파이프라인 분석 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("다중 특성 회귀 모델 분석 및 실시간 시뮬레이션 웹 서비스")
st.markdown("""
본 대시보드는 WHO의 국가별 보건 통계 데이터를 기반으로 **선형 회귀, 다항 회귀, 그리고 릿지 규제 모델**의 일반화 성능을 비교합니다.
제한된 데이터 환경에서 복잡도가 높은 모델이 유도하는 과대적합 문제를 정량적으로 분석하고, 규제 항이 모델 안정성에 미치는 영향을 시각적으로 파악할 수 있습니다.
""")
st.write("---")

# =================================================================
# 내부 데이터 처리 및 실험 환경 재현 (수치 및 규격 일관성 확보)
# =================================================================
@st.cache_data
def load_and_preprocess_dataset():
    url = "https://github.com/dongupak/DataML/raw/main/csv/life_expectancy.csv"
    data = pd.read_csv(url)
    # [Key Error 완전 예방]
    data.columns = data.columns.str.strip().str.lower()
    data = data.dropna()
    return data

life_data = load_and_preprocess_dataset()
selected_features = ['adult mortality', 'bmi', 'gdp']  # 정제된 소문자 피처 지정
target_variable = 'life expectancy'

X = life_data[selected_features]
y = life_data[target_variable]

# 1단계 파일과의 통계량 일치를 위해 난수 시드 기반으로 동일 분할 및 샘플링 수행
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train_all.sample(n=50, random_state=42)
y_train = y_train_all.loc[X_train.index]

# =================================================================
# 직렬화된 외부 학습 모델 파일(.pkl) 로드 및 예외 처리
# =================================================================
try:
    models = {
        "Linear": joblib.load("model_linear.pkl"),
        "Poly": joblib.load("model_poly.pkl"),
        "Ridge": joblib.load("model_ridge.pkl")
    }
except FileNotFoundError:
    st.error("⚠️ [시스템 경고] 사전 학습된 .pkl 모델 파일을 탐색할 수 없습니다. 1단계 파이썬 코드를 먼저 구동해 주세요.")
    st.stop()

# =================================================================
# 세션 1: 통계적 모델 성능 분석 및 기하학적 비교 차트 렌더링
# =================================================================
st.header("1️⃣ 학습 알고리즘별 정량적 평가 지표 대조")
st.markdown("학습에 참여한 3가지 파이프라인 모델의 훈련/검증 데이터별 결정계수($R^2$) 및 평균제곱오차(MSE) 지표입니다.")

# 평가지표 수집 및 통계 데이터프레임 구조화
performance_records = []
for name, model in models.items():
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    r2_train = r2_score(y_train, train_predictions)
    r2_test = r2_score(y_test, test_predictions)
    mse_train = mean_squared_error(y_train, train_predictions)
    mse_test = mean_squared_error(y_test, test_predictions)

    # 각 모델 파이프라인의 구조적 복잡도 추적
    if 'poly_features' in model.named_steps:
        complexity = model.named_steps['poly_features'].n_output_features_
    else:
        complexity = X_train.shape[1] + 1

    performance_records.append({
        "Model Type": f"{name} Regression",
        "Complexity (최종 차원수)": int(complexity),
        "Train R² (결정계수)": round(r2_train, 4),
        "Test R² (결정계수)": round(r2_test, 4),
        "Train MSE": round(mse_train, 4),
        "Test MSE": round(mse_test, 4)
    })

df_summary = pd.DataFrame(performance_records)

# 시각적 가시성을 극대화하기 위해 다중 컬럼 분할 기법 적용
left_layout, right_layout = st.columns([4, 3])

with left_layout:
    st.write("##### 통계적 평가지표 데이터프레임 [상시 출력]")
    # 고정형 테이블 형태로 지표 배치하여 가독성 강화
    st.table(df_summary.set_index("Model Type"))
    st.caption("**통계적 고찰:** 규제 장치가 결여된 Poly 모델은 차수 확장으로 인해 훈련 데이터에 극단적으로 동화되어 "
               "Test 데이터 세트에서 일반화 성능이 완전히 붕괴되는 현상을 보입니다. 반면, L2 가중치 벌칙이 부여된 Ridge 모델은 "
               "다항 차수를 동일하게 유지하더라도 안정적인 일반화 예측력을 유지하는 것을 확인할 수 있습니다.")

with right_layout:
    st.write("##### 검증 데이터 기준 결정계수(Test R²) 대조군 차트")
    # Matplotlib 라이브러리를 활용한 수치 시각화 커스텀 빌드
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    # palette 경고를 피하기 위해 x축 변수를 hue에 직접 매핑하고 범례를 off
    sns.barplot(
        data=df_summary,
        x="Model Type",
        y="Test R² (결정계수)",
        hue="Model Type",
        palette=["#3498DB", "#E74C3C", "#2ECC71"],
        legend=False,
        ax=axis,
        zorder=3
    )

    # 과대적합 모델의 마이너스 무한대 발산 현상에 따른 차트 깨짐 방지 설정
    axis.set_ylim(bottom=max(-1.5, df_summary["Test R² (결정계수)"].min() - 0.2), top=1.1)
    axis.set_title("Generalization Performance (Test R² Comparison)", fontsize=10, fontweight='bold')
    axis.set_xlabel("Evaluated Regression Pipelines")
    axis.set_ylabel("R² Metric")

    # 데이터 라벨 수치 직접 결합
    for patch in axis.patches:
        patch_height = patch.get_height()
        axis.annotate(f"{patch_height:.3f}", (patch.get_x() + patch.get_width() / 2., patch_height),
                    ha='center', va='bottom',
                    xytext=(0, 3), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()
    st.pyplot(figure)

st.write("---")

# =================================================================
# 세션 2: 인터랙티브 사이드바 연동 및 실시간 서빙 시스템
# =================================================================
st.header("2️⃣ 사용자 지정 도메인 변수를 통한 실시간 동적 시뮬레이션")
st.markdown("좌측 제어 패널에서 가상의 보건 및 경제적 입력값을 슬라이더로 조절하면, 즉각적으로 학습된 모델들이 반응하여 기대수명을 실시간 산출합니다.")

# 사이드바 입력 인터페이스 설계
st.sidebar.header("도메인 독립변수 수치 미세조정")
st.sidebar.markdown("시뮬레이션할 국가의 통계 범위를 가상으로 지정합니다.")

# 왜곡된 예측 입력을 예방하기 위해 데이터셋의 실제 요약 통계량(Min, Max, Mean)을 슬라이더 경계값으로 바인딩
val_mortality = st.sidebar.slider(
    "Adult Mortality (성인 사망률 기준 범위)",
    int(life_data['adult mortality'].min()),
    int(life_data['adult mortality'].max()),
    int(life_data['adult mortality'].mean())
)

val_bmi = st.sidebar.slider(
    "BMI (체질량지수 기준 범위)",
    float(life_data['bmi'].min()),
    float(life_data['bmi'].max()),
    float(life_data['bmi'].mean()),
    step=0.1
)

val_gdp = st.sidebar.slider(
    "GDP (국내총생산 기준 범위)",
    int(life_data['gdp'].min()),
    int(life_data['gdp'].max()),
    int(life_data['gdp'].mean())
)

# 실시간 분석 알고리즘 전환용 드롭다운 UI 배치
selected_algorithm = st.selectbox(
    "실시간 예측 연산에 주입할 머신러닝 파이프라인 모델 선택:",
    ["Linear", "Poly", "Ridge"]
)

# 사용자 가상 데이터를 모델 주입 표준 규격인 데이터프레임으로 직렬화 (소문자 표준 이름 엄격 일치)
user_input_dataframe = pd.DataFrame([{
    'adult mortality': val_mortality,
    'bmi': val_bmi,
    'gdp': val_gdp
}])

# 지정된 파이프라인 모델 호출 및 연산 처리
target_pipeline = models[selected_algorithm]
live_prediction_result = target_pipeline.predict(user_input_dataframe)[0]

# 분석 결과를 대시보드 중앙에 시각적인 카드 형태로 배치
st.markdown("### 실시간 예측 연산 결과")
dashboard_card_html = f"""
<div style="
    background-color: #fcfcfc;
    padding: 25px;
    border-radius: 10px;
    border-left: 5px solid #3498DB;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02);
">
    <p style="font-size: 15px; color: #555; margin-bottom: 5px;">
        활성화된 알고리즘: <strong>{selected_algorithm} Regression Pipeline</strong>
    </p>
    <p style="font-size: 14px; color: #888; margin: 0;">입력된 국가 통계 인프라 기준 예측 결과</p>
    <h1 style="font-size: 42px; font-weight: 800; color: #2C3E50; margin: 12px 0;">
        {live_prediction_result:.2f} <span style="font-size: 22px; font-weight: 400; color: #666;">세 (Years)</span>
    </h1>
</div>
"""
st.markdown(dashboard_card_html, unsafe_allow_html=True)
