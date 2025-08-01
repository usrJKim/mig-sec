import pandas as pd

# 파일 불러오기
df = pd.read_csv("prior_res.csv")  # 파일명 수정 필요

EPS = 0.5  # 변화 감지 허용 오차

levels = []
start_idx = 0

for i in range(1, len(df)):
    v_prev = df.loc[i - 1, "power_w"]
    v_curr = df.loc[i, "power_w"]

    if abs(v_curr - v_prev) > EPS:
        # 레벨이 증가한 경우에만 평균 기록
        if v_curr > v_prev:
            level = df.loc[start_idx:i - 1, "power_w"].mean()
            levels.append(level)
        # 어쨌든 시작 인덱스는 이동 (변화 감지 기준이니까)
        start_idx = i

# 마지막 구간이 증가구간이면 기록
if df.loc[len(df)-1, "power_w"] >= df.loc[len(df)-2, "power_w"]:
    levels.append(df.loc[start_idx:, "power_w"].mean())

# 레벨 간 gap 계산
gaps = [round(levels[i+1] - levels[i], 2) for i in range(len(levels) - 1)]

print("Detected levels (excluding downward transitions):", levels)
print("Level-to-level gaps:", gaps)
