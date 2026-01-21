# 프로젝트 2 백테스트 GUI 실행 방법

이 폴더는 **GUI(그래픽 화면)**에서 정적/동적/종합 포트폴리오 백테스트를 실행할 수 있게 구성되어 있습니다.

## 1) 설치(가상환경 권장)

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) 실행

```bash
python main.py
```

실행하면 GUI 창이 뜨고,
- 정적 포트폴리오
- 동적 포트폴리오
- 종합 포트폴리오

중 원하는 항목을 체크한 뒤 **Run** 버튼으로 실행합니다.

## 3) 결과 확인

각 탭(정적/동적/종합)에 아래가 표시됩니다.
- 그래프(누적수익률 + 드로다운)
- 성과 지표 테이블: 누적수익률 / CAGR / MDD / Sharpe / Vol

### 저장 기능
각 탭에서
- **CSV 저장**: 선택한 실행 결과(포트폴리오/자산별 누적수익률 등)를 CSV로 저장
- **PNG 저장**: 현재 그래프를 PNG로 저장

저장 경로는 버튼 클릭 시 파일 선택 창에서 지정합니다.

## 4) 참고
- 데이터는 인터넷을 통해 다운로드됩니다(FinanceDataReader 등). 네트워크 환경에 따라 시간이 걸리거나 실패할 수 있습니다.
- 한글이 포함된 제목/라벨은 자동으로 한글 폰트를 설정하도록 되어 있습니다(Windows: 맑은 고딕).
