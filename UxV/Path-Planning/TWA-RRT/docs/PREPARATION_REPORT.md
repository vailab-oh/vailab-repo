# 공개용 폴더 정리 최종 보고

> 후속 변경: 공개용 TB/ST/STL의 3D 출력 블록을 제거했습니다. TWA/HPO는 기존 2D 출력을 유지합니다. 계산 로직과 CSV/JSON 저장은 변경하지 않았습니다.

> 현재 공개본은 후속 요청에 따라 JSON별 논문 case 하나씩, 총 4개만 포함합니다. 이름은 moderate/narrow_passage/cluttered/early_arrival이며 수치 설정은 유지했습니다. 아래 21-case 검증과 기존 ID 기술은 최초 준비 작업 기록입니다. 최신 목록과 제거/이름 변경 기록은 BENCHMARK_CASES.md 및 BENCHMARK_SELECTION.json을 참조하세요.

원본 실험 폴더 옆 `TWA-RRT-public`을 생성했습니다. 원본을 이동·수정·삭제하지 않았으며, GitHub 업로드는 수행하지 않았습니다. 실행 가능한 공개 준비본이지만, 아래 출처/논문 대조 항목 때문에 완전한 논문 재현 및 전체 코드의 공개 라이선스 확정 상태는 아닙니다.

## A. 생성 구조

```text
TWA-RRT-public/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── twa_rrt.py
├── run_early_arrival_experiment.py
├── baselines/
│   ├── tb_rrt.py
│   ├── st_rrt.py
│   ├── stl_rrt.py
│   └── hpo_rrt/
│       ├── hpo_rrt_star.py
│       ├── run_hpo_benchmark.py
│       ├── benchmark_adapter.py
│       ├── hpo_parameters.json
│       └── LICENSE_STATUS.md
├── benchmarks/
│   ├── benchmark_cases_2.json
│   ├── benchmark_cases_M1.json
│   ├── benchmark_cases_N1.json
│   └── benchmark_cases_M1_with_early_arrival.json
├── analysis/generate_comparison_table.py
├── docs/  # 전체 필드 감사·출처·논문 대조·검증·변경 diff
├── example_results/
│   ├── table1/  # 작은 과거 집계 CSV와 경로를 제거한 run ID 기록
│   └── figures/  # 기존 M3 성공 trial에서 생성한 Fig. 5a/5b 예제
└── output/.gitkeep
```

## B. 복사 및 이름 변경

| 원본 상대경로 | 공개용 상대경로 |
|---|---|
| STL_RRT_Prop/stl_rrt_Prop_r7_wiser_R7_r2.py | twa_rrt.py |
| TB_RRT/tb_rrt_r7_260226.py | baselines/tb_rrt.py |
| ST_RRT/st_rrt_r4_260228.py | baselines/st_rrt.py |
| STL_RRT_r0/stl_rrt_r6_260227.py | baselines/stl_rrt.py |
| HPO-RRT/hpo_rrt_star_r1.py | baselines/hpo_rrt/hpo_rrt_star.py |
| HPO-RRT/run_hpo_rrt_star_r1.py | baselines/hpo_rrt/run_hpo_benchmark.py |
| HPO-RRT/benchmark_adapter.py | baselines/hpo_rrt/benchmark_adapter.py |
| HPO-RRT/hpo_parameters.json | baselines/hpo_rrt/hpo_parameters.json |
| generate_comparison_table_r7.py | analysis/generate_comparison_table.py |
| run_early_arrival_experiment.py | run_early_arrival_experiment.py |
| benchmark_cases_2.json | benchmarks/benchmark_cases_2.json |
| benchmark_cases_M1.json | benchmarks/benchmark_cases_M1.json |
| benchmark_cases_N1.json | benchmarks/benchmark_cases_N1.json |
| benchmark_cases_M1_with_early_arrival.json | benchmarks/benchmark_cases_M1_with_early_arrival.json |

TWA/TB/ST/STL 파일은 알고리즘과 자체 runner/plotter를 포함합니다. HPO는 core·runner·adapter가 분리되어 있으며 adapter가 TWA 평가와 ST 충돌 검사를 읽습니다. comparison은 5개 방법의 집계/분포도와 Fig. 5 패널을 담당하고, early-arrival runner는 M5를 별도 100회 설정으로 실행합니다. HPO 구현 메모 두 파일은 실행 경로가 맞는 문서로 정리해 docs에 보존했습니다.

## C. Benchmark JSON 변경

| JSON | Case 수 | 삭제 | 수정 | 유지 |
|---|---:|---|---|---|
| benchmark_cases_2.json | 8 | 모든 case의 algorithms.st_rrt.dt_min | outputs.out_dir를 output으로; meta 이름의 wiser 표기를 twa로(있는 경우) | 나머지 모든 값 |
| benchmark_cases_M1.json | 4 | 동일 | 동일 | 나머지 모든 값 |
| benchmark_cases_N1.json | 4 | 동일 | 동일 | 나머지 모든 값 |
| benchmark_cases_M1_with_early_arrival.json | 5 | 동일 | 동일 | 나머지 모든 값 |

삭제 근거: ST run_case가 해당 JSON key를 읽지 않고, rewiring의 동명 지역변수는 거리/v_max에서 별도로 계산합니다. TB/STL의 dt_min은 실제로 읽으므로 유지했습니다. v_min/a_max/v0 및 WISER의 가중치·시간창 필드는 현재 효과가 작아 보여도 loader/default 또는 평가에 연결되므로 유지했습니다. obstacle id도 추적성을 위해 유지했습니다. 미사용 추정만으로 추가 삭제하지 않았습니다.

world/start/goal/obstacles/time/dynamics/random/planner 및 모든 case ID의 값 일치를 확인했습니다. M5 원본 5회와 실제 실험 runner 100회의 차이를 보존·문서화했으며 smoke test는 임시 사본만 1회로 설정했습니다. benchmark_cases.json 및 benchmark_cases_3.json은 현재 주 실험 명령/완료 기록에서 필요성이 확인되지 않은 이전 설정으로 제외했습니다.

## D. 누락 및 확인되지 않은 파일

- HPO runner: 발견 및 포함. summary.csv, raw_trials.jsonl, aggregate.json, config_used.json 생성과 분석기 수용 확인.
- Fig. 5: comparison 내부 성공 trial 전용 함수 존재. 기존 결과 기반 2개 예제 패널 포함.
- Fig. 6–8: 개별 plotter는 존재하지만 최종 조합 스크립트·정확한 trial/seed 선택표·확정된 원본 패널 세트는 missing. 수작업 조합 여부도 확정하지 않음.
- 최종 논문 DOI/출판 정보: 심사 중이므로 미정. 제목과 Taegeun Oh 저자는 첨부 PDF에서 확인해 CITATION에 반영.
- 전체 raw 결과: 대용량 공개 제외 원칙에 따라 포함하지 않음. 같은 역사적 run을 재집계하려면 외부 raw archive가 필요.

## E. 위험 요소 및 논문 대조

1. Table 1의 Moderate/Narrow-passage 수치가 실제 source caseA3/caseN2와 반대로 연결됩니다. 예: A3의 TWA 94%, 1.984454초가 논문 Narrow에, N2의 100%, 1.346157초가 Moderate에 대응합니다. 원본 수치나 ID를 변경하지 않았습니다.
2. HPO의 A3 runtime 0.077은 성공 trial 평균(전체 평균 0.110217)이고 다른 표 셀은 전체 평균을 사용합니다. 시간창 위반 비율도 서로 다른 집계 정의가 혼용되어 있습니다. MANUSCRIPT_CROSSCHECK.md에 정확한 열과 값을 기록했습니다.
3. Baseline의 원저자 코드 사용 여부·재구현 여부·재배포 라이선스가 완전히 확인되지 않았습니다. TWA 코드에만 MIT를 적용했습니다. TB는 논문 TB-RRT와 현재 star 확장이 다르고 ST는 단일 트리, STL/HPO는 정적 공통 benchmark로 적응되어 있습니다.
4. 내부 wiser_rrt/class 이름, 기존 CLI --wiser_run 및 HPO output ID hpo_rrt_r1은 호환성을 위해 유지했습니다. 사람이 보는 이름은 TWA-RRT*로 정리했습니다.
5. 상대 output 경로와 HPO 동적 import를 새 구조에 맞게 수정했습니다. 개인 절대경로·계정정보·API key 패턴을 공개 파일에서 검사했습니다.
6. 과거 실행시간은 환경과 native 종료 조건의 영향을 받습니다. 6000 상한이 같은 실제 탐색량을 뜻하지 않습니다.

## F. 실행 검증

| Planner | 1 trial 실행 | 시간창 success | 원본 대비 비시간 지표 |
|---|---|---:|---|
| TWA-RRT* | 정상 종료·그림 저장 | 1 | 동일 |
| TB-RRT* | 정상 종료·그림 저장 | 1 | 동일 |
| ST-RRT* | 정상 종료·그림 저장 | 1 | 동일 |
| STL-RRT* | 정상 종료·그림 저장 | 0 | 동일 |
| HPO-RRT* | 정상 종료·그림 저장 | 1 | 동일 |

N2/seed 42/6000 예산, 원본·사본 총 10회 실행입니다. STL의 success=0은 두 버전에서 같은 시간창 실패이며 실행 오류가 아닙니다. 9개 Python 문법, 21개 case의 정식 JSON Schema 검증, 112개 core 정의의 구조 동일성, comparison 실행, HPO 부속 결과 파일 입력을 확인했습니다. 100-trial 전체 재실험은 수행하지 않았습니다. 선택한 원본 코드/설정/참조 문서 17개의 SHA-256이 최초 기록과 일치합니다. 전체 대용량 결과 폴더의 해시를 검증했다는 의미는 아닙니다.

## G. GitHub 공개 전 최종 확인

- [ ] 각 baseline/분석 코드/데이터의 저작권자·upstream 재사용 여부와 라이선스 확인.
- [ ] Table 1의 scenario label 연결과 HPO runtime/violation 집계 정의 확정.
- [ ] Fig. 6–8의 원본 trial·seed·run ID와 조합 과정 추가.
- [ ] 논문 게재 확정 후 CITATION DOI/출판 정보 업데이트.
- [ ] 동일 역사적 통계 재집계가 필요하면 raw archive와 실행환경 기록을 별도 공개.
- [ ] 실제 공개 파일 목록과 각 파일의 라이선스 범위를 확인한 뒤 업로드.

## 원본 대비 차이와 충분성 평가

알고리즘 계산은 유지하고 파일명·폴더·참조경로·출력 위치·표시명·주석/문서 및 검증된 legacy JSON key만 정리했습니다. 원본 버전군·캐시·임시 테스트·대용량 결과·논문 원고·개인 경로는 공개본에서 제외했습니다.

주요 알고리즘 실행, 주 benchmark 재실험, 표 집계 및 Fig. 5 패널 생성에 필요한 코드는 갖추었습니다. 그러나 baseline 공개 권리, 논문 Table 1 불일치, Fig. 6–8 원본 선택 정보가 해결되지 않아 **논문의 모든 주요 결과를 정확히 재현하는 최종 공개 repository로 확정할 수는 없습니다.** 이 제한을 숨기지 않고 실행 가능한 준비본과 감사 결과를 제공합니다.
