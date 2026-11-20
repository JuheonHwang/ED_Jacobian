# 0. Setup / 환경 준비

[← README](../README.md) | [Next: ED model →](01_ed_model.md)

## Requirements / 요구 사항

| | |
|---|---|
| MATLAB | developed and tested on R2026b only; the `Name=Value` call syntax needs R2021a or newer / R2026b에서만 개발·검증, `Name=Value` 호출 문법은 R2021a 이상 필요 |
| Optimization Toolbox | `lsqnonlin` |
| Computer Vision Toolbox | nearest-neighbour search / 최근접 탐색 (`pointCloud`) |
| Parallel Computing Toolbox | optional, only for the `parfor` comparison in chapter 6 / 선택 사항, 6장의 `parfor` 비교에만 사용 |

**EN** — No GPU, no Statistics Toolbox and no OpenCV/mexopencv are needed. The original scripts required all three (`gpuArray`, `knnsearch`, `cv.Rodrigues`); the refactored code runs on the CPU, searches neighbours through `pointCloud`, and has its own Rodrigues implementation.

**KO** — GPU, Statistics Toolbox, OpenCV/mexopencv는 필요 없습니다. 원본 스크립트는 셋 다 필요했지만(`gpuArray`, `knnsearch`, `cv.Rodrigues`), 정리한 코드는 CPU에서 돌고, 이웃 탐색은 `pointCloud`로 하며, Rodrigues는 자체 구현을 씁니다.

## Layout / 폴더 구조

```
+edreg/            library: model, deformation, residual + Jacobian, checks, registration
  +solvers/        hand-written optimisers (chapter 5)
+jwutils/          mesh I/O and display helpers
tutorials/         t01 ... t06, one script per chapter
docs/              this tutorial
data/              sample meshes
results/figures/   figures written by the tutorials
tests/             unit tests
```

**EN** — `+edreg` and `+jwutils` are MATLAB packages: functions are called as `edreg.buildModel(...)`, and the *parent* folder (the repository root) must be on the path. Every tutorial script adds it by itself.

**KO** — `+edreg`와 `+jwutils`는 MATLAB 패키지입니다. 함수는 `edreg.buildModel(...)`처럼 호출하고, *상위* 폴더(저장소 루트)가 path에 있어야 합니다. 튜토리얼 스크립트는 각자 알아서 path를 추가합니다.

## Data / 데이터

| File | Vertices | Faces | Role / 역할 |
|---|---|---|---|
| `data/human_020_f10k.ply` | 5,027 | 9,995 | source / 소스 |
| `data/human_022_f10k.ply` | 5,033 | 10,000 | target / 타깃 |
| `data/human_020_f20k.ply` | 10,038 | 19,995 | larger source for chapter 6 / 6장용 큰 소스 |
| `data/human_022_f20k.ply` | 10,042 | 19,999 | larger target / 큰 타깃 |

**EN** — The meshes are two frames (020 and 022) of a human body scan sequence, Poisson-reconstructed and decimated. Coordinates are in millimetres. Any pair of triangle meshes works: the method uses no body-specific prior.

**Not included.** The original experiments also used the full-resolution sequence (`Dataset7CResult/mat_mesh_*.mat`) and an 80k-face pair. These are not in the repository. To run the 80k comparison in chapter 6, place the files at `data/human_020_f80k.ply` and `data/human_022_f80k.ply`; the script skips that case when they are missing.

**KO** — 메쉬는 사람 전신 스캔 시퀀스의 두 프레임(020, 022)을 Poisson 재구성한 뒤 단순화한 것입니다. 좌표 단위는 밀리미터입니다. 삼각 메쉬 쌍이면 무엇이든 쓸 수 있습니다. 이 방법은 인체에 특화된 사전 정보를 쓰지 않습니다.

**포함되지 않은 데이터.** 원래 실험에서는 전체 해상도 시퀀스(`Dataset7CResult/mat_mesh_*.mat`)와 face 8만 개짜리 쌍도 사용했습니다. 이 파일들은 저장소에 없습니다. 6장의 80k 비교를 돌리려면 `data/human_020_f80k.ply`, `data/human_022_f80k.ply` 위치에 파일을 두면 되고, 없으면 스크립트가 그 경우를 건너뜁니다.

## Running / 실행

**EN** — From the MATLAB prompt, in the repository root:

**KO** — MATLAB 프롬프트에서, 저장소 루트를 현재 폴더로 두고:

```matlab
run('tutorials/t01_ed_model.m')
```

**EN** — Or headless from a shell. Figures are written to `results/figures/` either way.

**KO** — 또는 셸에서 화면 없이 실행합니다. 어느 쪽이든 그림은 `results/figures/`에 저장됩니다.

```bash
matlab -batch "run('tutorials/t01_ed_model.m')"
```

**EN** — To run the unit tests, or every tutorial in order:

**KO** — 단위 테스트 실행, 또는 튜토리얼 전체를 순서대로 실행:

```matlab
runtests('tests')
run_all_tutorials      % all six chapters, about two minutes / 여섯 장 전체, 약 2분
```

## Quick start / 빠른 시작

```matlab
src = edreg.loadMesh('data/human_020_f10k.ply');
tgt = edreg.loadMesh('data/human_022_f10k.ply');

model = edreg.buildModel(src.V, src.F, Seed=1);        % ED graph on the source
[P, out] = edreg.register(model, tgt);                 % lsqnonlin + analytic Jacobian

edreg.showMeshes({struct('V', out.Vd, 'F', src.F), tgt});
```

[Next: ED model / 다음: ED 모델 →](01_ed_model.md)
