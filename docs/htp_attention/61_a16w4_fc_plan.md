# 61. HMX FC 경로의 a16w4 (계획서, 2026-10-18)

**상태 (2026-10-26): §3의 1–3단계와 5단계의 host 검사가 `defa7002c`에 들어갔다.** 4단계(skel 진입점, host
o-proj 배선)는 아직이다. Gate 0은 건너뛰었다. 대신 단말(v79) 벤치에서 a16w4 출력이 double 기준값과
f32 반올림 수준으로 일치하고, 같은 입력의 a8w4보다 오차가 몇 자릿수 작은 것을 확인했다.
- 커널: `hexkl_mm_opts.act16` → `hexkl_mm_u8i4_layer_run`이 두 번 sweep한다. 양자화:
  `hvx_quant_params_pack_u16_ah`. 합치기: `hvx_dq_tiles_job.lo_off`. 고정 encoding을 쓰는 producer를
  위해 `act_ah_prepacked`로 두 plane을 미리 받는 경로도 있다.
- 검사: `fc_layer_host_check.c`의 a16 케이스, `quant_u16_host_check.c`(실제 HVX 코드를 hvx_emu에서 실행,
  비트 단위 비교, 변형 검사 포함).
- 속도 작업 (`9a7a9bc74`):
  - accumulator를 라이브러리 int32 읽기 대신 convert 3회로 int24 byte plane에 쓴다
    (`hvx_dq_tiles_job.planes24`, K ≤ 4111). a8 경로도 같이 쓴다.
  - 패킷 루프의 정렬 검사는 호출당 한 번으로 줄였다.
  - dequant는 정렬 store를 쓴다.
  - 기각한 것: `weight.n:2x`는 처리량 모드가 아니라 int8 weight 읽기였다.
- 4단계 연결 시 지킬 것: 호출당 행 수는 staging buffer마다 결과 tile 32개가 남게 잡는다
  (K=2816 a16에서 768행). 남는 tile이 8개면 dequant worker를 기다리는 시간이 크게 늘어난다. a8의
  `fcMaxRows`에도 같은 효과가 있었다.
- 남은 격차: HMX가 노는 두 구간, 즉 시작 전 양자화와 f32 출력을 쓰는 worker 대기.
- HMX 탐침 결과 (v79, 2026-10-22, `run_main_on_hexagon`):
  - HMX 클럭 투표(`HAP_power_set_HMX_v2`)가 없으면 패킷이 몇 배 느리다. 투표를 더 세게 해도 그대로이므로
    투표한 상태의 패킷 시간이 하한이다. operand를 재사용해도, `:deep`을 써도 같다. HMX 시간을 재는
    탐침은 반드시 `nntr_hvx_open`처럼 투표해야 한다.
  - convert(`mxmem(..):after:retain:cm.ub = acc`) 다음의 clear에는 fence가 필요 없다. 64개 tile이
    int32 읽기와 비트 단위로 같았다.
  - u16 출력은 두 단계 형식으로만 가능하다: `bias = mxmem2` 256 B 블록(word c: fp16 scale,
    word c+32: bias), `cvt.uh = acc(Rt):2x1|2x2`, `mxmem(Rd, Rs) = cvt`.
  - `:2x1`은 2 KiB, 즉 u16 1024개를 쓴다. 4행 묶음마다 두 행 분량을 열 순서로 번갈아 쓰는 배치다.
    값은 대략 `acc · s · 257/512`이지만, scale 4에서는 "행 하나를 그대로 쓴다"는 해석과 "인접한
    두 행을 평균한다"는 해석에 원소가 반씩 맞았다. 행 두 개를 하나로 줄이는(2:1) 모드로 보인다.
  - `Rt`는 bit 12, 13, 16, 17, 18이 결과를 바꾼다. 의미는 미해독이다.
  - convert를 두 번 한 뒤 한 번 저장하면 0이 나온다.
  - 따라서 FC의 64×32 출력 tile을 u16으로 바로 꺼내는 방법은 아직 없다.

규칙은 `01_working_style.md`, 검증 순서는 `hexagon-gates` skill.

이 문서 시점 브랜치 `htp/gemma-e2b`, HEAD `f696c9347`. 줄 번호는 그 시점 기준이다.

참고 커밋 (plan 24 문서 `24_attention_a16_plan.md`는 이 트리에 없다):
- `48f04ddb4` a16 / kv8 attention 커널 — u8×i8 array 위의 16-bit Q·P, 이 계획의 원본
- `7d9490881` FC 호출이 u16 activation을 quant pass 안에서 dequant — §2에서 고칠 자리
- `f5a70c4d5` attention의 u16 context를 o-proj로 넘기는 배선
- `0547df38f` readout 합치기를 convert 한 tile 뒤로 미룬 최적화 — §3-2 파이프라인에 참고

---

## 0. 왜, 그리고 정말 필요한가

HTP context의 FC는 a8w8(`hexkl_mm_u8i8_dma.c`)과 a8w4(`hexkl_mm_u8i4_dma.c`)뿐이다. 예외가 attention core의
a16 / kv8 커널(`hexkl_attn_q2.c`, plan 24)이다. 이 문서는 그 a16 방식을 u8×i4 FC에 옮기는 계획이다.

지금 o-proj는 attention의 u16 context를 받아, DSP의 quant pass 안에서 f32로 dequant한 뒤 **u8로 다시
양자화**한다(`htp_compute_ops.cpp:1797-1824`의 `u16_in` 분기 → `nntr_hvx_mm_u8i4_layer_res_add_u16`,
`hexkl_mm_opts.h:70-79`의 `act_u16`). a16w4가 얻는 것은 이 u8 재양자화 손실을 없애는 것뿐이다.
속도는 얻지 못하고 잃는다(§5).

**Gate 0 (host, DSP 코드 작성 전):** 레이어별로 a8w4 / a16w4 / f32 출력 오차를 CPU에서 비교하고,
Gemma-4 E2B PPL 차이를 본다. 의미 있는 이득이 없으면 여기서 멈추고 이 문서에 결과만 적는다.

---

## 1. 수학 — attention의 a16 트릭 그대로

`hexkl_attn_q2.c:43-48`:

```
u   = 256·u_hi + u_lo                       (u16 activation을 두 u8 plane으로)
A·W = 256·(A_hi·W) + (A_lo·W)               (u8×i4 sweep 두 번, 각자 누산)
y   = s_a·s_w·(256·acc_hi + acc_lo − zp·colsum_w) + bias
```

attention과 다른 점:

- **readout이 더 단순하다.** attention은 convert unit plane(QK int16, PV int24 + 2^s shift)으로 읽는다.
  FC 경로는 이미 `hexkl_micro_hmx_acc_read_int32`로 int32를 직접 읽으므로(`hexkl_mm_u8i4_dma.c:783`)
  shift 계획이 필요 없다.
- **합치기는 f32에서.** sweep 하나의 |acc| ≤ 255·8·K는 K=8192에서도 int32 안이다. 하지만 256배하면
  K≳1024부터 int32를 넘는다. 기존 dequant가 colsum을 이미 f32로 다루므로(`hvx_dequant_i32.c:118`)
  같은 자리에서 `256·f(hi) + f(lo)`로 합친다. f32 반올림(상대 2^-24)은 16-bit 양자화 step보다 훨씬 작다.
- **zero point는 쪼개지 않는다.** zp·colsum 항은 dequant에서 한 번에 뺀다. 16-bit zp·colsum도 f32라
  넘치지 않는다.

---

## 2. 함정 — o-proj 입력은 head마다 encoding이 다르다

attention 출력은 head마다 (scale, zp)를 갖는다(`act16_enc`, `act16_hd`). K 방향으로 scale이 바뀌면
하나의 누산에서 scale을 밖으로 뺄 수 없다. head마다 따로 누산·readout하는 것은 head 수만큼 acc read가
늘어 비싸다.

**해결:** 행마다 u16 encoding 하나로 재양자화한다. quant pass가 이미 u16 → f32를 `pre_scratch`에 만들고
있으므로, 출력만 u8 대신 **per-row u16**으로 바꾸면 된다. f32 입력(qkv, FFN)도 같은 경로를 탄다.
16-bit라 head를 합쳐도 여유가 충분하다. 같은 재양자화를 attention decode의 GQA fold가 이미 한다
(`hexkl_attn_q2.c:263-290`, `q_dma_push`).

---

## 3. 구현 단계 (가장 짧은 diff)

1. **HVX quant pack** — `hvx_quant_u8.c`
   - `hvx_quant_params_pack_u8_ah`(`hvx_quant_u8.h:83`)에 16-bit 모드: per-row scale/zp를 [0, 65535]로,
     AH tile을 lo plane / hi plane 두 장으로 쓴다.
   - 바이트 분리는 `stage_qprep`의 `vdeal`(`hexkl_attn_q2.c:325-337`)을 재사용한다.
2. **커널** — `hexkl_mm_u8i4_dma.c` layer 경로
   - `hexkl_mm_opts`에 `act16` 플래그.
   - 출력 tile 루프(`:770-789`)를 hi sweep → acc read → clear → lo sweep → acc read로. `mm_u8i4_tile`은 그대로.
   - VTCM: activation 영역 ×2(`act_bytes`), staging ×2 → `acc_tiles` 절반(`:582`).
3. **Dequant** — `hvx_dequant_i32.c`
   - `DQ_TILE_ROW`가 lo tile도 읽어 `256·f(hi) + f(lo)`. `hvx_dq_tiles_job`에 lo tile offset 추가.
   - `!acc_layout->usable` fallback(`acc_scratch`)은 a16에서 `AEE_EUNSUPPORTED`로 거절한다(`out_off`와
     같은 방식). `ponytail:` 주석으로 천장과 업그레이드 경로를 남긴다.
4. **Skel / host 배선**
   - 새 entry 대신 `nntr_hvx_mm_u8i4_layer_res_add_u16`에 a16 모드 인자 하나(`test/htp/nntr_hvx.idl`).
   - host는 `gemm_qs4cx_res_add_fp32`의 `u16_in` 분기에서 켠다. 1차 대상은 o-proj만.
5. **검증**
   - `test/htp/host/fc_layer_host_check.c`에 a16w4 케이스: hi/lo 두 sweep 합이 u16×i4 int64 직접 dot와
     정수 단계에서 bit-exact, 최종 f32가 기준과 일치.
   - 그 뒤 `hexagon-gates` 순서. 기기에서 o-proj 시간과 PPL. 측정 instrumentation은 커밋하지 않는다.

---

## 4. 범위 밖

- **Decode (M=1):** HVX GEMV 경로라 이 HMX 커널을 타지 않는다. 그 경로가 이미 f32 activation을 받는지는
  확인하지 않았다 — 손대기 전에 확인할 것.
- **MoE 커널** `hexkl_mm_u8i4_moe.c`: Gate 0 결과를 보고 결정.
- **a16w8 FC** `hexkl_mm_u8i8_dma.c`: 같은 패턴이라 이 작업 뒤에 거의 그대로 옮길 수 있다.

---

## 5. 예상 비용과 미확인 사항

- HMX packet과 acc read가 2배, activation DMA와 VTCM도 2배. prefill o-proj의 HMX 구간은 약 2배로
  예상한다 — **가정이다.** 착수 시 a8w4 o-proj의 단계별 시간(`HEXKL_PROBE_*`)을 먼저 재고 판단한다.
- **v81 HMX에 네이티브 16-bit 정수 activation은 없다 (2026-10-18 확인).** 그래서 §3의 두 번 sweep이
  유일한 경로다.
  - SDK 6.4.0.2 어셈블러(`-mv81 -mhmx`)가 받는 activation 타입은 `.ub`와 `.hf`뿐이다. `.uh`/`.h`/`.b`는
    "invalid operand"로 거절된다.
  - QNN 2.47 `libQnnHtpV81Skel.so`의 "16-bit activation" 커널(`hmx_v73_convhnh*`, `hbh*`)도 SDK objdump로
    보면 `activation.ub` × `weight.n`/`.b`다. 16-bit는 출력 readout(`cvt.uh = acc:2x2`)에만 있다. skel
    전체의 패킷 조합은 `ub×b`, `ub×n`, `ub×c`, `hf×hf` 네 가지뿐이다. `native_hmx_a16w4`는 런타임 옵션
    문자열이고 커널이 아니다.
  - 어셈블러가 받는 `activation.hf × weight.n`을 S26 Ultra(v81) cDSP에서 실행해 봤다(HexKL micro +
    `run_main_on_hexagon`). int4로 디코딩되지 않는다. WH i4 tile의 바이트를 **fp16 비트 그대로** 읽는다:
    k=0의 nibble q는 q·2^-24(fp16 subnormal), k=1의 nibble은 halfword의 bit 8–11, k=5의 nibble은 bit
    12–15(지수부)에 들어간다. 대조군 hf×hf는 fp16 정밀도로 기준값과 일치했다. 따라서 fp16×int4
    네이티브 모드로는 쓸 수 없다.
  - 같은 실험에서 패킷당 시간은 hf×hf, hf×n, ub×n이 모두 같았다. 같은 tile을 반복한 측정이다.
    hf tile은 32행, ub tile은 64행이므로 행당 처리량은 ub가 2배다. 따라서 u8 두 번 sweep의 HMX 비용은
    hf 한 번 sweep과 같은 수준이다.
  - 실험 소스는 커밋하지 않았다. 다시 하려면 HexKL `examples/hexkl_micro_hmx_mm_f16`을 바탕으로,
    빌드 플래그에 `-mhmx`를 추가하면 된다.
