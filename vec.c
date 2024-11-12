#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

typedef struct Flags {
  bool active;
  uint32_t per_lane;
} Flags;
#define ALL_LANES_ENABLED 0xffffffffu

typedef union VReg {
  uint32_t u[32];
  float f[32];
} VReg;

typedef union DstRow {
  uint32_t u[16];
  float f[16];
} DstRow;

typedef struct Env {
  VReg vreg[16];
  Flags flags;
  uint32_t flags_stack_ptr;
  Flags flags_stack[8];
  uint32_t shfl_bug[4];
  VReg prng;
  uint32_t rwc_dst;
  DstRow dst[512];
} Env;

#define insn_bits(start_bit, nbits) ((insn >> start_bit) & ((1u << nbits) - 1))
#define insn_bits_signed(start_bit, nbits) (((int32_t)(insn << (32 - start_bit - nbits)) >> (32 - nbits)))

static void Exec_Undefined(Env* env, uint32_t insn) {
  // Calls to this function act as markers for undefined behaviour.
}

static uint32_t HalfToFloat_Correct(uint16_t x) {
  uint32_t bits = (x & 0x7fffu);
  uint32_t sign = (x & 0x8000u);
  if (bits >= 0x7c00u) {
    // NaN or Inf
    bits += 0x38000u;
  } else if (bits) {
    // Non-zero
    if (!(bits & 0x7c00u)) {
      // Denormal
      uint32_t n = __builtin_clz(bits) - 21;
      bits <<= n;
      bits -= (n << 10);
    }
    bits += 0x1c000u;
  }
  return (sign * 8 + bits) << 13;
}

static uint32_t HalfToFloat_FastAndLoose(uint16_t x) {
  // Just the "normal" path of HalfToFloat_Correct. This is what the hardware
  // does in most (all?) cases.
  uint32_t bits = (x & 0x7fffu);
  uint32_t sign = (x & 0x8000u);
  bits += 0x1c000u;
  return (sign * 8 + bits) << 13;
}

static void WriteVReg(Env* env, uint32_t insn, uint32_t vd, VReg* contents) {
  if (vd >= 8) {
    // Cannot write to constants via this path.
    return;
  }

  uint32_t per_lane = env->flags.per_lane;
  if (env->flags.active && per_lane != ALL_LANES_ENABLED) {
    // Non-trivial flags? Write the enabled lanes, preserve other lanes.
    for (uint32_t i = 0; i < 32; ++i) {
      if (per_lane & (1u << i)) {
        env->vreg[vd].u[i] = contents->u[i];
      }
    }
  } else {
    // Write all the lanes.
    memcpy(&env->vreg[vd], contents, sizeof(VReg));
  }
}

static void WriteVRegIndirect(Env* env, uint32_t insn, VReg* contents) {
  uint32_t per_lane = env->flags.active ? env->flags.per_lane : ALL_LANES_ENABLED;
  for (uint32_t i = 0; i < 32; ++i) {
    if (per_lane & (1u << i)) {
      // The low four bits of L7 specify which Lreg to write to.
      uint32_t lane_vd = env->vreg[7].u[i] & 15;
      if (lane_vd >= 8) {
        // Cannot write to constants via this path.
      } else {
        env->vreg[lane_vd].u[i] = contents->u[i];
      }
    }
  }
}

static void RefineFlags(Env* result, uint32_t per_lane) {
  result->flags.per_lane &= result->flags.active ? per_lane : 0;
}

static float u2f(uint32_t u) {
  float result;
  memcpy(&result, &u, 4);
  return result;
}

static uint32_t f2u(float f) {
  uint32_t result;
  memcpy(&result, &f, 4);
  return result;
}

static uint32_t FusedMulAdd(uint32_t a, uint32_t b, uint32_t c) {
  // Denormal inputs treated as +0, as is -0.
  if (!(a & 0x7f800000u)) a = 0;
  if (!(b & 0x7f800000u)) b = 0;
  if (!(c & 0x7f800000u)) c = 0;

  uint32_t result = f2u(fmaf(u2f(a), u2f(b), u2f(c)));

  // Denormal outputs flushed to +0, as is -0.
  uint32_t exp = result & 0x7f800000u;
  if (!(result & 0x7f800000u)) result = 0;

  // NaN outputs always 0x7f800001 or 0xff800001
  // NB: This is a reasonable approximation to hardware, but doesn't match it.
  if ((result & 0x7fffffffu) > 0x7f800000u) result = (result & 0x80000000) + 0x7f800001;

  return result;
}

static uint32_t StepPRNGLane(uint32_t input) {
  // This is an LFSR with four taps. This really is what the hardware does.
  // It repeats every 3758096376 steps. As it only steps by one bit every
  // time, consecutive outputs have 31 of their 32 bits in common.
  uint32_t xnor_of_taps = (__builtin_popcount(input & 0x80200003) & 1) ^ 1;
  return (xnor_of_taps << 31) + (input >> 1);
}

static void StepPRNG(Env* env) {
  for (uint32_t i = 0; i < 32; ++i) {
    env->prng.u[i] = StepPRNGLane(env->prng.u[i]);
  }
}

static void SeedPRNG(Env* env, uint32_t seed) {
  // The logic within this function corresponds to writing the seed value
  // to PRNG_SEED_Seed_Val_ADDR32 and then executing a bunch of SFPNOP.

  for (uint32_t i = 0; i < 32; ++i) {
    seed = StepPRNGLane(seed);
  }
  // This method of distributing the seed to the 32 lanes is really bad,
  // as it means that adjacent lanes have 30 of their 32 bits in common.
  // It is however what the hardware does.
  for (uint32_t i = 32; i != 0;) {
    --i;
    seed = StepPRNGLane(seed);
    seed = StepPRNGLane(seed);
    env->prng.u[i] = seed;
  }
}

// --- SFPLOAD ---

#define SFPLOAD_OPCODE 0x70

#define SFPLOAD_MOD0_FMT_SRCB         0
#define SFPLOAD_MOD0_FMT_FP16A        1
#define SFPLOAD_MOD0_FMT_FP16B        2
#define SFPLOAD_MOD0_FMT_FP32         3
#define SFPLOAD_MOD0_FMT_INT32        4
#define SFPLOAD_MOD0_FMT_INT32_TO_SM 12
#define SFPLOAD_MOD0_FMT_INT8         5
#define SFPLOAD_MOD0_FMT_INT8_COMP   13
#define SFPLOAD_MOD0_FMT_LO16         6
#define SFPLOAD_MOD0_FMT_LO16_ONLY   14
#define SFPLOAD_MOD0_FMT_HI16         7
#define SFPLOAD_MOD0_FMT_HI16_ONLY   15

static uint32_t HalfToFloat_Dst(uint32_t x) {
  x >>= 16; // Low 16 bits ignored
  // Then an exotic packing of fp16 fields into 16 bits, and a slightly fast
  // and loose conversion.
  uint32_t hi_mant = x & 0x7f;
  uint32_t exp = (x >> 7) & 0x1f;
  uint32_t lo_mant = (x >> 12) & 7;
  uint32_t sign = x >> 15;
  if (exp) exp += 112;
  return (sign << 31) + (exp << 23) + (hi_mant << 16) + (lo_mant << 13);
}

static uint32_t Bf16ToFloat_Dst(uint32_t x) {
  return x & 0xffff0000;
}

static uint32_t Identity_Dst(uint32_t x) {
  return x;
}

static uint32_t SignMag32ToInt32(uint32_t x) {
  return x & 0x80000000u ? -(x & 0x7fffffffu) : x;
}

static uint32_t SignMag8ToSignMag32_Dst(uint32_t x) {
  x >>= 16; // Low 16 bits ignored
  uint32_t hi_mag = x & 0xf;
  uint32_t lo_mag = (x >> 12) & 7;
  uint32_t sign = x >> 15;
  return (sign << 31) + (hi_mag << 3) + lo_mag;
}

static uint32_t SignMag11ToInt32_Dst(uint32_t x) {
  x >>= 16; // Low 16 bits ignored
  uint32_t hi_mag = x & 0x7f;
  uint32_t lo_mag = (x >> 12) & 7;
  uint32_t sign = x >> 15;
  uint32_t result = (hi_mag << 3) + lo_mag;
  return sign && result ? -result : result;
}

static uint32_t UnpackSignMag16_DstHi(uint32_t x) {
  x >>= 16; // Low 16 bits ignored
  uint32_t hi_mag = x & 0x7f;
  uint32_t lo_mag = (x >> 7) & 0xff;
  uint32_t sign = x >> 15;
  return (sign << 31) + (hi_mag << 24) + (lo_mag << 16);
}

static uint32_t UnpackSignMag16_DstLo(uint32_t x) {
  return UnpackSignMag16_DstHi(x) >> 16;
}

static void Exec_SFPLOAD(Env* env, uint32_t insn) {
  uint32_t imm10 = insn_bits(0, 10);
  uint32_t am = insn_bits(14, 2);
  uint32_t mod0 = insn_bits(16, 4);
  uint32_t vd = insn_bits(20, 4);

  imm10 += env->rwc_dst;
  uint32_t odd = (imm10 & 2) ? 1 : 0;
  imm10 &= 0x1fc;

  uint32_t preserve = 0;
  uint32_t (*converter)(uint32_t);
  switch (mod0) {
  case SFPLOAD_MOD0_FMT_SRCB: return Exec_Undefined(env, insn); // I have not implemented this.
  case SFPLOAD_MOD0_FMT_FP16A:       converter = HalfToFloat_Dst; break;
  case SFPLOAD_MOD0_FMT_FP16B:       converter = Bf16ToFloat_Dst; break;
  case SFPLOAD_MOD0_FMT_FP32:        converter = Identity_Dst; break;
  case SFPLOAD_MOD0_FMT_INT32:       converter = Identity_Dst; break;
  case SFPLOAD_MOD0_FMT_INT32_TO_SM: converter = SignMag32ToInt32; break;
  case SFPLOAD_MOD0_FMT_INT8:        converter = SignMag8ToSignMag32_Dst; break;
  case SFPLOAD_MOD0_FMT_INT8_COMP:   converter = SignMag11ToInt32_Dst; break;
  case SFPLOAD_MOD0_FMT_LO16:        converter = UnpackSignMag16_DstLo; break;
  case SFPLOAD_MOD0_FMT_LO16_ONLY:   converter = UnpackSignMag16_DstLo, preserve = 0xffff0000; break;
  case SFPLOAD_MOD0_FMT_HI16:        converter = UnpackSignMag16_DstHi; break;
  case SFPLOAD_MOD0_FMT_HI16_ONLY:   converter = UnpackSignMag16_DstHi, preserve = 0x0000ffff; break;
  default: return Exec_Undefined(env, insn);
  }

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t val = env->vreg[vd].u[i]; // NB: Old contents of VD only used when `preserve != 0`.
    uint32_t dst = env->dst[imm10 + (i / 8)].u[odd + (i & 7) * 2];
    val = (val & preserve) | converter(dst);
    result.u[i] = val;
  }
  WriteVReg(env, insn, vd, &result);

  (void)am; // I have not implemented this.
}

// --- SFPLOADI ---

#define SFPLOADI_OPCODE 0x71

#define SFPLOADI_MOD0_FLOATB 0 // Immediate is bf16
#define SFPLOADI_MOD0_FLOATA 1 // Immediate is fp16 (ish)
#define SFPLOADI_MOD0_USHORT 2 // Immediate is u16
#define SFPLOADI_MOD0_SHORT  4 // Immediate is i16
#define SFPLOADI_MOD0_UPPER  8 // Immediate overwrites upper 16 bits
#define SFPLOADI_MOD0_LOWER 10 // Immediate overwrites lower 16 bits

static void Exec_SFPLOADI(Env* env, uint32_t insn) {
  uint16_t imm16 = insn_bits(0, 16);
  uint32_t mod0 = insn_bits(16, 4);
  uint32_t vd = insn_bits(20, 4);

  uint32_t imm32;
  uint32_t preserve = 0;
  switch (mod0) {
  case SFPLOADI_MOD0_FLOATB: imm32 = (uint32_t)imm16 << 16; break;
  case 9:
  case SFPLOADI_MOD0_FLOATA: imm32 = HalfToFloat_FastAndLoose(imm16); break;
  case 3:
  case SFPLOADI_MOD0_USHORT: imm32 = imm16; break;
  case 5: case 6: case 7: case 12: case 13:
  case SFPLOADI_MOD0_SHORT: imm32 = (uint32_t)(int32_t)(int16_t)imm16; break;
  case SFPLOADI_MOD0_UPPER: imm32 = (uint32_t)imm16 << 16, preserve = 0x0000ffff; break;
  case 11: case 14: case 15:
  case SFPLOADI_MOD0_LOWER: imm32 = imm16, preserve = 0xffff0000; break;
  default: return Exec_Undefined(env, insn);
  }

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t val = env->vreg[vd].u[i]; // NB: Old contents of VD only used when `preserve != 0`.
    val = (val & preserve) | imm32;
    result.u[i] = val;
  }
  WriteVReg(env, insn, vd, &result);
}

// --- SFPSTORE ---

#define SFPSTORE_OPCODE 0x72

static void Exec_SFPSTORE(Env* env, uint32_t insn) {
  uint32_t imm10 = insn_bits(0, 10);
  uint32_t am = insn_bits(14, 2);
  uint32_t mod0 = insn_bits(16, 4);
  uint32_t vd = insn_bits(20, 4);

  imm10 += env->rwc_dst;
  uint32_t odd = (imm10 & 2) ? 1 : 0;
  imm10 &= 0x1fc;

  if (vd >= 8) {
    return Exec_Undefined(env, insn);
  }

  (void)mod0; // I have not implemented the data type conversions.

  uint32_t per_lane = env->flags.active ? env->flags.per_lane : ALL_LANES_ENABLED;
  for (uint32_t i = 0; i < 32; ++i) {
    if (per_lane & (1u << i)) {
      env->dst[imm10 + (i / 8)].u[odd + (i & 7) * 2] = env->vreg[vd].u[i];
    }
  }

  (void)am; // I have not implemented this.
}

// --- SFPLUT ---

#define SFPLUT_OPCODE 0x73

#define SFPLUT_MOD0_SGN_UPDATE 0
#define SFPLUT_MOD0_SGN_RETAIN 4

#define SFPLUT_MOD0_INDIRECT_VD 8

static uint32_t Fp8ToFloat_LUT(uint32_t v) {
  // This is not any kind of standard Fp8 format!
  if ((uint8_t)v == 0xff) {
    return 0;
  } else {
    uint32_t sign = (v & 0x80) << 24;
    uint32_t exp  = (127 - ((v & 0x70) >> 4)) << 23;
    uint32_t mant = (v & 0xf) << 19;
    return sign + exp + mant;
  }
}

static void Exec_SFPLUT(Env* env, uint32_t insn) {
  uint32_t mod0 = insn_bits(16, 4);
  uint32_t vd = insn_bits(20, 4);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t d = env->vreg[3].u[i];
    int32_t e = (int32_t)((d >> 23) & 0xff) - 126;
    if (e < 0) e = 0;
    else if (e > 2) e = 2;
    uint32_t tmp = env->vreg[e].u[i];
    uint32_t val = FusedMulAdd(Fp8ToFloat_LUT(tmp >> 8), d & 0x7fffffffu, Fp8ToFloat_LUT(tmp));
    if (mod0 & SFPLUT_MOD0_SGN_RETAIN) {
      val = (val & 0x7fffffffu) + (d & 0x80000000u);
    }
    result.u[i] = val;
  }

  if (mod0 & SFPLUT_MOD0_INDIRECT_VD) {
    WriteVRegIndirect(env, insn, &result);
  } else {
    WriteVReg(env, insn, vd, &result);
  }
}

// --- SFPMULI ---

#define SFPMULI_OPCODE 0x74

#define SFPMULI_MOD1_INDIRECT_VD 8

static void Exec_SFPMULI(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t imm16 = insn_bits(8, 16);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD *= Bf16ToFp32(Imm16)
    result.u[i] = FusedMulAdd(env->vreg[vd].u[i], imm16 << 16, 0);
  }

  if (mod1 & SFPMULI_MOD1_INDIRECT_VD) {
    WriteVRegIndirect(env, insn, &result);
  } else {
    WriteVReg(env, insn, vd, &result);
  }
}

// --- SFPADDI ---

#define SFPADDI_OPCODE 0x75

#define SFPADDI_MOD1_INDIRECT_VD 8

static void Exec_SFPADDI(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t imm16 = insn_bits(8, 16);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD += Bf16ToFp32(Imm16)
    result.u[i] = FusedMulAdd(env->vreg[vd].u[i], 0x3f800000u, imm16 << 16);
  }

  if (mod1 & SFPADDI_MOD1_INDIRECT_VD) {
    WriteVRegIndirect(env, insn, &result);
  } else {
    WriteVReg(env, insn, vd, &result);
  }
}

// --- SFPDIVP2 ---

#define SFPDIVP2_OPCODE 0x76

#define SFPSDIVP2_MOD1_ADD 1

static void Exec_SFPDIVP2(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);
  uint32_t imm12 = insn_bits(12, 12);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t old_val = env->vreg[vc].u[i];
    uint32_t new_exp = imm12 << 23;
    if (mod1 & SFPSDIVP2_MOD1_ADD) {
      // VD = {VC.Sign, VC.Exponent ± Imm7, VC.Mantissa}
      if ((uint8_t)(old_val >> 23) == 0xff) {
        // If VC is Inf or NaN, VC is unchanged.
        new_exp = 0;
      }
      new_exp += old_val;
    } else {
      // VD = {VC.Sign, Imm8, VC.Mantissa}
    }
    result.u[i] = (new_exp & 0x7f800000u) + (old_val & 0x807fffffu);
  }
  WriteVReg(env, insn, vd, &result);
}

// --- SFPEXEXP ---

#define SFPEXEXP_OPCODE 0x77

#define SFPEXEXP_MOD1_DEBIAS   0 // Subtract fp32 exponent bias (result in range -127 through +128)
#define SFPEXEXP_MOD1_NODEBIAS 1 // Do not subtract fp32 exponent bias (result in range 0 through 255)
#define SFPEXEXP_MOD1_SET_CC_SGN_EXP       2 // Refine flags based on result < 0
#define SFPEXEXP_MOD1_SET_CC_COMP_EXP      8
#define SFPEXEXP_MOD1_SET_CC_SGN_COMP_EXP 10 // Refine flags based on result >= 0

static void Exec_SFPEXEXP(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  VReg result;
  uint32_t flags = mod1 & SFPEXEXP_MOD1_SET_CC_SGN_EXP ? 0 : ALL_LANES_ENABLED;
  uint32_t bias = mod1 & SFPEXEXP_MOD1_NODEBIAS ? 0 : 127;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD = VC.Exponent - bias
    uint32_t val = ((env->vreg[vc].u[i] >> 23) & 0xff) - bias;
    flags |= (val >> 31) << i;
    result.u[i] = val;
  }

  WriteVReg(env, insn, vd, &result);
  if (mod1 & (SFPEXEXP_MOD1_SET_CC_SGN_EXP | SFPEXEXP_MOD1_SET_CC_COMP_EXP)) {
    // Refine flags based on VD < 0
    if (mod1 & SFPEXEXP_MOD1_SET_CC_COMP_EXP) {
      // ... or inverse thereof.
      flags = ~flags;
    }
    RefineFlags(env, flags);
  }
}

// --- SFPEXMAN ---

#define SFPEXMAN_OPCODE 0x78

#define SFPEXMAN_MOD1_PAD8 0 // Include implicit mantissa bit in result (even when input is zero!)
#define SFPEXMAN_MOD1_PAD9 1 // Do not include implicit mantissa bit in result

static void Exec_SFPEXMAN(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  VReg result;
  uint32_t implicit = mod1 & SFPEXMAN_MOD1_PAD9 ? 0 : 1u << 23;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD = {0, !Imm1, VC.Mantissa}
    result.u[i] = (env->vreg[vc].u[i] & 0x7fffffu) + implicit;
  }
  WriteVReg(env, insn, vd, &result);
}

// --- SFPIADD ---

#define SFPIADD_OPCODE 0x79

#define SFPIADD_MOD1_ARG_LREG_DST        0
#define SFPIADD_MOD1_ARG_IMM             1 // Use immediate as 2nd source rather than VD
#define SFPIADD_MOD1_ARG_2SCOMP_LREG_DST 2 // Two's complement (i.e. negation) of VD
#define SFPIADD_MOD1_CC_LT0  0 // Refine flags based on result < 0
#define SFPIADD_MOD1_CC_NONE 4 // Don't update flags
#define SFPIADD_MOD1_CC_GTE0 8 // Refine flags based on result >= 0

static void Exec_SFPIADD(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);
  int32_t imm12 = insn_bits_signed(12, 12);

  VReg result;
  if (mod1 & SFPIADD_MOD1_ARG_IMM) {
    // VD = VC ± Imm11
    for (uint32_t i = 0; i < 32; ++i) {
      result.u[i] = env->vreg[vc].u[i] + (uint32_t)imm12;
    }
  } else if (mod1 & SFPIADD_MOD1_ARG_2SCOMP_LREG_DST) {
    // VD = VC - VD
    for (uint32_t i = 0; i < 32; ++i) {
      result.u[i] = env->vreg[vc].u[i] - env->vreg[vd].u[i];
    }
  } else {
    // VD = VC + VD
    for (uint32_t i = 0; i < 32; ++i) {
      result.u[i] = env->vreg[vc].u[i] + env->vreg[vd].u[i];
    }
  }
  WriteVReg(env, insn, vd, &result);

  uint32_t flags = 0;
  if (insn & SFPIADD_MOD1_CC_NONE) {
    flags = ALL_LANES_ENABLED;
  } else {
    // Refine flags based VD < 0
    for (uint32_t i = 0; i < 32; ++i) {
      flags |= (result.u[i] >> 31) << i;
    }
  }
  if (insn & SFPIADD_MOD1_CC_GTE0) {
    // ... or inverse thereof.
    flags = ~flags;
  }
  RefineFlags(env, flags);
}

// --- SFPSHFT ---

#define SFPSHFT_OPCODE 0x7a

#define SFPSHFT_MOD1_ARG_IMM 1 // Use immediate as shift amount rather than VC

static uint32_t Shift(uint32_t value, int32_t amount) {
  if (amount >= 0) {
    return value << (amount & 31);
  } else {
    return value >> ((-amount) & 31);
  }
}

static void Exec_SFPSHFT(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);
  int32_t imm12 = insn_bits_signed(12, 12);

  VReg result;
  if (mod1 & SFPSHFT_MOD1_ARG_IMM) {
    // VD = VD << Imm or VD = VD >> -Imm
    for (uint32_t i = 0; i < 32; ++i) {
      result.u[i] = Shift(env->vreg[vd].u[i], imm12);
    }
  } else {
    // VD = VD << VC or VD = VD >> -VC
    for (uint32_t i = 0; i < 32; ++i) {
      result.u[i] = Shift(env->vreg[vd].u[i], env->vreg[vc].u[i]);
    }
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPSETCC ---

#define SFPSETCC_OPCODE 0x7b

#define SFPSETCC_MOD1_IMM_BIT0  1 // Refine flags based on immediate bit
#define SFPSETCC_MOD1_LREG_LT0  0 // Refine flags based on VC < 0
#define SFPSETCC_MOD1_LREG_NE0  2 // Refine flags based on VC != 0
#define SFPSETCC_MOD1_LREG_GTE0 4 // Refine flags based on VC >= 0
#define SFPSETCC_MOD1_LREG_EQ0  6 // Refine flags based on VC == 0
#define SFPSETCC_MOD1_CLEAR     8

static void Exec_SFPSETCC(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vc = insn_bits(8, 4);
  int32_t imm12 = insn_bits(12, 12);

  if (mod1 & SFPSETCC_MOD1_CLEAR) {
    RefineFlags(env, 0);
  } else if (mod1 & SFPSETCC_MOD1_IMM_BIT0) {
    RefineFlags(env, (imm12 & 1) ? ALL_LANES_ENABLED : 0);
  } else {
    // Refine flags based on VC != 0 or VC < 0
    uint32_t mask = mod1 & SFPSETCC_MOD1_LREG_NE0 ? 0xffffffffu : 0x80000000u;
    uint32_t flags = 0;
    for (uint32_t i = 0; i < 32; ++i) {
      if (env->vreg[vc].u[i] & mask) {
        flags |= 1u << i;
      }
    }
    if (mod1 & SFPSETCC_MOD1_LREG_GTE0) {
      // ... or inverse thereof.
      flags = ~flags;
    }
    RefineFlags(env, flags);
  }
}

// --- SFPMOV ---

#define SFPMOV_OPCODE 0x7c

#define SFPMOV_MOD1_COMPSIGN 1
#define SFPMOV_MOD1_SPECIAL  8

static void Exec_SFPMOV(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  if (mod1 == SFPMOV_MOD1_SPECIAL) {
    if (vc == 9) {
      StepPRNG(env);
      WriteVReg(env, insn, vd, &env->prng);
      return;
    } else {
      return Exec_Undefined(env, insn);
    }
  }

  VReg result;
  uint32_t xor = mod1 & SFPMOV_MOD1_COMPSIGN ? 0x80000000 : 0;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD = VC or VD = -VC
    uint32_t val = env->vreg[vc].u[i];
    result.u[i] = val ^ xor;
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPABS ---

#define SFPABS_OPCODE 0x7d

#define SFPABS_MOD1_FLOAT 1

static void Exec_SFPABS(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD = Abs(VC)
    uint32_t val = env->vreg[vc].u[i];
    if (val & 0x80000000) {
      if (mod1 & SFPABS_MOD1_FLOAT) {
        // Float negate, but don't touch NaN
        if (val <= 0xff800000u) {
          val = val & 0x7fffffffu;
        }
      } else {
        // Integer negate
        val = -val;
      }
    }
    result.u[i] = val;
  }
  WriteVReg(env, insn, vd, &result);
}

// --- SFPAND ---

#define SFPAND_OPCODE 0x7e

static void Exec_SFPAND(Env* env, uint32_t insn) {
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD &= VC
    result.u[i] = env->vreg[vd].u[i] & env->vreg[vc].u[i];
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPOR ---

#define SFPOR_OPCODE 0x7f

static void Exec_SFPOR(Env* env, uint32_t insn) {
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD |= VC
    result.u[i] = env->vreg[vd].u[i] | env->vreg[vc].u[i];
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPNOT ---

#define SFPNOT_OPCODE 0x80

static void Exec_SFPNOT(Env* env, uint32_t insn) {
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD = ~VC
    result.u[i] = ~env->vreg[vc].u[i];
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPLZ ---

#define SFPLZ_OPCODE 0x81

#define SFPLZ_MOD1_CC_NE0     2 // Refine flags based on input != 0
#define SFPLZ_MOD1_NOSGN_MASK 4 // Mask off sign bit of input
#define SFPLZ_MOD1_CC_COMP    8 // Refine flags based on input == 0

static void Exec_SFPLZ(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  VReg result;
  uint32_t flags = insn & SFPLZ_MOD1_CC_NE0 ? 0 : ALL_LANES_ENABLED;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD = CountLeadingZeros(VC)
    uint32_t val = env->vreg[vc].u[i];
    if (mod1 & SFPLZ_MOD1_NOSGN_MASK) {
      val &= 0x7fffffffu;
    }
    if (val != 0) {
      result.u[i] = __builtin_clz(val);
      flags |= 1u << i;
    } else {
      result.u[i] = 32;
    }
  }

  WriteVReg(env, insn, vd, &result);
  if (insn & (SFPLZ_MOD1_CC_NE0 | SFPLZ_MOD1_CC_COMP)) {
    // Refine flags based on VC != 0
    if (insn & SFPLZ_MOD1_CC_COMP) {
      // ... or inverse thereof.
      flags = ~flags;
    }
    RefineFlags(env, flags);
  }
}

// --- SFPSETEXP ---

#define SFPSETEXP_OPCODE 0x82

#define SFPSETEXP_MOD1_ARG_MANTISSA 0
#define SFPSETEXP_MOD1_ARG_IMM      1
#define SFPSETEXP_MOD1_ARG_EXPONENT 2

static void Exec_SFPSETEXP(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);
  uint32_t imm12 = insn_bits(12, 12);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t new_exp_bits;
    if (mod1 & SFPSETEXP_MOD1_ARG_IMM) {
      // VD = {VC.Sign, Imm8, VC.Mantissa}
      new_exp_bits = imm12 << 23;
    } else if (mod1 & SFPSETEXP_MOD1_ARG_EXPONENT) {
      // VD = {VC.Sign, VD.Exponent, VC.Mantissa}
      new_exp_bits = env->vreg[vd].u[i];
    } else {
      // VD = {VC.Sign, VD.Mantissa & 255, VC.Mantissa}
      new_exp_bits = env->vreg[vd].u[i] << 23;
    }
    result.u[i] = (new_exp_bits & 0x7f800000u) + (env->vreg[vc].u[i] & 0x807fffffu);
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPSETMAN ---

#define SFPSETMAN_OPCODE 0x83

#define SFPSETMAN_MOD1_ARG_IMM 1

static void Exec_SFPSETMAN(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);
  uint32_t imm12 = insn_bits(12, 12);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD = {VC.Sign, VC.Exponent, VD.Mantissa} or
    // VD = {VC.Sign, VC.Exponent, Imm12 << 11}
    uint32_t new_man = mod1 & SFPSETMAN_MOD1_ARG_IMM ? imm12 << 11 : env->vreg[vd].u[i] & 0x7fffffu;
    result.u[i] = (env->vreg[vc].u[i] & 0xff800000u) + new_man;
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPMAD ---

#define SFPMAD_OPCODE 0x84
#define SFPADD_OPCODE 0x85
#define SFPMUL_OPCODE 0x86

#define SFPMAD_MOD1_INDIRECT_VA 4
#define SFPMAD_MOD1_INDIRECT_VD 8

static void Exec_SFPMAD(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);
  uint32_t vb = insn_bits(12, 4);
  uint32_t va = insn_bits(16, 4);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD = VA * VB + VC
    uint32_t lane_va;
    if (mod1 & SFPMAD_MOD1_INDIRECT_VA) {
      lane_va = env->vreg[7].u[i] & 15;
    } else {
      lane_va = va;
    }
    result.u[i] = FusedMulAdd(env->vreg[lane_va].u[i], env->vreg[vb].u[i], env->vreg[vc].u[i]);
  }

  if (mod1 & SFPMAD_MOD1_INDIRECT_VD) {
    WriteVRegIndirect(env, insn, &result);
  } else {
    WriteVReg(env, insn, vd, &result);
  }
}

// --- SFPPUSHC ---

#define SFPPUSHC_OPCODE 0x87

#define SFPPUSHC_MOD1_PUSH 0 // Push on to stack

static void Exec_SFPPUSHC(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);

  if (mod1 == SFPPUSHC_MOD1_PUSH) {
    uint32_t flags_stack_ptr = env->flags_stack_ptr;
    env->flags_stack[flags_stack_ptr & 7] = env->flags;
    env->flags_stack_ptr = flags_stack_ptr + 1;
  }
}

// --- SFPPOPC ---

#define SFPPOPC_OPCODE 0x88

#define SFPPOPC_MOD1_POP            0 // Regular pop from stack
#define SFPPOPC_MOD1_READ_TOP       1 // flags = stack_top, do not pop
#define SFPPOPC_MOD1_INVERSE_OF_TOP 2 // flags = ~stack_top, do not pop
#define SFPPOPC_MOD1_ALL_ENABLED    4 // flags = ALL_LANES_ENABLED, do not pop

static void Exec_SFPPOPC(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);

  uint32_t flags_stack_ptr = env->flags_stack_ptr;
  if (flags_stack_ptr & 0xf) {
    env->flags = env->flags_stack[(flags_stack_ptr - 1) & 7];
  } else {
    env->flags.active = true;
    env->flags.per_lane = ALL_LANES_ENABLED;
  }
  if (mod1 & SFPPOPC_MOD1_INVERSE_OF_TOP) {
    env->flags.per_lane = ~env->flags.per_lane;
  }
  if (mod1 & SFPPOPC_MOD1_ALL_ENABLED) {
    env->flags.per_lane = ALL_LANES_ENABLED;
  }
  if (mod1 == SFPPOPC_MOD1_POP) {
    env->flags_stack_ptr = flags_stack_ptr - 1;
  }
}

// --- SFPSETSGN ---

#define SFPSETSGN_OPCODE 0x89

#define SFPSETSGN_MOD1_ARG_IMM 1

static void Exec_SFPSETSGN(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);
  int32_t imm12 = insn_bits(12, 12);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD = {VD.Sign, VC.Exponent, VC.Mantissa} or
    // VD = {   Imm1, VC.Exponent, VC.Mantissa}
    uint32_t new_sign = mod1 & SFPSETSGN_MOD1_ARG_IMM ? imm12 << 31 : env->vreg[vd].u[i] & 0x80000000;
    result.u[i] = new_sign + (env->vreg[vc].u[i] & 0x7fffffffu);
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPENCC ---

#define SFPENCC_OPCODE 0x8a

#define SFPENCC_MOD1_EC 1 // Complement active
#define SFPENCC_MOD1_EI 2 // Set active from immediate bit
#define SFPENCC_MOD1_RI 8 // Set per_lane from immediate bit (otherwise set it to ALL_LANES_ENABLED)

#define SFPENCC_IMM12_E 1 // When SFPENCC_MOD1_EI, immediate bit for active
#define SFPENCC_IMM12_R 2 // When SFPENCC_MOD1_RI, immediate bit for per_lane

static void Exec_SFPENCC(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint16_t imm12 = insn_bits(12, 12);

  if (mod1 & SFPENCC_MOD1_EC) {
    env->flags.active = !env->flags.active;
  }
  if (mod1 & SFPENCC_MOD1_EI) {
    env->flags.active = (imm12 & SFPENCC_IMM12_E) ? true : false;
  }

  if (mod1 & SFPENCC_MOD1_RI) {
    env->flags.per_lane = (imm12 & SFPENCC_IMM12_R) ? ALL_LANES_ENABLED : 0;
  } else {
    env->flags.per_lane = ALL_LANES_ENABLED;
  }
}

// --- SFPCOMPC ---

#define SFPCOMPC_OPCODE 0x8b

static void Exec_SFPCOMPC(Env* env, uint32_t insn) {
  Flags stack_top;
  if (env->flags_stack_ptr & 0xf) {
    stack_top = env->flags_stack[(env->flags_stack_ptr - 1) & 7];
  } else {
    stack_top.active = true;
    stack_top.per_lane = ALL_LANES_ENABLED;
  }

  if (stack_top.active && env->flags.active) {
    env->flags.per_lane = stack_top.per_lane & ~env->flags.per_lane;
  } else {
    env->flags.per_lane = 0;
  }
}

// --- SFPTRANSP ---

#define SFPTRANSP_OPCODE 0x8c

static void Transpose4(VReg* v) {
  uint32_t tmp[8];
  for (uint32_t i = 0; i < 4; ++i) {
    for (uint32_t j = 0; j < i; ++j) {
      uint32_t* a = &v[i].u[j*8];
      uint32_t* b = &v[j].u[i*8];
      memcpy(tmp, a, sizeof(tmp));
      memcpy(a, b, sizeof(tmp));
      memcpy(b, tmp, sizeof(tmp));
    }
  }
}

static void Exec_SFPTRANSP(Env* env, uint32_t insn) {
  Transpose4(env->vreg);
  Transpose4(env->vreg + 4);
}

// --- SFPXOR ---

#define SFPXOR_OPCODE 0x8d

static void Exec_SFPXOR(Env* env, uint32_t insn) {
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD ^= VC
    result.u[i] = env->vreg[vd].u[i] ^ env->vreg[vc].u[i];
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPSTOCHRND ---

#define SFPSTOCHRND_OPCODE 0x8e

#define SFPSTOCHRND_MOD1_FP32_TO_FP16A  0
#define SFPSTOCHRND_MOD1_FP32_TO_FP16B  1
#define SFPSTOCHRND_MOD1_FP32_TO_UINT8  2
#define SFPSTOCHRND_MOD1_FP32_TO_INT8   3
#define SFPSTOCHRND_MOD1_INT32_TO_UINT8 4
#define SFPSTOCHRND_MOD1_INT32_TO_INT8  5
#define SFPSTOCHRND_MOD1_FP32_TO_UINT16 6
#define SFPSTOCHRND_MOD1_FP32_TO_INT16  7

#define SFPSTOCHRND_MOD1_IMM8 8

#define SFPSTOCHRND_RND_EVEN  0
#define SFPSTOCHRND_RND_STOCH 1

static void Exec_SFPSTOCHRND(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);
  uint32_t vb = insn_bits(12, 4);
  uint32_t imm5 = insn_bits(16, 5);
  uint32_t rnd = insn_bits(21, 1);

  if (rnd & SFPSTOCHRND_RND_STOCH) {
    StepPRNG(env);
  }

  uint32_t limit;
  switch (mod1 &~ SFPSTOCHRND_MOD1_IMM8) {
  case SFPSTOCHRND_MOD1_FP32_TO_FP16A: imm5 = 13; break;
  case SFPSTOCHRND_MOD1_FP32_TO_FP16B: imm5 = 16; break;
  case SFPSTOCHRND_MOD1_FP32_TO_UINT8: limit = 255; break;
  case SFPSTOCHRND_MOD1_FP32_TO_INT8: limit = 127; break;
  case SFPSTOCHRND_MOD1_FP32_TO_UINT16: limit = 65535; break;
  case SFPSTOCHRND_MOD1_FP32_TO_INT16: limit = 32767; break;
  case SFPSTOCHRND_MOD1_INT32_TO_UINT8: limit = 255; break;
  case SFPSTOCHRND_MOD1_INT32_TO_INT8: limit = 127; break;
  }

  VReg result;
  switch (mod1 &~ SFPSTOCHRND_MOD1_IMM8) {
  case SFPSTOCHRND_MOD1_FP32_TO_FP16A:
  case SFPSTOCHRND_MOD1_FP32_TO_FP16B:
    for (uint32_t i = 0; i < 32; ++i) {
      uint64_t u64 = env->vreg[vc].u[i];
      uint32_t exp = (u64 >> 23) & 0xff;
      if (exp == 0) {
        // Denormal? Becomes zero.
        u64 = 0;
      } else if (exp == 0xff) {
        // Inf or NaN? Becomes ±Inf.
        u64 = (u64 & 0x80000000u) | 0x7f800000u;
      }
      u64 <<= 32 - imm5;
      u64 += rnd & SFPSTOCHRND_RND_STOCH ? env->prng.u[i] : 0x80000000u;
      result.u[i] = (u64 >> 32) << imm5;
    }
    break;
  case SFPSTOCHRND_MOD1_FP32_TO_UINT8:
  case SFPSTOCHRND_MOD1_FP32_TO_INT8:
  case SFPSTOCHRND_MOD1_FP32_TO_UINT16:
  case SFPSTOCHRND_MOD1_FP32_TO_INT16:
    for (uint32_t i = 0; i < 32; ++i) {
      uint64_t u64 = env->vreg[vc].u[i];
      uint32_t sign = (mod1 & 1) ? u64 & 0x80000000u : 0;
      uint32_t exp = (u64 >> 23) & 0xff;
      u64 = (u64 & 0x7fffff) | 0x800000;
      if (exp >= 149) {
        u64 = limit;
      } else {
        if (exp >= 118) {
          u64 <<= exp - 118;
        } else if (exp >= 87) {
          u64 >>= 118 - exp;
        } else {
          u64 = 0;
        }
        u64 += rnd & SFPSTOCHRND_RND_STOCH ? env->prng.u[i] : 0x80000000u;
        u64 >>= 32;
        if (u64 > limit) u64 = limit;
      }
      result.u[i] = (u64 ? sign : 0) + (uint32_t)u64;
    }
    break;
  case SFPSTOCHRND_MOD1_INT32_TO_UINT8:
  case SFPSTOCHRND_MOD1_INT32_TO_INT8:
    for (uint32_t i = 0; i < 32; ++i) {
      uint64_t u64 = env->vreg[vc].u[i];
      uint32_t sign = (mod1 & 1) ? u64 & 0x80000000u : 0;
      u64 &= 0x7fffffffu;
      u64 <<= 32 - (mod1 & SFPSTOCHRND_MOD1_IMM8 ? imm5 : env->vreg[vb].u[i] & 31);
      u64 += rnd & SFPSTOCHRND_RND_STOCH ? env->prng.u[i] : 0x80000000u;
      u64 >>= 32;
      if (u64 > limit) u64 = limit;
      result.u[i] = (u64 ? sign : 0) | (uint32_t)u64;
    }
    break;
  }
  WriteVReg(env, insn, vd, &result);
}

// --- SFPNOP ---

#define SFPNOP_OPCODE 0x8f

// --- SFPCAST ---

#define SFPCAST_OPCODE 0x90

#define SFPCAST_MOD1_RND_EVEN  0
#define SFPCAST_MOD1_RND_STOCH 1

static void Exec_SFPCAST(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  if (mod1 & SFPCAST_MOD1_RND_STOCH) {
    StepPRNG(env);
  }

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    // VD = SignMag32ToFp32(VC)
    uint64_t u64 = env->vreg[vc].u[i];
    if (u64 & 0x7fffffffu) {
      uint32_t sign = (uint32_t)u64 & 0x80000000u;
      u64 &= 0x7fffffffu;
      uint32_t e = __builtin_clz((uint32_t)u64);
      u64 <<= 24 + e;
      u64 += (uint64_t)(157 - e) << 55;
      // Rounding
      if (mod1 & SFPCAST_MOD1_RND_STOCH) {
        u64 += env->prng.u[i];
      } else {
        u64 += 0x7fffffffu + ((u64 >> 32) & 1);
      }
      u64 = sign + (u64 >> 32);
    }
    result.u[i] = (uint32_t)u64;
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPSWAP ---

#define SFPSWAP_OPCODE 0x92

#define SFPSWAP_MOD1_SWAP               0
#define SFPSWAP_MOD1_VEC_MIN_MAX        1
#define SFPSWAP_MOD1_SUBVEC_MIN01_MAX23 2
#define SFPSWAP_MOD1_SUBVEC_MIN02_MAX13 3
#define SFPSWAP_MOD1_SUBVEC_MIN03_MAX12 4
#define SFPSWAP_MOD1_SUBVEC_MIN0_MAX123 5
#define SFPSWAP_MOD1_SUBVEC_MIN1_MAX023 6
#define SFPSWAP_MOD1_SUBVEC_MIN2_MAX013 7
#define SFPSWAP_MOD1_SUBVEC_MIN3_MAX012 8

static uint32_t Fp32LessTotalOrder(uint32_t lhs, uint32_t rhs) {
  lhs ^= (uint32_t)((int32_t)lhs >> 30) >> 1;
  rhs ^= (uint32_t)((int32_t)rhs >> 30) >> 1;
  return (int32_t)lhs < (int32_t)rhs;
}

static void Exec_SFPSWAP(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);

  static const uint32_t is_minmax_mod1[] = {
    0, // Plain swap
    0xffffffffu, // Always MinMax
    0x0000ffffu, // MinMax for first 16 lanes, MaxMin for last 16
    0x00ff00ffu,
    0xff0000ffu, // MinMax for first and last 8 lanes, MaxMin for middle 16
    0x000000ffu, // MinMax for first 8 lanes, MaxMin for others
    0x0000ff00u,
    0x00ff0000u,
    0xff000000u
  };
  if (mod1 > (sizeof(is_minmax_mod1) / sizeof(*is_minmax_mod1))) {
    return Exec_Undefined(env, insn);
  }
  uint32_t is_minmax = is_minmax_mod1[mod1];

  VReg result_vd;
  VReg result_vc;
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t val_d = env->vreg[vd].u[i];
    uint32_t val_c = env->vreg[vc].u[i];
    if (mod1 && Fp32LessTotalOrder(val_d, val_c) == ((is_minmax >> i) & 1u)) {
      // Already in correct order, no need to swap.
      result_vd.u[i] = val_d;
      result_vc.u[i] = val_c;
    } else {
      // Do the swap.
      result_vd.u[i] = val_c;
      result_vc.u[i] = val_d;
    }
  }
  WriteVReg(env, insn, vd, &result_vd);
  WriteVReg(env, insn, vc, &result_vc);
}

// --- SFPSHFT2 ---

#define SFPSHFT2_OPCODE 0x94

#define SFPSHFT2_MOD1_COPY4                     0
#define SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4      1
#define SFPSHFT2_MOD1_SUBVEC_SHFLROR1_AND_COPY4 2
#define SFPSHFT2_MOD1_SUBVEC_SHFLROR1           3
#define SFPSHFT2_MOD1_SUBVEC_SHFLSHR1           4
#define SFPSHFT2_MOD1_SHFT_IMM                  5
#define SFPSHFT2_MOD1_SHFT_LREG                 6

static void SubvecShflRor1(Env* env, VReg* result, const VReg* input) {
  for (uint32_t i = 0; i < 32; ++i) {
    // This isn't meant to assign to shfl_bug, but it does.
    result->u[i] = (i & 7) ? input->u[i - 1] : (env->shfl_bug[i / 8] = input->u[i + 7]);
  }
}

static void SubvecShflShr1(Env* env, VReg* result, const VReg* input) {
  for (uint32_t i = 0; i < 32; ++i) {
    // This is meant to have 0 instead of shfl_bug.
    result->u[i] = (i & 7) ? input->u[i - 1] : env->shfl_bug[i / 8];
  }
}

static void Exec_SFPSHFT2(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);
  uint32_t vc = insn_bits(8, 4);
  uint32_t vb = insn_bits(12, 4); // This overlaps with imm12!
  int32_t imm12 = insn_bits_signed(12, 12);

  VReg result;
  switch (mod1) {
  case SFPSHFT2_MOD1_COPY4:
  case SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4:
  case SFPSHFT2_MOD1_SUBVEC_SHFLROR1_AND_COPY4:
    memset(&result, 0, sizeof(result));
    if (mod1 == SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4) {
      memcpy(&result, &env->vreg[0].u[8], sizeof(uint32_t) * 24);
    } else if (mod1 == SFPSHFT2_MOD1_SUBVEC_SHFLROR1_AND_COPY4) {
      SubvecShflRor1(env, &result, &env->vreg[vc]);
    }
    WriteVReg(env, insn, 0, &env->vreg[1]);
    WriteVReg(env, insn, 1, &env->vreg[2]);
    WriteVReg(env, insn, 2, &env->vreg[3]);
    vd = 3;
    break;
  case SFPSHFT2_MOD1_SUBVEC_SHFLROR1:
    SubvecShflRor1(env, &result, &env->vreg[vc]);
    break;
  case SFPSHFT2_MOD1_SUBVEC_SHFLSHR1:
    SubvecShflShr1(env, &result, &env->vreg[vc]);
    break;
  case SFPSHFT2_MOD1_SHFT_IMM:
    for (uint32_t i = 0; i < 32; ++i) {
      result.u[i] = Shift(env->vreg[vb].u[i], env->vreg[vc].u[i]);
    }
    break;
  case SFPSHFT2_MOD1_SHFT_LREG:
    for (uint32_t i = 0; i < 32; ++i) {
      result.u[i] = Shift(env->vreg[vb].u[i], imm12); // vb overlaps with imm12!
    }
    break;
  default:
    return Exec_Undefined(env, insn);
  }

  WriteVReg(env, insn, vd, &result);
}

// --- SFPLUTFP32 ---

#define SFPLUTFP32_OPCODE 0x95

#define SFPLUTFP32_MOD1_FP32_3ENTRY_TABLE  0
#define SFPLUTFP32_MOD1_FP16_3ENTRY_TABLE 10
#define SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE1 2
#define SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE2 3

#define SFPLUTFP32_MOD1_SGN_UPDATE 0
#define SFPLUTFP32_MOD1_SGN_RETAIN 4

#define SFPLUTFP32_MOD1_INDIRECT_VD 8 // Overlaps with SFPLUTFP32_MOD1_FP16_3ENTRY_TABLE!

static uint32_t HalfToFloat_LUT(uint16_t x) {
  if ((x & 0x7c00u) == 0x7c00u) {
    // Exponent of all ones treated as zero, rather than as NaN/Inf.
    return 0;
  }
  return HalfToFloat_FastAndLoose(x);
}

static void Exec_SFPLUTFP32(Env* env, uint32_t insn) {
  uint32_t mod1 = insn_bits(0, 4);
  uint32_t vd = insn_bits(4, 4);

  uint32_t table; // Nibbles: >4, 3-4, 2-3, 1.5-2, 1-1.5, 0.75-1, 0.5-0.75, <0.5
  switch (mod1 &~ SFPLUTFP32_MOD1_SGN_RETAIN) {
  case SFPLUTFP32_MOD1_FP32_3ENTRY_TABLE:
  case SFPLUTFP32_MOD1_FP16_3ENTRY_TABLE:
    table = 0x22211000;
    break;
  case SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE1:
    table = 0xAA291880;
    break;
  case SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE2:
    table = 0xA2291880;
    break;
  default:
    return Exec_Undefined(env, insn);
  }

  VReg result;
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t d = env->vreg[3].u[i];
    int32_t em = ((d >> 22) & 0x1ff) - 251;
    if (em <= 0) em = table & 0xf;
    else if (em >= 7) em = table >> 28;
    else em = (table >> (em * 4)) & 0xf;

    uint32_t mul, add;
    switch (mod1 &~ SFPLUTFP32_MOD1_SGN_RETAIN) {
    case SFPLUTFP32_MOD1_FP32_3ENTRY_TABLE:
      mul = env->vreg[em].u[i];
      add = env->vreg[4 + em].u[i];
      break;
    case SFPLUTFP32_MOD1_FP16_3ENTRY_TABLE: {
      uint32_t tmp = env->vreg[em].u[i];
      mul = HalfToFloat_LUT(tmp >> 16);
      add = HalfToFloat_LUT(tmp & 0xffff);
      break; }
    case SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE1:
    case SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE2: {
      uint32_t shift = (em * 2) & 16;
      em &= 3;
      mul = HalfToFloat_LUT(env->vreg[em].u[i] >> shift);
      add = HalfToFloat_LUT(env->vreg[4 + em].u[i] >> shift);
      break; }
    }
    uint32_t val = FusedMulAdd(mul, d & 0x7fffffffu, add);
    if (mod1 & SFPLUTFP32_MOD1_SGN_RETAIN) {
      val = (val & 0x7fffffffu) + (d & 0x80000000u);
    }
    result.u[i] = val;
  }

  if (mod1 & SFPLUTFP32_MOD1_INDIRECT_VD) {
    WriteVRegIndirect(env, insn, &result);
  } else {
    WriteVReg(env, insn, vd, &result);
  }
}

// --- Public functions ---

void Init(Env* env) {
  memset(env, 0, sizeof(*env));

  // Initialise hardware-provided constants.
  // 8, 9, 10 are fixed constants
  const uint32_t constants[] = {
    0x3F56594B,  //   
    0x00000000,  // 0.0
    0x3F800000}; // 1.0
  for (uint32_t k = 0; k < 3; ++k) {
    for (uint32_t i = 0; i < 32; ++i) {
      env->vreg[8 + k].u[i] = constants[k];
    }
  }
  // 11, 12, 13, 14 are programmable constants
  // 15 is lane_id << 1
  for (uint32_t i = 0; i < 32; ++i) {
    env->vreg[15].u[i] = i << 1;
  }
}

void Exec(Env* env, uint32_t insn) {
  uint32_t opcode = insn >> 24;
  switch (opcode) {
  case SFPLOAD_OPCODE: Exec_SFPLOAD(env, insn); break;
  case SFPLOADI_OPCODE: Exec_SFPLOADI(env, insn); break;
  case SFPSTORE_OPCODE: Exec_SFPSTORE(env, insn); break;
  case SFPLUT_OPCODE: Exec_SFPLUT(env, insn); break;
  case SFPMULI_OPCODE: Exec_SFPMULI(env, insn); break;
  case SFPADDI_OPCODE: Exec_SFPADDI(env, insn); break;
  case SFPDIVP2_OPCODE: Exec_SFPDIVP2(env, insn); break;
  case SFPEXEXP_OPCODE: Exec_SFPEXEXP(env, insn); break;
  case SFPEXMAN_OPCODE: Exec_SFPEXMAN(env, insn); break;
  case SFPIADD_OPCODE: Exec_SFPIADD(env, insn); break;
  case SFPSHFT_OPCODE: Exec_SFPSHFT(env, insn); break;
  case SFPSETCC_OPCODE: Exec_SFPSETCC(env, insn); break;
  case SFPMOV_OPCODE: Exec_SFPMOV(env, insn); break;
  case SFPABS_OPCODE: Exec_SFPABS(env, insn); break;
  case SFPAND_OPCODE: Exec_SFPAND(env, insn); break;
  case SFPOR_OPCODE: Exec_SFPOR(env, insn); break;
  case SFPNOT_OPCODE: Exec_SFPNOT(env, insn); break;
  case SFPLZ_OPCODE: Exec_SFPLZ(env, insn); break;
  case SFPSETEXP_OPCODE: Exec_SFPSETEXP(env, insn); break;
  case SFPSETMAN_OPCODE: Exec_SFPSETMAN(env, insn); break;
  case SFPMAD_OPCODE: Exec_SFPMAD(env, insn); break;
  case SFPADD_OPCODE: Exec_SFPMAD(env, insn); break;
  case SFPMUL_OPCODE: Exec_SFPMAD(env, insn); break;
  case SFPPUSHC_OPCODE: Exec_SFPPUSHC(env, insn); break;
  case SFPPOPC_OPCODE: Exec_SFPPOPC(env, insn); break;
  case SFPSETSGN_OPCODE: Exec_SFPSETSGN(env, insn); break;
  case SFPENCC_OPCODE: Exec_SFPENCC(env, insn); break;
  case SFPCOMPC_OPCODE: Exec_SFPCOMPC(env, insn); break;
  case SFPTRANSP_OPCODE: Exec_SFPTRANSP(env, insn); break;
  case SFPXOR_OPCODE: Exec_SFPXOR(env, insn); break;
  case SFPNOP_OPCODE: break;
  case SFPSTOCHRND_OPCODE: Exec_SFPSTOCHRND(env, insn); break;
  case SFPCAST_OPCODE: Exec_SFPCAST(env, insn); break;
  case SFPSWAP_OPCODE: Exec_SFPSWAP(env, insn); break;
  case SFPSHFT2_OPCODE: Exec_SFPSHFT2(env, insn); break;
  case SFPLUTFP32_OPCODE: Exec_SFPLUTFP32(env, insn); break;
  }
}
