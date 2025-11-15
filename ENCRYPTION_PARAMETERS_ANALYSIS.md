# CryptoFace Encryption Parameters and Bootstrapping Analysis

## 1. ENCRYPTION PARAMETERS

### 1.1 CKKS Scheme Parameters (from `infer_seal.cpp:560-596`)

#### Core Parameters
- **Encryption Scheme**: CKKS (Cheon-Kim-Kim-Song)
- **Polynomial Modulus Degree (N)**: `2^16 = 65,536`
  - Determines ring dimension and ciphertext size
  - Higher degree = more security + larger ciphertexts

- **Secret Key Hamming Weight**: `192`
  - Sparsity parameter for key generation
  - Controls number of non-zero coefficients in secret key

#### Modulus Chain Configuration
```
Total Levels: 30 (16 computation + 14 bootstrapping)
├── Root Prime (logq):          51 bits
├── Data Levels (logp):         46 bits × 16 levels  (for main computation)
├── Bootstrapping Levels:       51 bits × 14 levels  (for noise refresh)
└── Special Prime:              51 bits              (final precision)
```

**Coefficient Modulus Bit Sizes**:
- `logq = 51` bits (initial/bootstrapping)
- `logp = 46` bits (computation levels)
- `log_special_prime = 51` bits

**Modulus Chain Vector** (from line 580-584):
```cpp
coeff_bit_vec = [51, 46×16, 51×14, 51]
                 ↑    ↑      ↑      ↑
                 │    │      │      └─ Special prime
                 │    │      └─ Bootstrapping primes
                 │    └─ Computation primes
                 └─ Root prime
```

#### Scaling and Precision
- **Scaling Factor (scale)**: `2^46`
  - Used for fixed-point representation of real numbers
  - Determines precision of encrypted computations

- **Initial Scale**: `2^51` (after encryption)
- **Target Scale**: `2^46` (after first rescaling)

- **Integer Part Precision**:
  ```cpp
  log_integer_part = logq - logp - loge + 5
                   = 51 - 46 - 10 + 5
                   = 0
  ```

#### Slot Configuration
- **Full Slots (logn)**: `15` → 2^15 = 32,768 slots
- **Sparse Slots**:
  - `logn_1 = 14` → 16,384 slots (for layer set 1)
  - `logn_2 = 13` → 8,192 slots (for layer set 2)
  - `logn_3 = 12` → 4,096 slots (for layer set 3)

### 1.2 Bootstrapping-Specific Parameters (from `infer_seal.cpp:562-565`)

#### Core Bootstrapping Settings
- **Boundary K**: `25`
  - Defines modular reduction range for bootstrapping
  - Used in scaling ciphertexts back to valid range

- **Boot Degree (sin_cos_deg)**: `59`
  - Polynomial degree for cosine approximation in bootstrapping
  - Higher degree = better accuracy but more computation

- **Scale Factor**: `2`
  - Scaling factor used in bootstrapping operations

- **Inverse Degree**: `1`
  - Polynomial degree for arcsine approximation
  - Lower degree for faster computation

- **loge**: `10`
  - Logarithm of error tolerance parameter

#### Bootstrapping Polynomial Approximations

**Cosine Polynomial** (`cosine.txt`):
- **Degree**: 15
- **Regions**: 75 piecewise regions (0-75)
- Used in modular reduction for bootstrapping
- Approximates cos(x) for coefficient-to-slot transformation

**Arcsine Polynomial** (`inverse_sine.txt`):
- **Degree**: 7
- **Regions**: 31 piecewise regions (0-31)
- Used in inverse modular reduction
- Approximates arcsin(x) for slot-to-coefficient transformation

### 1.3 Rotation Keys Configuration (from `infer_seal.cpp:619-634`)

The system requires extensive rotation keys for:
1. **Bootstrapping rotations**: Powers of 2 (2^0 to 2^14)
2. **CNN-specific rotations**: 334 different rotation amounts

**Rotation Key Indices** (partial list):
```
{0,1,2,3,...,31,32,33,56,62,63,64,66,84,124,128,132,256,512,
 959,960,990,991,1008,1023,1024,...,32767}
```

These enable:
- Convolution operations (sliding window)
- Pooling operations (summation across regions)
- Linear transformations in bootstrapping (FFT/iFFT)

### 1.4 Neural Network-Specific Parameters

#### HerPN Activation (Hermite Polynomial Normal)
**Polynomial Form** (from `layers.py:154`):
```
HerPN(x) = w2·x² + w1·x + w0
```

**Mathematical Definition** (from `layers.py:132-133`):
```
out = x0/√(2π) + x1/2 + x2/√(4π)
where:
  x0 = BatchNorm(1)
  x1 = BatchNorm(x)
  x2 = BatchNorm((x²-1)/√2)
```

**Fused Form**:
- Batch normalization parameters are fused into polynomial coefficients
- Degree: 2 (quadratic polynomial)
- Enables FHE-friendly activation approximation

#### L2 Normalization Approximation (from `helper.py`)
**Polynomial Form**:
```
1/√x ≈ a·x² + b·x + c
```
- Used for final feature normalization
- Threshold values stored per fold in `threshold_<dataset>.txt`
- Polynomial coefficients estimated during training

### 1.5 Security Level
- **Estimated Security**: ~128-bit security
  - Based on polynomial modulus degree (2^16) and modulus sizes
  - Meets standard SEAL security parameters
  - Resistant to known lattice attacks

---

## 2. BOOTSTRAPPING PLACEMENT IN NETWORK LAYERS

### 2.1 Bootstrapping Location Strategy

Bootstrapping is **dynamically placed** based on configuration loaded from:
```cpp
// infer_seal.cpp:163-165
in.open(dir + "/boot_loc.txt");
for(long i=0; i<(layer_num-2) * 2; i++) {
    in >> val2;
    boot_loc.emplace_back(val2);
}
```

**File**: `boot_loc.txt` (per-model configuration)
- Contains binary flags (0 or 1) for each layer position
- Format: `[(layer_num-2) × 2]` values
- Two positions per layer:
  - `curr_boot`: before activation
  - `next_boot`: after activation

### 2.2 Bootstrapping Trigger Points (from `infer_seal.cpp:820-870`)

#### Layer Processing Loop Structure:
```
For each layer in {layer_1, layer_2, layer_3}:
    Read boot_loc flags: curr_boot, next_boot

    Case I: curr_boot == 1
        → ConvBN
        → Bootstrapping ← HERE
        → EvoReLU

    Case II: curr_boot == 0 && next_boot == 1
        → ConvBN
        → EvoReLU
        → Bootstrapping ← HERE

    Case III: curr_boot == 0 && next_boot == 0
        → ConvBN
        → EvoReLU
        (no bootstrapping)
```

### 2.3 Layer-Specific Bootstrapping Implementation

**Three Bootstrapper Instances** (from `infer_seal.cpp:614-616`):
```cpp
Bootstrapper bootstrapper_1(loge, logn_1, logN-1, total_level, scale, ...); // 2^14 slots
Bootstrapper bootstrapper_2(loge, logn_2, logN-1, total_level, scale, ...); // 2^13 slots
Bootstrapper bootstrapper_3(loge, logn_3, logN-1, total_level, scale, ...); // 2^12 slots
```

**Selection Logic** (from `infer_seal.cpp:841-843`):
```cpp
if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, ...);      // Layer 1_x (16 channels)
else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, ...); // Layer 2_x (32 channels)
else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, ...); // Layer 3_x (64 channels)
```

Different slot counts optimize for different channel dimensions.

### 2.4 Bootstrapping Algorithm Used

**Primary Method** (from `cnn_seal.cpp:339`):
```cpp
bootstrapper.bootstrap_real_3(rtn, ctxt);
```

**Algorithm**: `bootstrap_real_3`
- Optimized for real-valued ciphertexts
- Uses level-3 linear transformations (reduced depth)
- Variant optimized for sparse slot usage

**Full Process** (from `Bootstrapper.h`):
1. **Coefficient to Slot** (`coefftoslot_3`)
   - Transform polynomial coefficients to evaluation slots
   - Uses FFT-like linear transformation
   - Baby-Step Giant-Step (BSGS) optimization

2. **Modular Reduction** (via `ModularReducer`)
   - Polynomial approximation of sin/cos functions
   - Scales ciphertext modulus back to valid range
   - Uses precomputed Remez polynomials

3. **Slot to Coefficient** (`slottocoeff_3`)
   - Inverse FFT transformation
   - Returns refreshed ciphertext to coefficient representation

### 2.5 Pre-Bootstrapping Operations (from `infer_seal.cpp:835-840`)

**Before bootstrapping**, modulus switching is performed:
```cpp
cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
if (cur_level > 0) {
    ctxt = cnn.cipher();
    for (size_t i = 0; i < cur_level; i++)
        evaluator.mod_switch_to_next_inplace(ctxt);
    cnn.set_ciphertext(ctxt);
}
```

**Purpose**:
- Switch to lowest modulus level before bootstrapping
- Ensures bootstrapping starts from minimal noise level
- Maximizes remaining computation levels after refresh

### 2.6 Bootstrapping Preprocessing (from `infer_seal.cpp:636-662`)

**One-time Setup** (before inference):
1. **Generate Minimax Polynomials** (line 637-640):
   ```cpp
   bootstrapper_1.prepare_mod_polynomial();
   bootstrapper_2.prepare_mod_polynomial();
   bootstrapper_3.prepare_mod_polynomial();
   ```
   - Loads precomputed Remez coefficients from `cosine.txt` and `inverse_sine.txt`

2. **Generate Rotation Keys** (line 642-653):
   - Adds all required rotation indices for linear transformations
   - Creates Galois keys for all rotations (334+ keys)

3. **Generate Linear Transformation Coefficients** (line 659-662):
   ```cpp
   bootstrapper_1.generate_LT_coefficient_3();
   bootstrapper_2.generate_LT_coefficient_3();
   bootstrapper_3.generate_LT_coefficient_3();
   ```
   - Precomputes FFT/iFFT matrix coefficients
   - Stored as plaintext for repeated use

### 2.7 Bootstrapping Cost and Performance

**Level Consumption**:
- Bootstrapping uses **14 levels** from the 30-level budget
- Remaining **16 levels** available for CNN operations
- Each Conv-BN-ReLU block typically consumes 2-4 levels

**Observed Placement** (inferred from code structure):
```
Layer 0:        Conv-BN-ReLU (initial layer, no bootstrap)
Layer 1_x:      [Optional Bootstrap] → Conv-BN-ReLU → [Optional Bootstrap]
Layer 2_x:      [Optional Bootstrap] → Conv-BN-ReLU → [Optional Bootstrap]
Layer 3_x:      [Optional Bootstrap] → Conv-BN-ReLU → [Optional Bootstrap]
Final Layer:    BatchNorm1D → L2Norm (no bootstrap)
```

**Typical Pattern**:
- Bootstrapping placed **every 3-5 layers** or when noise accumulates
- Strategic placement minimizes total bootstrapping count
- Each bootstrap refreshes noise, allowing 16 more levels of computation

### 2.8 Layer Depth Information

**Depth Array** (from `infer_seal.cpp:160-161`):
```cpp
in.open(dir + "/depth.txt");
for(long i=0; i<layer_num-1; i++) {
    in >> val2;
    depth.emplace_back(val2);
}
```

Specifies ReLU polynomial depth for each layer:
- `depth = 0`: No ReLU (identity/linear)
- `depth = 2`: Low-degree polynomial ReLU approximation
- `depth > 2`: Higher-degree polynomial approximation

**Integration with Bootstrapping**:
- Higher depth ReLU consumes more levels
- Bootstrapping typically placed after high-depth ReLU operations
- Prevents noise overflow from deep polynomial evaluations

---

## 3. SUMMARY

### Encryption Parameter Hierarchy
```
CKKS Scheme
├── Ring: Z[X]/(X^65536 + 1)
├── Modulus Chain: [51, 46×16, 51×14, 51] bits
├── Security: ~128-bit
└── Precision: 46-bit fixed-point

Neural Network
├── Activation: Degree-2 HerPN polynomial
├── Normalization: Degree-2 L2norm polynomial
└── Depth Budget: 16 levels before refresh

Bootstrapping
├── Algorithm: bootstrap_real_3 (optimized sparse)
├── Cost: 14 levels
├── Polynomials: cos(59°) + arcsin(7°)
└── Placement: Dynamic per boot_loc.txt
```

### Key Findings

1. **Modular Depth Management**:
   - 30-level modulus chain split into 16 (computation) + 14 (bootstrap)
   - Enables multiple refresh cycles during inference

2. **Adaptive Slot Usage**:
   - Three bootstrappers with varying slot counts (2^12, 2^13, 2^14)
   - Optimized for different channel widths in CNN layers

3. **Strategic Bootstrap Placement**:
   - Configurable via `boot_loc.txt` for each model
   - Placed before/after activations based on noise accumulation
   - Minimizes total bootstrap count while preventing overflow

4. **Polynomial Approximations**:
   - All non-linear operations (activation, normalization) use degree-2 polynomials
   - Bootstrapping uses high-degree approximations (59 for cos, 7 for arcsin)
   - Precomputed Remez coefficients for optimal accuracy

5. **Performance Optimization**:
   - BSGS algorithm for linear transformations
   - Hoisting techniques for rotation optimization
   - Multiplexed parallel convolutions
   - Sparse slot encoding

---

## References

### Source Files
- **Encryption Setup**: `/home/user/CryptoFace/cnn_ckks/cpu-ckks/single-key/cnn/infer_seal.cpp:560-596`
- **Bootstrapping Core**: `/home/user/CryptoFace/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.h`
- **Bootstrapping Impl**: `/home/user/CryptoFace/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp`
- **Modular Reduction**: `/home/user/CryptoFace/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.h`
- **CNN Operations**: `/home/user/CryptoFace/cnn_ckks/cpu-ckks/single-key/cnn/cnn_seal.cpp:327-351`
- **HerPN Activation**: `/home/user/CryptoFace/models/layers.py:119-155`
- **Bootstrap Placement**: `/home/user/CryptoFace/cnn_ckks/cpu-ckks/single-key/cnn/infer_seal.cpp:820-870`

### Polynomial Coefficients
- **Cosine**: `/home/user/CryptoFace/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/cosine.txt`
- **Arcsine**: `/home/user/CryptoFace/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/inverse_sine.txt`
