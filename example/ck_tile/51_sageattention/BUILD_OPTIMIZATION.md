# SageAttention - 如何减少编译时间

SageAttention目前链接`tile_fmha_fwd_instances`库，该库包含约1400个kernel实例。

## 快速减少编译时间的方法

修改 `example/ck_tile/01_fmha/CMakeLists.txt`：

### 1. 只启用基础 FWD API (第14行左右)
```cmake
# 原来
set(FMHA_FWD_ENABLE_APIS "fwd;fwd_appendkv;fwd_splitkv;fwd_pagedkv")

# 修改为
set(FMHA_FWD_ENABLE_APIS "fwd")
```

### 2. 只保留常用的 head dimensions (第50行左右)
```cmake
# 原来
set(FMHA_DEFAULT_OPTDIM "32;64;96;128;256")

# 修改为  
set(FMHA_DEFAULT_OPTDIM "64;128")
```

**效果**：编译实例从 ~1400 减少到 ~100，编译时间大幅减少！

## 说明

SageAttention复用了FMHA的kernel基础设施，创建完全独立的kernel instances需要：
- 重新实现mask、dropout、alibi等大量模板类
- 深入理解CK的内部kernel pipeline
- 维护成本很高

当前方案：**链接FMHA但通过配置大幅减少编译实例数**，是**最实用**的方案。

