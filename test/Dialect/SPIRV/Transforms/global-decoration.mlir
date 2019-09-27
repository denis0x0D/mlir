// RUN: mlir-opt -decorate-with-layoutinfo-spirv %s -o - | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK: spv.globalVariable @var1 bind(0, 1) : !spv.ptr<!spv.struct<i32 [0], !spv.struct<f32 [0], i32 [4]> [4], f32 [12]>, Uniform>
  spv.globalVariable @var1 bind(0,1) : !spv.ptr<!spv.struct<i32, !spv.struct<f32, i32>, f32>, Uniform>

  // CHECK: spv.globalVariable @var2 bind(0, 2) : !spv.ptr<!spv.struct<!spv.array<64 x i32 [4]> [0], f32 [256]>, StorageBuffer>
  spv.globalVariable @var2 bind(0,2) : !spv.ptr<!spv.struct<!spv.array<64xi32>, f32>, StorageBuffer>

  // CHECK: spv.globalVariable @var3 bind(1, 0) : !spv.ptr<!spv.struct<!spv.struct<!spv.array<64 x i32 [4]> [0], f32 [256]> [0], i32 [260]>, StorageBuffer>
  spv.globalVariable @var3 bind(1,0) : !spv.ptr<!spv.struct<!spv.struct<!spv.array<64xi32>, f32>, i32>, StorageBuffer>

  // CHECK: spv.globalVariable @var4 : !spv.ptr<!spv.struct<!spv.array<16 x !spv.struct<f32 [0], f32 [4], !spv.array<16 x f32 [4]> [8]> [72]> [0], f32 [1152]>, StorageBuffer>
  spv.globalVariable @var4 : !spv.ptr<!spv.struct<!spv.array<16x!spv.struct<f32, f32, !spv.array<16xf32>>>, f32>, StorageBuffer>

  // CHECK: spv.globalVariable @var5 bind(1, 2) : !spv.ptr<!spv.struct<!spv.ptr<!spv.struct<i1 [0], i8 [4], i16 [8], i32 [12], i64 [16]>, StorageBuffer> [0], f32 [8], f64 [12]>, StorageBuffer>
  spv.globalVariable @var5 bind(1,2) : !spv.ptr<!spv.struct<!spv.ptr<!spv.struct<i1, i8, i16, i32, i64>, StorageBuffer>, f32, f64>, StorageBuffer>

  // CHECK: spv.globalVariable @var6 bind(1, 3) : !spv.ptr<!spv.struct<!spv.array<256 x f32 [4]> [0]>, StorageBuffer>
  spv.globalVariable @var6 bind(1,3) : !spv.ptr<!spv.struct<!spv.array<256xf32>>, StorageBuffer>

  // CHECK: spv.globalVariable @var7 bind(1, 4) : !spv.ptr<!spv.struct<vector<3xi32> [0], f32 [12]>, StorageBuffer>
  spv.globalVariable @var7 bind(1, 4) : !spv.ptr<!spv.struct<vector<3xi32>, f32>, StorageBuffer>

  // CHECK: spv.globalVariable @var8 : !spv.ptr<!spv.struct<!spv.struct<> [0]>, StorageBuffer>
  spv.globalVariable @var8 : !spv.ptr<!spv.struct<!spv.struct<>>, StorageBuffer>

  // CHECK: spv.globalVariable @arrayType : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, StorageBuffer>
  spv.globalVariable @arrayType : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, StorageBuffer>

  // CHECK: spv.globalVariable @InputStorage : !spv.ptr<!spv.struct<!spv.array<256 x f32>>, Input>
  spv.globalVariable @InputStorage : !spv.ptr<!spv.struct<!spv.array<256xf32>>, Input>

  // CHECK: spv.globalVariable @customLayout : !spv.ptr<!spv.struct<f32 [64], i32 [128]>, Uniform>
  spv.globalVariable @customLayout : !spv.ptr<!spv.struct<f32 [64], i32 [128]>, Uniform>

  // CHECK:  spv.globalVariable @emptyStruct : !spv.ptr<!spv.struct<>, Uniform>
  spv.globalVariable @emptyStruct : !spv.ptr<!spv.struct<>, Uniform>

  func @kernel() -> () {
    %c0 = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv._address_of @var1 : !spv.ptr<!spv.struct<i32 [0], !spv.struct<f32 [0], i32 [4]> [4], f32 [12]>, Uniform>
    %0 = spv._address_of @var1 : !spv.ptr<!spv.struct<i32, !spv.struct<f32, i32>, f32>, Uniform>
    // CHECK:  {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}] : !spv.ptr<!spv.struct<i32 [0], !spv.struct<f32 [0], i32 [4]> [4], f32 [12]>, Uniform>
    %1 = spv.AccessChain %0[%c0] : !spv.ptr<!spv.struct<i32, !spv.struct<f32, i32>, f32>, Uniform>
    spv.Return
  }
}
