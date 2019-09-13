// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.FunctionCall
//===----------------------------------------------------------------------===//

func @fmain(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> () {
  // CHECK: {{%.*}} = spv.FunctionCall @f_add_vec({{%.*}}, {{%.*}}) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %0 = spv.FunctionCall @f_add_vec(%arg0, %arg1) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

  // CHECK: spv.FunctionCall @f_void({{%.*}}, {{%.*}}) : (vector<4xf32>, vector<4xf32>) -> ()
  spv.FunctionCall @f_void(%arg0, %arg1) : (vector<4xf32>, vector<4xf32>) ->  ()

  // CHECK: spv.FunctionCall @fno_op() : () -> ()
  spv.FunctionCall @fno_op() : () -> ()
  spv.Return
}

func @f_struct(%arg0 : !spv.struct<!spv.array<4xf32>>) -> (!spv.struct<!spv.array<4xf32>>) {
  // CHEKC: {{%.*}} = spv.FunctionCall @f_struct({{%.*}}) : (!spv.struct<!spv.array<4xf32>>) -> !spv.struct<!spv.array<4xf32>>
  %0 = spv.FunctionCall @f_struct(%arg0) : (!spv.struct<!spv.array<4xf32>>) -> !spv.struct<!spv.array<4xf32>>
  spv.ReturnValue %0: !spv.struct<!spv.array<4xf32>>
}

func @f_add_vec(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> (vector<4xf32>) {
  %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
  spv.ReturnValue %0 : vector<4xf32>
}

func @fno_op() -> () {
  spv.Return
}

func @f_void(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> () {
  spv.Return
}

// -----

func @f_invalid_result_type(%arg0 : i32, %arg1 : i32) -> () {
  // expected-error @+1 {{custom op 'spv.FunctionCall' callee function must have 0 or 1 result, but provided 2}}
  %0 = spv.FunctionCall @f_invalid_result_type(%arg0, %arg1) : (i32, i32) -> (i32, i32)
  spv.Return
}
