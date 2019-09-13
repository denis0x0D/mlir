// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

spv.module "Logical" "GLSL450" {
   func @fmain(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> () {
    // CHECK: {{%.*}} = spv.FunctionCall @fadd({{%.*}} {{%.*}}) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %0 = spv.FunctionCall @fadd(%arg0, %arg1) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

    // CHECK: spv.FunctionCall @f_void({{%.*}} {{%.*}}) : (vector<4xf32>, vector<4xf32>) -> ()
    spv.FunctionCall @f_void(%arg0, %arg1) : (vector<4xf32>, vector<4xf32>) ->  ()

    // CHECK: spv.FunctionCall @fno_op() : () -> ()
    spv.FunctionCall @fno_op() : () -> ()
    spv.Return
  }
  func @fadd(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> (vector<4xf32>) {
    %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
    spv.ReturnValue %0 : vector<4xf32>
  }
  func @fno_op() -> () {
    spv.FunctionCall @fno_op() : () -> ()
    spv.Return
  }
  func @f_void(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> () {
    %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  func @loop(%count : i32) -> () {
    %zero = spv.constant 0: i32
    %one = spv.constant 1: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

    spv.loop {
      spv.Branch ^header
    ^header:
      %val0 = spv.Load "Function" %var : i32
      %cmp = spv.SLessThan %val0, %count : i32
      spv.BranchConditional %cmp, ^body, ^merge
    ^body:
      spv.Branch ^continue
    ^continue:
      %2 = spv.FunctionCall @f_inc(%zero) : (i32) -> i32
      %val1 = spv.Load "Function" %var : i32
      %add = spv.IAdd %val1, %one : i32
      spv.Store "Function" %var, %add : i32
      spv.Branch ^header
    ^merge:
      spv._merge
    }
    spv.Return
  }
  func @f_inc(%arg0 :i32) -> i32 {
    spv.ReturnValue %arg0 : i32
  }
}
