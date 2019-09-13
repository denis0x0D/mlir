// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

spv.module "Logical" "VulkanKHR" {
  func @fmain(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> () {
    %0 = spv.FunctionCall @fadd(%arg0, %arg1) : (vector<4xf32>, vector<4xf32>) -> (vector<4xf32>)
    spv.FunctionCall @f_void(%arg0, %arg1) : (vector<4xf32>, vector<4xf32>) ->  ()
    spv.FunctionCall @fno_op() : () -> ()
    spv.Return
  }
  func @fadd(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> (vector<4xf32>) {
    // CHECK: {{%.*}} = spv.FAdd {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
    spv.ReturnValue %0 : vector<4xf32>
  }
  func @fno_op() -> () {
    spv.Return
  }
  func @f_void(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> () {
    // CHECK: {{%.*}} = spv.FAdd {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
}
