// RUN: mlir-tblgen -gen-reference-implementations -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def X_Dialect : Dialect {
  let name = "x";
}
class X_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<X_Dialect, mnemonic, traits>;

def X_AddOp : X_Op<"add">,
    Arguments<(ins AnyTensor:$A, AnyTensor:$B)>,
    Results<(outs AnyTensor: $C)> {
    // TODO: extract referenceImplementation to Op.
  code referenceImplementation = [{
    auto ivs = IndexedLinalgValuemakeIndexHandles(view_A.rank());
    auto pivs = IndexedLinalgValuemakeIndexHandlePointers(ivs);
    IndexedValue A(arg_A), B(arg_B), C(arg_C);
    AffineLoopNestBuilder(pivs, view_A.getLbs(), view_A.getUbs(), view_A.getSteps())({
      C(ivs) = A(ivs) + B(ivs)
    });
  }];
}

// CHECK: printRefImplementation
