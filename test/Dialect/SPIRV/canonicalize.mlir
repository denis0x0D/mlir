// RUN: mlir-opt %s -split-input-file -canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.CompositeExtract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: extract_vector
func @extract_vector() -> (i32, i32, i32) {
  // CHECK: spv.constant 42 : i32
  // CHECK: spv.constant -33 : i32
  // CHECK: spv.constant 6 : i32
  %0 = spv.constant dense<[42, -33, 6]> : vector<3xi32>
  %1 = spv.CompositeExtract %0[0 : i32] : vector<3xi32>
  %2 = spv.CompositeExtract %0[1 : i32] : vector<3xi32>
  %3 = spv.CompositeExtract %0[2 : i32] : vector<3xi32>
  return %1, %2, %3 : i32, i32, i32
}

// -----

// CHECK-LABEL: extract_array_final
func @extract_array_final() -> (i32, i32) {
  // CHECK: spv.constant 4 : i32
  // CHECK: spv.constant -5 : i32
  %0 = spv.constant [dense<[4, -5]> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %1 = spv.CompositeExtract %0[0 : i32, 0 : i32] : !spv.array<1 x vector<2 x i32>>
  %2 = spv.CompositeExtract %0[0 : i32, 1 : i32] : !spv.array<1 x vector<2 x i32>>
  return %1, %2 : i32, i32
}

// -----

// CHECK-LABEL: extract_array_interm
func @extract_array_interm() -> (vector<2xi32>) {
  // CHECK: spv.constant dense<[4, -5]> : vector<2xi32>
  %0 = spv.constant [dense<[4, -5]> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %1 = spv.CompositeExtract %0[0 : i32] : !spv.array<1 x vector<2 x i32>>
  return %1 : vector<2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

// TODO(antiagainst): test constants in different blocks

func @deduplicate_scalar_constant() -> (i32, i32) {
  // CHECK: %[[CST:.*]] = spv.constant 42 : i32
  %0 = spv.constant 42 : i32
  %1 = spv.constant 42 : i32
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : i32, i32
}

// -----

func @deduplicate_vector_constant() -> (vector<3xi32>, vector<3xi32>) {
  // CHECK: %[[CST:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %0 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : vector<3xi32>, vector<3xi32>
}

// -----

func @deduplicate_composite_constant() -> (!spv.array<1 x vector<2xi32>>, !spv.array<1 x vector<2xi32>>) {
  // CHECK: %[[CST:.*]] = spv.constant [dense<5> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %0 = spv.constant [dense<5> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %1 = spv.constant [dense<5> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : !spv.array<1 x vector<2xi32>>, !spv.array<1 x vector<2xi32>>
}

// -----

//===----------------------------------------------------------------------===//
// spv.selection
//===----------------------------------------------------------------------===//

func @canonicalize_selection_op_scalar_type(%cond: i1) -> () {
  %0 = spv.constant 0: i32
  // CHECK: %[[TRUE_VALUE:.*]] = spv.constant 1 : i32
  %1 = spv.constant 1: i32
  // CHECK: %[[FALSE_VALUE:.*]] = spv.constant 2 : i32
  %2 = spv.constant 2: i32
  // CHECK: %[[DST_VALUE:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<i32, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<i32, Function>

  // CHECK: %[[SRC_VALUE:.*]] = spv.Select {{%.*}}, %[[TRUE_VALUE]], %[[FALSE_VALUE]] : i1, i32
  // CHECK-NEXT: spv.Store "Function" %[[DST_VALUE]], %[[SRC_VALUE]] ["Aligned", 4] : i32
  // CHECK-NEXT: spv.Return
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^else:
    spv.Store "Function" %3, %2 ["Aligned", 4]: i32
    spv.Branch ^merge

  ^then:
    spv.Store "Function" %3, %1 ["Aligned", 4]: i32
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

func @canonicalize_selection_op_vector_type(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[TRUE_VALUE:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[FALSE_VALUE:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VALUE:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: %[[SRC_VALUE:.*]] = spv.Select {{%.*}}, %[[TRUE_VALUE]], %[[FALSE_VALUE]] : i1, vector<3xi32>
  // CHECK-NEXT: spv.Store "Function" %[[DST_VALUE]], %[[SRC_VALUE]] ["Aligned", 8] : vector<3xi32>
  // CHECK-NEXT: spv.Return
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    spv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spv.Branch ^merge

  ^else:
    spv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}
