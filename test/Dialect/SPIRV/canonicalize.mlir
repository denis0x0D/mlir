// RUN: mlir-opt %s -split-input-file -canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.AccsessChain
//===----------------------------------------------------------------------===//

func @access_chain_canonicalization_full_0() -> f32 {
  // CHECK: %[[INDEX_0:.*]] = spv.constant 0 : i32
  // CHECK-NEXT: %[[COMPOSITE:.*]] = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<4 x f32>>, !spv.array<4 x i32>>, Function>
  // CHECK-NEXT: %[[COMPONENT_0:.*]] = spv.AccessChain %[[COMPOSITE]][%[[INDEX_0]], %[[INDEX_0]], %[[INDEX_0]]] : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<4 x f32>>, !spv.array<4 x i32>>, Function>
  // CHECK-NEXT: %[[RESULT:.*]] = spv.Load "Function" %[[COMPONENT_0]] : f32
  // CHECK-NEXT: spv.ReturnValue %[[RESULT]] : f32
  %c0 = spv.constant 0: i32
  %0 = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %1 = spv.AccessChain %0[%c0] : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %2 = spv.AccessChain %1[%c0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %3 = spv.AccessChain %2[%c0] : !spv.ptr<!spv.array<4xf32>, Function>
  %5 = spv.Load "Function" %3 : f32
  spv.ReturnValue %5 : f32
}

// -----

func @access_chain_canonicalization_full_1() -> i32 {
  // CHECK: %[[INDEX_0:.*]] = spv.constant 0 : i32
  // CHECK-NEXT: %[[INDEX_1:.*]] = spv.constant 1 : i32
  // CHECK-NEXT: %[[COMPOSITE:.*]] = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4 x !spv.struct<!spv.array<4 x i32>>>>, Function>
  // CHECK-NEXT: %[[COMPONENT:.*]] = spv.AccessChain %[[COMPOSITE]][%[[INDEX_0]], %[[INDEX_1]], %[[INDEX_0]], %[[INDEX_1]]] : !spv.ptr<!spv.struct<!spv.array<4 x !spv.struct<!spv.array<4 x i32>>>>, Function>
  // CHECK-NEXT: %[[RESULT:.*]] = spv.Load "Function" %[[COMPONENT]] : i32
  // CHECK-NEXT: spv.ReturnValue %[[RESULT]] : i32
  %c0 = spv.constant 0: i32
  %c1 = spv.constant 1: i32
  %0 = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4x!spv.struct<!spv.array<4xi32>>>>, Function>
  %1 = spv.AccessChain %0[%c0, %c1] : !spv.ptr<!spv.struct<!spv.array<4x!spv.struct<!spv.array<4xi32>>>>, Function>
  %2 = spv.AccessChain %1[%c0, %c1] : !spv.ptr<!spv.struct<!spv.array<4xi32>>, Function>
  %3 = spv.Load "Function" %2 : i32
  spv.ReturnValue %3 : i32
}

// -----

// The result of the second `spv.AccessChain` operation has two uses and therefore cannot be combined with the third.
func @access_chain_canonicalization_partial() -> !spv.array<4xf32> {
  // CHECK: %[[INDEX_0:.*]] = spv.constant 0 : i32
  // CHECK-NEXT: %[[COMPOSITE:.*]] = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<4 x f32>>, !spv.array<4 x i32>>, Function>
  // CHECK-NEXT: %[[COMPONENT_0:.*]] = spv.AccessChain %[[COMPOSITE]][%[[INDEX_0]], %[[INDEX_0]]] : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<4 x f32>>, !spv.array<4 x i32>>, Function>
  // CHECK-NEXT: %[[COMPONENT_1:.*]] = spv.AccessChain %[[COMPONENT_0]][%[[INDEX_0]]] : !spv.ptr<!spv.array<4 x f32>, Function>
  // CHECK-NEXT: %[[RESULT:.*]] = spv.Load "Function" %[[COMPONENT_0]] : !spv.array<4 x f32>
  // CHECK-NEXT: %{{.*}} = spv.Load "Function" %[[COMPONENT_1]] : f32
  // CHECK-NEXT: spv.ReturnValue %[[RESULT]] : !spv.array<4 x f32>
  %c0 = spv.constant 0: i32
  %0 = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %1 = spv.AccessChain %0[%c0] : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %2 = spv.AccessChain %1[%c0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %3 = spv.AccessChain %2[%c0] : !spv.ptr<!spv.array<4xf32>, Function>
  %4 = spv.Load "Function" %2 : !spv.array<4xf32>
  %5 = spv.Load "Function" %3 : f32
  spv.ReturnValue %4: !spv.array<4xf32>
}

// -----

// Each result of the first two `spv.AccessChain` operations has more than one use.
func @cannot_canonicalize_access_chain_0() -> f32 {
  // CHECK: %[[INDEX_0:.*]] = spv.constant 0 : i32
  // CHECK-NEXT: %[[COMPOSITE:.*]] = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<4 x f32>>, !spv.array<4 x i32>>, Function>
  // CHECK-NEXT: %[[COMPONENT_0:.*]] = spv.AccessChain %[[COMPOSITE]][%[[INDEX_0]]] : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<4 x f32>>, !spv.array<4 x i32>>, Function>
  // CHECK-NEXT: %[[COMPONENT_1:.*]] = spv.AccessChain %[[COMPONENT_0]][%[[INDEX_0]]] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
  // CHECK-NEXT: %[[COMPONENT_2:.*]] = spv.AccessChain %[[COMPONENT_1]][%[[INDEX_0]]] : !spv.ptr<!spv.array<4 x f32>, Function>
  // CHECK-NEXT: {{%.*}} = spv.Load "Function" %[[COMPONENT_0]] : !spv.array<4 x !spv.array<4 x f32>>
  // CHECK-NEXT: {{%.*}} = spv.Load "Function" %[[COMPONENT_1]] : !spv.array<4 x f32>
  // CHECK-NEXT: %[[RESULT:.*]] = spv.Load "Function" %[[COMPONENT_2]] : f32
  // CHECK-NEXT: spv.ReturnValue %[[RESULT]] : f32
  %c0 = spv.constant 0: i32
  %0 = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %1 = spv.AccessChain %0[%c0] : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %2 = spv.AccessChain %1[%c0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %3 = spv.AccessChain %2[%c0] : !spv.ptr<!spv.array<4xf32>, Function>
  %4 = spv.Load "Function" %1 : !spv.array<4x!spv.array<4xf32>>
  %5 = spv.Load "Function" %2 : !spv.array<4xf32>
  %6 = spv.Load "Function" %3 : f32
  spv.ReturnValue %6: f32
}

// -----

// Not a chained accesses.
func @cannot_canonicalize_access_chain_1() -> !spv.array<4xi32> {
  // CHECK: %[[INDEX_1:.*]] = spv.constant 1 : i32
  // CHECK-NEXT: %[[COMPOSITE_0:.*]] = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<4 x f32>>, !spv.array<4 x i32>>, Function>
  // CHECK-NEXT: %[[COMPOSITE_1:.*]] = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<4 x f32>>, !spv.array<4 x i32>>, Function>
  // CHECK-NEXT: %[[COMPOSITE_0_COMPONENT_0:.*]] = spv.AccessChain %[[COMPOSITE_0]][%[[INDEX_1]]] : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<4 x f32>>, !spv.array<4 x i32>>, Function>
  // CHECK-NEXT: %[[COMPOSITE_1_COMPONENT_0:.*]] = spv.AccessChain %[[COMPOSITE_1]][%[[INDEX_1]]] : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<4 x f32>>, !spv.array<4 x i32>>, Function>
  // CHECK-NEXT: %[[RESULT_0:.*]] = spv.Load "Function" %[[COMPOSITE_0_COMPONENT_0]] : !spv.array<4 x i32>
  // CHECK-NEXT: %[[RESULT_1:.*]] = spv.Load "Function" %[[COMPOSITE_1_COMPONENT_0]] : !spv.array<4 x i32>
  // CHECK-NEXT: spv.ReturnValue %[[RESULT_0]] : !spv.array<4 x i32>
  %c1 = spv.constant 1: i32
  %0 = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %1 = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %2 = spv.AccessChain %0[%c1] : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %3 = spv.AccessChain %1[%c1] : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %4 = spv.Load "Function" %2 : !spv.array<4xi32>
  %5 = spv.Load "Function" %3 : !spv.array<4xi32>
  spv.ReturnValue %4 : !spv.array<4xi32>
}

// -----

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
  // CHECK: %[[DST_VAR:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<i32, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<i32, Function>

  // CHECK: %[[SRC_VALUE:.*]] = spv.Select {{%.*}}, %[[TRUE_VALUE]], %[[FALSE_VALUE]] : i1, i32
  // CHECK-NEXT: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE]] ["Aligned", 4] : i32
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
  // CHECK: %[[DST_VAR:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: %[[SRC_VALUE:.*]] = spv.Select {{%.*}}, %[[TRUE_VALUE]], %[[FALSE_VALUE]] : i1, vector<3xi32>
  // CHECK-NEXT: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE]] ["Aligned", 8] : vector<3xi32>
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

// -----

// Store to a different variables.
func @cannot_canonicalize_selection_op_0(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_0:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_1:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR_0:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>
  // CHECK: %[[DST_VAR_1:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %4 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: spv.selection {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spv.Store "Function" %[[DST_VAR_0]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spv.Branch ^merge

  ^else:
    // CHECK: spv.Store "Function" %[[DST_VAR_1]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %4, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

// A conditional block consists of more than 2 operations.
func @cannot_canonicalize_selection_op_1(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_0:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_1:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR_0:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>
  // CHECK: %[[DST_VAR_1:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %4 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: spv.selection {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spv.Store "Function" %[[DST_VAR_0]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %1 ["Aligned", 8] : vector<3xi32>
    // CHECK: spv.Store "Function" %[[DST_VAR_1]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %4, %1 ["Aligned", 8]:  vector<3xi32>
    spv.Branch ^merge

  ^else:
    // CHECK: spv.Store "Function" %[[DST_VAR_1]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %4, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

// A control-flow goes into `^then` block from `^else` block.
func @cannot_canonicalize_selection_op_2(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_0:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_1:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: spv.selection {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spv.Branch ^merge

  ^else:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^then

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

// `spv.Return` as a block terminator.
func @cannot_canonicalize_selection_op_3(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_0:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_1:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: spv.selection {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spv.Return

  ^else:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

// Different memory access attributes.
func @cannot_canonicalize_selection_op_4(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_0:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_1:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: spv.selection {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_0]] ["Aligned", 4] : vector<3xi32>
    spv.Store "Function" %3, %1 ["Aligned", 4]:  vector<3xi32>
    spv.Branch ^merge

  ^else:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}
