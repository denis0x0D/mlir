//===-- SPIRVDerivedOps.td - MLIR SPIR-V ops spec ------*- tablegen -*-----===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file contains definitions for ops derived from base classes. It
// helps to represent ops which share the same TableGen definition, but have
// different opcodes.
//
//===----------------------------------------------------------------------===//

#ifdef SPIRV_DERIVED_OPS
#else
#define SPIRV_DERIVED_OPS

#ifdef SPIRV_BASE
#else
include "mlir/Dialect/SPIRV/SPIRVBase.td"
#endif // SPIRV_BASE

// -----

def SPV_FAddOp : SPV_ArithmeticOp<"FAdd", SPV_Float, [Commutative]> {
  let summary = "Floating-point addition of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fadd-op ::= ssa-id `=` `spv.FAdd` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.FAdd %0, %1 : f32
    %5 = spv.FAdd %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FDivOp : SPV_ArithmeticOp<"FDiv", SPV_Float> {
  let summary = "Floating-point division of Operand 1 divided by Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.
    ### Custom assembly form
    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fdiv-op ::= ssa-id `=` `spv.FDiv` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.FDiv %0, %1 : f32
    %5 = spv.FDiv %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FModOp : SPV_ArithmeticOp<"FMod", SPV_Float> {
  let summary = [{
    The floating-point remainder whose sign matches the sign of Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 2.

    ### Custom assembly form
    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmod-op ::= ssa-id `=` `spv.FMod` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.FMod %0, %1 : f32
    %5 = spv.FMod %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FMulOp : SPV_ArithmeticOp<"FMul", SPV_Float, [Commutative]> {
  let summary = "Floating-point multiplication of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.

    ### Custom assembly form

    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmul-op ::= `spv.FMul` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.FMul %0, %1 : f32
    %5 = spv.FMul %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FRemOp : SPV_ArithmeticOp<"FRem", SPV_Float> {
  let summary = [{
    The floating-point remainder whose sign matches the sign of Operand 1.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 1.

    ### Custom assembly form
    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    frem-op ::= ssa-id `=` `spv.FRemOp` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.FRemOp %0, %1 : f32
    %5 = spv.FRemOp %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FSubOp : SPV_ArithmeticOp<"FSub", SPV_Float> {
  let summary = "Floating-point subtraction of Operand 2 from Operand 1.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fsub-op ::= ssa-id `=` `spv.FRemOp` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.FRemOp %0, %1 : f32
    %5 = spv.FRemOp %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_IAddOp : SPV_ArithmeticOp<"IAdd", SPV_Integer, [Commutative]> {
  let summary = "Integer addition of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

    The resulting value will equal the low-order N bits of the correct
    result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    iadd-op ::= ssa-id `=` `spv.IAdd` ssa-use, ssa-use
                          `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.IAdd %0, %1 : i32
    %5 = spv.IAdd %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_IEqualOp : SPV_LogicalOp<"IEqual", SPV_Integer, [Commutative]> {
  let summary = "Integer comparison for equality.";

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    iequal-op ::= ssa-id `=` `spv.IEqual` ssa-use, ssa-use
                             `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.IEqual %0, %1 : i32
    %5 = spv.IEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_INotEqualOp : SPV_LogicalOp<"INotEqual", SPV_Integer, [Commutative]> {
  let summary = "Integer comparison for inequality.";

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    inot-equal-op ::= ssa-id `=` `spv.INotEqual` ssa-use, ssa-use
                                 `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.INotEqual %0, %1 : i32
    %5 = spv.INotEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_IMulOp : SPV_ArithmeticOp<"IMul", SPV_Integer, [Commutative]> {
  let summary = "Integer multiplication of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

    The resulting value will equal the low-order N bits of the correct
    result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    imul-op ::= ssa-id `=` `spv.IMul` ssa-use, ssa-use
                          `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.IMul %0, %1 : i32
    %5 = spv.IMul %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_ISubOp : SPV_ArithmeticOp<"ISub", SPV_Integer> {
  let summary = "Integer subtraction of Operand 2 from Operand 1.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

    The resulting value will equal the low-order N bits of the correct
    result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    isub-op ::= `spv.ISub` ssa-use, ssa-use
                          `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.ISub %0, %1 : i32
    %5 = spv.ISub %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SDivOp : SPV_ArithmeticOp<"SDiv", SPV_Integer> {
  let summary = "Signed-integer division of Operand 1 divided by Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sdiv-op ::= ssa-id `=` `spv.SDiv` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.SDiv %0, %1 : i32
    %5 = spv.SDiv %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SGreaterThanOp : SPV_LogicalOp<"SGreaterThan", SPV_Integer, []> {
  let summary = [{
    Signed-integer comparison if Operand 1 is greater than  Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sgreater-than-op ::= ssa-id `=` `spv.SGreaterThan` ssa-use, ssa-use
                                    `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SGreaterThan %0, %1 : i32
    %5 = spv.SGreaterThan %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SGreaterThanEqualOp : SPV_LogicalOp<"SGreaterThanEqual", SPV_Integer, []> {
  let summary = [{
    Signed-integer comparison if Operand 1 is greater than or equal to
    Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sgreater-than-equal-op ::= ssa-id `=` `spv.SGreaterThanEqual` ssa-use, ssa-use
                                          `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SGreaterThanEqual %0, %1 : i32
    %5 = spv.SGreaterThanEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SLessThanOp : SPV_LogicalOp<"SLessThan", SPV_Integer, []> {
  let summary = [{
    Signed-integer comparison if Operand 1 is less than Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sless-than-op ::= ssa-id `=` `spv.SLessThan` ssa-use, ssa-use
                                 `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SLessThan %0, %1 : i32
    %5 = spv.SLessThan %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SLessThanEqualOp : SPV_LogicalOp<"SLessThanEqual", SPV_Integer, []> {
  let summary = [{
    Signed-integer comparison if Operand 1 is less than or equal to Operand
    2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sless-than-equal-op ::= ssa-id `=` `spv.SLessThanEqual` ssa-use, ssa-use
                                       `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SLessThanEqual %0, %1 : i32
    %5 = spv.SLessThanEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SModOp : SPV_ArithmeticOp<"SMod", SPV_Integer> {
  let summary = [{
    Signed remainder operation for the remainder whose sign matches the sign
    of Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 2.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    smod-op ::= ssa-id `=` `spv.SMod` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SMod %0, %1 : i32
    %5 = spv.SMod %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SRemOp : SPV_ArithmeticOp<"SRem", SPV_Integer> {
  let summary = [{
    Signed remainder operation for the remainder whose sign matches the sign
    of Operand 1.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 1.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    srem-op ::= ssa-id `=` `spv.SRem` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SRem %0, %1 : i32
    %5 = spv.SRem %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_UDivOp : SPV_ArithmeticOp<"UDiv", SPV_Integer> {
  let summary = "Unsigned-integer division of Operand 1 divided by Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type, whose Signedness
    operand is 0.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    udiv-op ::= ssa-id `=` `spv.UDiv` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.UDiv %0, %1 : i32
    %5 = spv.UDiv %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_UGreaterThanOp : SPV_LogicalOp<"UGreaterThan", SPV_Integer, []> {
  let summary = [{
    Unsigned-integer comparison if Operand 1 is greater than  Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    ugreater-than-op ::= ssa-id `=` `spv.UGreaterThan` ssa-use, ssa-use
                                    `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.UGreaterhan %0, %1 : i32
    %5 = spv.UGreaterThan %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_UGreaterThanEqualOp
    : SPV_LogicalOp<"UGreaterThanEqual", SPV_Integer, []> {
  let summary = [{
    Unsigned-integer comparison if Operand 1 is greater than or equal to
    Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    ugreater-than-equal-op ::= ssa-id `=` `spv.UGreaterThanEqual` ssa-use, ssa-use
                                          `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.UGreaterThanEqual %0, %1 : i32
    %5 = spv.UGreaterThanEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_ULessThanOp : SPV_LogicalOp<"ULessThan", SPV_Integer, []> {
  let summary = [{
    Unsigned-integer comparison if Operand 1 is less than Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    uless-than-op ::= ssa-id `=` `spv.ULessThan` ssa-use, ssa-use
                                 `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.ULessThan %0, %1 : i32
    %5 = spv.ULessThan %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_ULessThanEqualOp : SPV_LogicalOp<"ULessThanEqual", SPV_Integer, []> {
  let summary = [{
    Unsigned-integer comparison if Operand 1 is less than or equal to
    Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    uless-than-equal-op ::= ssa-id `=` `spv.ULessThanEqual` ssa-use, ssa-use
                                       `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.ULessThanEqual %0, %1 : i32
    %5 = spv.ULessThanEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_UModOp : SPV_ArithmeticOp<"UMod", SPV_Integer> {
  let summary = "Unsigned modulo operation of Operand 1 modulo Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type, whose Signedness
    operand is 0.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    umod-op ::= ssa-id `=` `spv.UMod` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.UMod %0, %1 : i32
    %5 = spv.UMod %2, %3 : vector<4xi32>

    ```
  }];
}



#endif
