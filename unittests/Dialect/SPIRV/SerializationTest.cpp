//===- SerializationTest.cpp - SPIR-V Seserialization Tests -------------===//
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
// The purpose of this file is to test a binary form of spir-v shader, generated
// by Serializer.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "gmock/gmock.h"

#include <memory>

using namespace mlir;

using ::testing::StrEq;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class SerializationTest : public ::testing::Test {
protected:
  SerializationTest() {}

protected:
  MLIRContext context;
  std::unique_ptr<Diagnostic> diagnostic;
};

//===----------------------------------------------------------------------===//
// Basics
//===----------------------------------------------------------------------===//

TEST_F(SerializationTest, ModuleTest) {}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

TEST_F(SerializationTest, TypeTest) {}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

TEST_F(SerializationTest, FunctionTest) {}

