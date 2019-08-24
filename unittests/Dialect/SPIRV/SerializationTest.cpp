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

#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "gmock/gmock.h"

#include <string>
using namespace mlir;

// SPIR-V dialect registration.
static mlir::DialectRegistration<mlir::spirv::SPIRVDialect> spirvDialect;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class SerializationTest : public ::testing::Test {
protected:
  SerializationTest() { createModuleOp(); }

  void createModuleOp() {
    Builder builder(&context);
    OperationState state(UnknownLoc::get(&context),
                         spirv::ModuleOp::getOperationName());
    state.addAttribute("major_version", builder.getI32IntegerAttr(1));
    state.addAttribute("minor_version", builder.getI32IntegerAttr(0));
    state.addAttribute("addressing_model", builder.getI32IntegerAttr(0));
    state.addAttribute("memory_model", builder.getI32IntegerAttr(1));
    spirv::ModuleOp::build(&builder, &state);
    module = cast<spirv::ModuleOp>(Operation::create(state));
  }

  OpBuilder initOpBuilder() {
    return OpBuilder(module.getOperation()->getRegion(0));
  }

  void addGlobalVar(OpBuilder &opBuilder, const std::string &varName, Type type,
                    spirv::StorageClass storageClass) {
    // Wrap into spv.ptr type.
    auto ptrType = spirv::PointerType::get(type, storageClass);
    opBuilder.create<spirv::GlobalVariableOp>(
        UnknownLoc::get(&context), opBuilder.getTypeAttr(ptrType),
        opBuilder.getStringAttr(varName), nullptr);
  }

  // !spv.struct<!spv.array<k x type> [0]>
  Type createStorageBufferType(Type elementType, uint32_t count) {
    auto arrayType = spirv::ArrayType::get(elementType, count);
    llvm::SmallVector<Type, 0> elementTypes{arrayType};
    llvm::SmallVector<spirv::StructType::LayoutInfo, 0> layoutInfo{0};
    return spirv::StructType::get(elementTypes, layoutInfo);
  }

  bool processHeader() {
    if (binary.size() < spirv::kHeaderWordCount) {
      return false;
    }
    if (binary[0] != spirv::kMagicNumber) {
      return false;
    }
    currOffset = spirv::kHeaderWordCount;
    return true;
  }

  bool processModule() {
    if (!processHeader()) {
      return false;
    }

    auto binarySize = binary.size();
    assert(currOffset < binarySize && "invalid binary size");

    while (currOffset < binarySize) {
      auto wordCount = (binary[currOffset] >> 16);
      assert(wordCount && "word count cannot be zero");

      uint32_t nextOffset = currOffset + wordCount;
      spirv::Opcode opcode =
          static_cast<spirv::Opcode>(binary[currOffset] & 0xffff);
      switch (opcode) {
      case spirv::Opcode::OpDecorate:
        ++decorationMap[static_cast<spirv::Decoration>(binary[currOffset + 2])];
        break;
      case spirv::Opcode::OpVariable:
        ++opcodeMap[spirv::Opcode::OpVariable];
        break;
      default:
        // TODO(denis0x0D): collect other data.
        break;
      }
      currOffset = nextOffset;
      assert(currOffset <= binarySize && "invalid binary size");
    }
    return true;
  }

protected:
  MLIRContext context;
  spirv::ModuleOp module;
  SmallVector<uint32_t, 0> binary;
  uint32_t currOffset = 0;
  llvm::DenseMap<spirv::Decoration, uint32_t> decorationMap;
  llvm::DenseMap<spirv::Opcode, uint32_t> opcodeMap;
};

//===----------------------------------------------------------------------===//
// StorageBuffer decoration
//===----------------------------------------------------------------------===//

TEST_F(SerializationTest, BlockDecorationTest) {
  auto opBuilder = initOpBuilder();
  // !spv.struct<!spv.array<16 x i32>> type.
  auto structArrayIntType =
      createStorageBufferType(opBuilder.getIntegerType(32), 16);
  // !spv.struct<!spv.array<16 x !spv.struct<!spv.array<16 x f32>>>> type.
  auto structArrayFloatType =
      createStorageBufferType(opBuilder.getF32Type(), 16);
  auto nestedStructType = createStorageBufferType(structArrayFloatType, 16);
  // !spv.array<16 x i32> type.
  auto ArrayIntType = spirv::ArrayType::get(opBuilder.getIntegerType(32), 16);
  // i32 type.
  auto IntType = opBuilder.getIntegerType(32);
  // Add globals.
  addGlobalVar(opBuilder, "var0", structArrayIntType,
               spirv::StorageClass::Uniform);
  addGlobalVar(opBuilder, "var1", nestedStructType,
               spirv::StorageClass::StorageBuffer);
  addGlobalVar(opBuilder, "var2", ArrayIntType, spirv::StorageClass::Uniform);
  addGlobalVar(opBuilder, "var3", IntType, spirv::StorageClass::StorageBuffer);
  addGlobalVar(opBuilder, "var4", structArrayIntType,
               spirv::StorageClass::Input);
  // Serialize module.
  ASSERT_FALSE(failed(spirv::serialize(module, binary)));
  ASSERT_TRUE(processModule());
  // Top-level struct with storage class Uniform, StorageBuffer: var0, var1.
  ASSERT_EQ(decorationMap[spirv::Decoration::Block], static_cast<uint32_t>(2));
  // Total amount of globals.
  ASSERT_EQ(opcodeMap[spirv::Opcode::OpVariable], static_cast<uint32_t>(5));
}
