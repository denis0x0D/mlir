//===- DecorateWithLayoutInfoSPIRVPass.cpp - Decorate with layout info ----===//
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
// This file implements a pass to decorate SPIR-V types with layout info.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class SPIRVLayoutInfoUtils {
public:
  /// Return a new spirv::StructType with layout info, increment an offset by
  /// proccessed type size in bytes.
  static Type decorateType(spirv::StructType structType,
                           spirv::StructType::LayoutInfo &offset);
  /// Check that type is legal in terms of SPIR-V layout info decoration.
  static bool isLegalType(Type type);

private:
  static Type decorateType(Type type, spirv::StructType::LayoutInfo &offset);
  static Type decorateType(spirv::CompositeType compositeType,
                           spirv::StructType::LayoutInfo &offset);
  static Type decorateType(spirv::PointerType pointerType,
                           spirv::StructType::LayoutInfo &offset);
  /// Create an array or a vector type with layout info, based on composite
  /// type kind.
  static Type getArrayOrVectorType(spirv::CompositeType compositeType,
                                   Type elementType, uint64_t arrayStride = 0);
  /// Calculate the size in bytes of the primitive type.
  static unsigned getPrimitiveTypeSizeInBytes(Type primitiveType);
  /// Check that type is primitive in terms of SPIR-V dialect.
  static bool isPrimitiveType(Type type);
};

Type SPIRVLayoutInfoUtils::decorateType(spirv::StructType structType,
                                        spirv::StructType::LayoutInfo &offset) {
  if (!structType.getNumElements()) {
    return structType;
  }
  llvm::SmallVector<Type, 4> memberTypes;
  llvm::SmallVector<spirv::StructType::LayoutInfo, 4> layoutInfo;
  spirv::StructType::LayoutInfo structOffset = 0;

  auto fieldOffset = structOffset;
  for (uint32_t i = 0, e = structType.getNumElements(); i < e; ++i) {
    // Process type and increment the struct's offset.
    auto memberType = SPIRVLayoutInfoUtils::decorateType(
        structType.getElementType(i), structOffset);

    memberTypes.push_back(memberType);
    layoutInfo.push_back(fieldOffset);
    fieldOffset = structOffset;
  }

  offset += structOffset;
  return spirv::StructType::get(memberTypes, layoutInfo);
}

Type SPIRVLayoutInfoUtils::decorateType(Type type,
                                  spirv::StructType::LayoutInfo &offset) {
  if (SPIRVLayoutInfoUtils::isPrimitiveType(type)) {
    offset += SPIRVLayoutInfoUtils::getPrimitiveTypeSizeInBytes(type);
    return type;
  }
  switch (type.getKind()) {
  case spirv::TypeKind::Struct:
    return SPIRVLayoutInfoUtils::decorateType(type.cast<spirv::StructType>(),
                                              offset);
  case spirv::TypeKind::Array:
  case StandardTypes::Vector:
    return SPIRVLayoutInfoUtils::decorateType(type.cast<spirv::CompositeType>(),
                                              offset);
  case spirv::TypeKind::Pointer:
    return SPIRVLayoutInfoUtils::decorateType(type.cast<spirv::PointerType>(),
                                              offset);
  default:
    llvm_unreachable("unhandled SPIR-V type");
  }
}

Type SPIRVLayoutInfoUtils::decorateType(spirv::PointerType pointerType,
                                  spirv::StructType::LayoutInfo &offset) {
  uint64_t currentOffset = 0;
  auto pointeeType = SPIRVLayoutInfoUtils::decorateType(
      pointerType.getPointeeType(), currentOffset);
  // According to the SPIR-V spec the size of the pointer depends on the
  // addressing model. So, it should be save to increment an offset for the
  // pointer by 8 bytes, because 8 bytes is the upper bound limit for the
  // pointer size.
  offset += 8;
  return spirv::PointerType::get(pointeeType, pointerType.getStorageClass());
}

Type SPIRVLayoutInfoUtils::decorateType(spirv::CompositeType compositeType,
                                  spirv::StructType::LayoutInfo &offset) {
  if (!compositeType.getNumElements()) {
    return compositeType;
  }
  uint64_t elementSizeInBytes = 0;
  auto memberType = SPIRVLayoutInfoUtils::decorateType(
      compositeType.getElementType(0), elementSizeInBytes);
  offset += elementSizeInBytes * compositeType.getNumElements();
  return SPIRVLayoutInfoUtils::getArrayOrVectorType(compositeType, memberType,
                                                    elementSizeInBytes);
}

Type SPIRVLayoutInfoUtils::getArrayOrVectorType(
    spirv::CompositeType compositeType, Type elementType,
    uint64_t arrayStride) {
  switch (compositeType.getKind()) {
  case spirv::TypeKind::Array:
    return spirv::ArrayType::get(elementType, compositeType.getNumElements(),
                                 arrayStride);
  case StandardTypes::Vector:
    return VectorType::get(compositeType.getNumElements(), elementType);
  default:
    llvm_unreachable("unhandled SPIR-V composite type");
  }
}

unsigned SPIRVLayoutInfoUtils::getPrimitiveTypeSizeInBytes(Type primitiveType) {
  // According to the SPIR-V spec:
  // "The ArrayStride, MatrixStride, and Offset decorations must be large
  // enough to hold the size of the objects they affect (that is, specifying
  // overlap is invalid)."
  auto bitWidth = primitiveType.getIntOrFloatBitWidth();
  if (bitWidth == 1 || bitWidth == 8 || bitWidth == 16)
    return 4;
  return bitWidth / 8;
}

bool SPIRVLayoutInfoUtils::isLegalType(Type type) {
  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType) {
    return true;
  }
  auto storageClass = ptrType.getStorageClass();
  auto structType = ptrType.getPointeeType().dyn_cast<spirv::StructType>();
  if (!structType) {
    return true;
  }
  if (storageClass != spirv::StorageClass::Uniform &&
           storageClass != spirv::StorageClass::StorageBuffer &&
           storageClass != spirv::StorageClass::PushConstant) {
    return true;
  }
  return structType.hasLayout();
}

bool SPIRVLayoutInfoUtils::isPrimitiveType(Type primitiveType) {
  if (auto integerType = primitiveType.dyn_cast<IntegerType>()) {
    return llvm::is_contained(llvm::ArrayRef<unsigned>({1, 8, 16, 32, 64}),
                              integerType.getWidth());
  }
  return primitiveType.isF32() || primitiveType.isF64();
}

namespace {
class SPIRVGlobalVariableOpLayoutInfoDecoration
    : public OpRewritePattern<spirv::GlobalVariableOp> {
public:
  using OpRewritePattern<spirv::GlobalVariableOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(spirv::GlobalVariableOp op,
                                     PatternRewriter &rewriter) const override {
    // Initial offset is zero.
    spirv::StructType::LayoutInfo baseOffset = 0;
    auto ptrType = op.type().cast<spirv::PointerType>();
    auto structType = SPIRVLayoutInfoUtils::decorateType(
        ptrType.getPointeeType().cast<spirv::StructType>(), baseOffset);
    auto decoratedType =
        spirv::PointerType::get(structType, ptrType.getStorageClass());
    // Save all named attributes except "type" attribute.
    SmallVector<NamedAttribute, 4> globalVarAttrs;

    for (const auto &attr : op.getAttrs()) {
      if (attr.first == "type") {
        continue;
      }
      globalVarAttrs.push_back(attr);
    }

    rewriter.replaceOpWithNewOp<spirv::GlobalVariableOp>(
        op, rewriter.getTypeAttr(decoratedType), globalVarAttrs);
    return matchSuccess();
  }
};

class SPIRVAddressOfOpLayoutInfoDecoration
    : public OpRewritePattern<spirv::AddressOfOp> {
public:
  using OpRewritePattern<spirv::AddressOfOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(spirv::AddressOfOp op,
                                     PatternRewriter &rewriter) const override {
    auto spirvModule = op.getParentOfType<spirv::ModuleOp>();
    if (!spirvModule) {
      return matchFailure();
    }
    auto varName = op.variable();
    auto varOp = spirvModule.lookupSymbol<spirv::GlobalVariableOp>(varName);
    if (!varOp) {
      return matchFailure();
    }

    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(
        op, varOp.type(), rewriter.getSymbolRefAttr(varName));
    return matchSuccess();
  }
};
} // namespace

void populateSPIRVLayoutInfoPatterns(OwningRewritePatternList &patterns,
                                     MLIRContext *ctx) {
  patterns.insert<SPIRVGlobalVariableOpLayoutInfoDecoration,
                  SPIRVAddressOfOpLayoutInfoDecoration>(ctx);
}

namespace {
class DecorateWithLayoutInfoSPIRVPass
    : public ModulePass<DecorateWithLayoutInfoSPIRVPass> {
private:
  void runOnModule() override;
};

void DecorateWithLayoutInfoSPIRVPass::runOnModule() {
  auto module = getModule();
  OwningRewritePatternList patterns;
  populateSPIRVLayoutInfoPatterns(patterns, module.getContext());
  ConversionTarget target(*(module.getContext()));
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addLegalOp<FuncOp>();
  // Type is illegal in terms of legalization if it's a
  // !spv.ptr<!spv.struct<...>> type in storage class
  // StorageBuffer/Uniform/PushConstant and does not have layout info.
  target.addDynamicallyLegalOp<spirv::GlobalVariableOp>(
      [](spirv::GlobalVariableOp op) {
        return SPIRVLayoutInfoUtils::isLegalType(op.type());
      });
  // Change the type for the direct users.
  target.addDynamicallyLegalOp<spirv::AddressOfOp>([](spirv::AddressOfOp op) {
    return SPIRVLayoutInfoUtils::isLegalType(op.pointer()->getType());
  });
  // TODO: Change the type for the indirect users such as spv.Load, spv.Store,
  // spv.FunctionCall and so on.
  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (failed(applyPartialConversion(spirvModule, target, patterns))) {
      signalPassFailure();
    }
  }
}
} // namespace

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::spirv::createDecorateWithLayoutInfoSPIRVPass() {
  return std::make_unique<DecorateWithLayoutInfoSPIRVPass>();
}

static PassRegistration<DecorateWithLayoutInfoSPIRVPass>
    pass("decorate-with-layoutinfo-spirv",
         "Decorate SPIR-V types with layout info");
