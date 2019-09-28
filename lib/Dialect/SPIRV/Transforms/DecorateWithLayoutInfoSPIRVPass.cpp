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
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// According to the SPIR-V spec:
/// "There are different alignment requirements depending on the specific
/// resources and on the features enabled on the device.
/// There are 3 types of alignment: scalar, base, extended.
///
/// Note: Even if scalar alignment is supported, it is generally more
/// performant to use the base alignment.
///
/// The memory layout must obey the following rules:
/// 1.The Offset decoration of any member must be a multiple of its alignment.
/// 2.Any ArrayStride or MatrixStride decoration must be a multiple of the
/// alignment of the array or matrix as defined above.
/// 3.The ArrayStride, MatrixStride, and Offset decorations must be large
/// enough to hold the size of the objects they affect (that is, specifying
/// overlap is invalid).
class SPIRVLayoutInfoUtils {
public:
  using Alignment = uint64_t;

  /// Return a new spirv::StructType with layout info, increment a base size by
  /// proccessed type size in bytes and populate an alignment.
  static Type decorateType(spirv::StructType structType,
                           spirv::StructType::LayoutInfo &baseSize,
                           Alignment &baseAlignment);
  /// Check that type is legal in terms of SPIR-V layout info decoration.
  static bool isLegalType(Type type);

private:
  static Type decorateType(Type type, spirv::StructType::LayoutInfo &baseSize,
                           Alignment &baseAlignment);
  static Type decorateType(VectorType compositeType,
                           spirv::StructType::LayoutInfo &baseSize,
                           Alignment &baseAligment);
  static Type decorateType(spirv::ArrayType compositeType,
                           spirv::StructType::LayoutInfo &baseSize,
                           Alignment &baseAligment);
  /// Calculate a scalar alignment.
  static Alignment getScalarTypeAlignment(Type scalarType);
};

Type SPIRVLayoutInfoUtils::decorateType(
    spirv::StructType structType, spirv::StructType::LayoutInfo &baseSize,
    SPIRVLayoutInfoUtils::Alignment &baseAlignment) {

  if (!structType.getNumElements()) {
    return structType;
  }

  llvm::SmallVector<Type, 4> memberTypes;
  llvm::SmallVector<spirv::StructType::LayoutInfo, 4> layoutInfo;
  llvm::SmallVector<spirv::StructType::MemberDecorationInfo, 4>
      memberDecorations;

  spirv::StructType::LayoutInfo structSize = 0;
  spirv::StructType::LayoutInfo memberOffset = 0;
  SPIRVLayoutInfoUtils::Alignment maxMemberAlignment = 1;
  SPIRVLayoutInfoUtils::Alignment memberAlignment = 1;

  for (uint32_t i = 0, e = structType.getNumElements(); i < e; ++i) {
    auto memberType = SPIRVLayoutInfoUtils::decorateType(
        structType.getElementType(i), structSize, memberAlignment);
    memberTypes.push_back(memberType);
    layoutInfo.push_back(memberOffset);
    memberOffset = structSize;
    // According to the SPIR-V spec:
    // "A structure has a base alignment equal to the largest base alignment of
    // any of its members."
    maxMemberAlignment = std::max(maxMemberAlignment, memberAlignment);
    memberAlignment = 1;
  }

  baseSize += llvm::alignTo(structSize, maxMemberAlignment);
  baseAlignment = std::max(baseAlignment, maxMemberAlignment);

  structType.getMemberDecorations(memberDecorations);
  return spirv::StructType::get(memberTypes, layoutInfo, memberDecorations);
}

Type SPIRVLayoutInfoUtils::decorateType(
    Type type, spirv::StructType::LayoutInfo &baseSize,
    SPIRVLayoutInfoUtils::Alignment &baseAlignment) {

  if (spirv::SPIRVDialect::isValidSPIRVScalarType(type)) {
    auto scalarTypeAlignment =
        SPIRVLayoutInfoUtils::getScalarTypeAlignment(type);
    // SPIR-V spec does not specify any padding for a scalar type.
    baseSize += scalarTypeAlignment;
    baseAlignment = std::max(baseAlignment, scalarTypeAlignment);
    return type;
  }

  switch (type.getKind()) {
  case spirv::TypeKind::Struct:
    return SPIRVLayoutInfoUtils::decorateType(type.cast<spirv::StructType>(),
                                              baseSize, baseAlignment);
  case spirv::TypeKind::Array:
    return SPIRVLayoutInfoUtils::decorateType(type.cast<spirv::ArrayType>(),
                                              baseSize, baseAlignment);
  case StandardTypes::Vector:
    return SPIRVLayoutInfoUtils::decorateType(type.cast<VectorType>(), baseSize,
                                              baseAlignment);
  default:
    llvm_unreachable("unhandled SPIR-V type");
  }
}

Type SPIRVLayoutInfoUtils::decorateType(
    VectorType vectorType, spirv::StructType::LayoutInfo &baseSize,
    SPIRVLayoutInfoUtils::Alignment &baseAlignment) {

  const auto numElements = vectorType.getNumElements();
  auto elementType = vectorType.getElementType();
  spirv::StructType::LayoutInfo elementSize = 0;
  SPIRVLayoutInfoUtils::Alignment elementAlignment = 1;

  auto memberType = SPIRVLayoutInfoUtils::decorateType(elementType, elementSize,
                                                       elementAlignment);
  // According to the SPIR-V spec:
  // 1."A two-component vector has a base alignment equal to twice its scalar
  // alignment."
  // 2."A three- or four-component vector has a base alignment equal to four
  // times its scalar alignment."
  auto vectorTypeAlignment =
      numElements == 2 ? elementAlignment * 2 : elementAlignment * 4;

  auto vectorSize = elementSize * numElements;
  baseSize += llvm::alignTo(vectorSize, vectorTypeAlignment);
  baseAlignment = std::max(baseAlignment, vectorTypeAlignment);

  return VectorType::get(numElements, memberType);
}

Type SPIRVLayoutInfoUtils::decorateType(
    spirv::ArrayType arrayType, spirv::StructType::LayoutInfo &baseSize,
    SPIRVLayoutInfoUtils::Alignment &baseAlignment) {

  const auto numElements = arrayType.getNumElements();
  auto elementType = arrayType.getElementType();
  spirv::ArrayType::LayoutInfo elementOffset = 0;
  SPIRVLayoutInfoUtils::Alignment elementAlignment = 1;

  auto memberType = SPIRVLayoutInfoUtils::decorateType(
      elementType, elementOffset, elementAlignment);
  // According to the SPIR-V spec:
  // "An array has a base alignment equal to the base alignment of its element
  // type."
  baseSize += elementOffset * numElements;
  baseAlignment = std::max(baseAlignment, elementAlignment);

  return spirv::ArrayType::get(memberType, numElements, elementOffset);
}

SPIRVLayoutInfoUtils::Alignment
SPIRVLayoutInfoUtils::getScalarTypeAlignment(Type scalarType) {
  // According to the SPIR-V spec:
  // 1."A scalar of size N has a scalar alignment of N."
  // 2."A scalar has a base alignment equal to its scalar alignment."
  auto bitWidth = scalarType.getIntOrFloatBitWidth();
  if (bitWidth == 1)
    return 1;
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

  // If struct type does not have layout info and struct type is empty, then
  // it is a legal type, because nothing to decorate here.
  return structType.hasLayout() || !structType.getNumElements();
}

namespace {
class SPIRVGlobalVariableOpLayoutInfoDecoration
    : public OpRewritePattern<spirv::GlobalVariableOp> {
public:
  using OpRewritePattern<spirv::GlobalVariableOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(spirv::GlobalVariableOp op,
                                     PatternRewriter &rewriter) const override {
    spirv::StructType::LayoutInfo baseSize = 0;
    SPIRVLayoutInfoUtils::Alignment baseAlignment = 1;
    SmallVector<NamedAttribute, 4> globalVarAttrs;

    auto ptrType = op.type().cast<spirv::PointerType>();
    auto structType = SPIRVLayoutInfoUtils::decorateType(
        ptrType.getPointeeType().cast<spirv::StructType>(), baseSize,
        baseAlignment);
    auto decoratedType =
        spirv::PointerType::get(structType, ptrType.getStorageClass());

    // Save all named attributes except "type" attribute.
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
    auto varName = op.variable();
    auto varOp = spirvModule.lookupSymbol<spirv::GlobalVariableOp>(varName);

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
  // Type is dynamically illegal in terms of legalization if:
  // 1.It's a !spv.ptr<!spv.struct<...>> type in storage class
  // StorageBuffer/Uniform/PushConstant.
  // 2.It does not have layout info and has at least one member.
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
    // The Logical addressing model means pointers are abstract, having no
    // physical size or numeric value. In this mode, pointers can only be
    // created from existing objects, and they cannot be stored into an object,
    // unless additional capabilities, e.g., VariablePointers, are declared to
    // add such functionality."
    if (spirvModule.addressing_model() != spirv::AddressingModel::Logical) {
      spirvModule.emitError(
          "Expected Logical addressing model for 'spv.module', but provided ")
          << spirv::stringifyAddressingModel(spirvModule.addressing_model());
      signalPassFailure();
    }
    if (failed(applyFullConversion(spirvModule, target, patterns))) {
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
