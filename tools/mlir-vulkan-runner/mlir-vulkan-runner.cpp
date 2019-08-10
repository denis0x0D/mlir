//===- mlir-vulkan-runner.cpp - MLIR Vulkan Execution Driver---------------===//
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
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StringExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <cstdint>
#include <iostream>
#include <vector>
#include <vulkan/vulkan.h>
#include <iostream>
#include <fstream>

using namespace mlir;
using namespace llvm;
using Descriptor = int32_t;

struct VulkanDeviceMemoryBuffer {
  VkBuffer buffer;
  VkDeviceMemory deviceMemory;
  int32_t descriptor{0};
};

struct VulkanBufferContent {
  void *ptr{nullptr};
  int64_t size{0};
};

struct VulkanMemoryContext {
  uint32_t queueFamilyIndex{0};
  uint32_t memoryTypeIndex{VK_MAX_MEMORY_TYPES};
  VkDeviceSize memorySize{0};
};

struct VulkanExecutionContext {
  struct LocalSize {
    int32_t x{1};
    int32_t y{1};
    int32_t z{1};
  } localSize;
  std::string entryPoint;
};

inline void emit_vulkan_error(const llvm::Twine &message, VkResult error) {
  llvm::errs()
      << message.concat(" failed with error code ").concat(llvm::Twine{error});
}

#define RETURN_ON_VULKAN_ERROR(result, msg)                                    \
  if ((result) != VK_SUCCESS) {                                                \
    emit_vulkan_error(msg, (result));                                          \
    return failure();                                                          \
  }

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

extern LogicalResult
runOnModule(raw_ostream &os, ModuleOp module,
            llvm::DenseMap<Descriptor, VulkanBufferContent> &vars);

extern void PopulateData(llvm::DenseMap<Descriptor, VulkanBufferContent> &vars,
                         int count);

int main(int argc, char **argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR Vulkan execution driver\n");
  std::string errorMessage;
  auto inputFile = openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto outputFile = openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), SMLoc());
  llvm::DenseMap<Descriptor, VulkanBufferContent> bufferContents;
  PopulateData(bufferContents, 3);
  MLIRContext context;
  OwningModuleRef moduleRef(parseSourceFile(sourceMgr, &context));
  if (!moduleRef) {
    llvm::errs() << "can not open the file" << '\n';
    return 1;
  }

  if (failed(runOnModule(outputFile->os(), moduleRef.get(), bufferContents))) {
    llvm::errs() << "can't run on module" << '\n';
    return 1;
  }
  return 0;
}
