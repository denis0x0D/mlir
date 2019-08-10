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

#include "mlir-vulkan-runtime.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>

using namespace mlir;
using namespace llvm;
static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

extern LogicalResult
runOnModule(raw_ostream &os, ModuleOp module,
            llvm::DenseMap<Descriptor, VulkanBufferContent> &vars);

static void Print(float *result, int size) {
  std::cout << "buffer started with size" << size << std::endl;
  for (int i = 0; i < size / sizeof(float); ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << "buffer ended" << std::endl;
}

static void populateData(llvm::DenseMap<Descriptor, VulkanBufferContent> &vars,
                         int count) {
  for (int i = 0; i < count; ++i) {
    float *ptr = new float[4];
    for (int j = 0; j < 4; ++j) {
      ptr[j] = 1.001 + j;
    }
    VulkanBufferContent content;
    content.ptr = ptr;
    content.size = sizeof(float) * 4;
    vars.insert({i, content});
  }
}

static void
checkResults(llvm::DenseMap<Descriptor, VulkanBufferContent> &data) {
  for (auto temp: data) {
    Print(((float *)temp.second.ptr), temp.second.size);
  }
}

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
  populateData(bufferContents, 3);
  MLIRContext context;
  OwningModuleRef moduleRef(parseSourceFile(sourceMgr, &context));
  if (!moduleRef) {
    llvm::errs() << "can not open the file" << '\n';
    return 1;
  }

  if (failed(runOnModule(moduleRef.get(), bufferContents))) {
    llvm::errs() << "can't run on module" << '\n';
    return 1;
  }

  checkResults(bufferContents);
  return 0;
}
