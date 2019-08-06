add_llvm_executable(mlir-vulkan-runner
  mlir-vulkan-runner.cpp
)
llvm_update_compile_flags(mlir-vulkan-runner)
whole_archive_link(mlir-vulkan-runner
  MLIRAffineOps
  MLIRControlFlowToCFG
  MLIRGPU
  MLIRLLVMIR
  MLIRStandardOps
  MLIRSPIRV
  MLIRStandardToLLVM
  MLIRTargetLLVMIR
  MLIRTransforms
  MLIRTranslation
)

#TODO Update it
set(VULKAN_PATH /home/khalikov/vulkan/1.1.101.0/x86_64/lib)
find_library(VULKAN_LIB vulkan HINTS ${VULKAN_PATH})

target_link_libraries(mlir-vulkan-runner PRIVATE
 MLIRIR
 MLIRParser
 MLIREDSC
 MLIRAnalysis
 MLIRExecutionEngine
 MLIRJitRunner
 MLIRSupport
 LLVMCore
 LLVMSupport
 ${VULKAN_LIB}
)
