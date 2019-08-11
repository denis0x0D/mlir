set(LLVM_OPTIONAL_SOURCES
  mlir-vulkan-runner.cpp
  mlir-vulkan-runtime.cpp
  )

if(MLIR_VULKAN_RUNNER_ENABLED)
add_llvm_executable(mlir-vulkan-runner
  mlir-vulkan-runner.cpp
)

add_llvm_library(mlir-vulkan-runtime SHARED
  mlir-vulkan-runtime.cpp
  #TODO: move header.
  mlir-vulkan-runtime.h
  )

target_link_libraries(mlir-vulkan-runtime
  MLIRSPIRVSerialization
  LLVMCore
  LLVMSupport
  ${VULKAN_LIB}
  )

llvm_update_compile_flags(mlir-vulkan-runner)
whole_archive_link(mlir-vulkan-runner
  MLIRStandardOps
  MLIRSPIRV
)

#TODO Check not null
set(VULKAN_PATH $ENV{VULKAN_SDK}/lib)
find_library(VULKAN_LIB vulkan HINTS ${VULKAN_PATH})

target_link_libraries(mlir-vulkan-runner PRIVATE
 MLIRSPIRVSerialization
 mlir-vulkan-runtime
 ${VULKAN_LIB}
)
endif()
