set(LLVM_OPTIONAL_SOURCES
  mlir-vulkan-runner.cpp
  mlir-vulkan-runtime.cpp
  )

add_llvm_executable(mlir-vulkan-runner
  mlir-vulkan-runner.cpp
)

add_llvm_library(mlir-vulkan-runtime SHARED
  mlir-vulkan-runtime.cpp
  )

target_link_libraries(mlir-vulkan-runtime
  MLIRSPIRVSerialization
  ${VULKAN_LIB}
  )

llvm_update_compile_flags(mlir-vulkan-runner)
whole_archive_link(mlir-vulkan-runner
  MLIRStandardOps
  MLIRSPIRV
)

#TODO Update it
set(VULKAN_PATH $ENV{VULKAN_SDK}/lib)
find_library(VULKAN_LIB vulkan HINTS ${VULKAN_PATH})

target_link_libraries(mlir-vulkan-runner PRIVATE
 MLIRSPIRVSerialization
 mlir-vulkan-runtime
 ${VULKAN_LIB}
)
