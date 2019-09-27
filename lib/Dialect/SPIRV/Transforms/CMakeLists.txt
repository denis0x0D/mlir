add_llvm_library(MLIRSPIRVTransforms
  DecorateWithLayoutInfoSPIRVPass.cpp
  )

target_link_libraries(MLIRSPIRVTransforms
  MLIRPass
  MLIRSPIRV
  MLIRStandardOps
  )
