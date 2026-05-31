module {
  func.func @elementwise(%input: memref<1x2x8xf32>, %output: memref<1x2x8xf32>) {
    affine.for %b = 0 to 1 {
      affine.for %s = 0 to 2 {
        affine.for %j = 0 to 8 {
          %value = affine.load %input[%b, %s, %j] : memref<1x2x8xf32>
          affine.store %value, %output[%b, %s, %j] : memref<1x2x8xf32>
        }
      }
    }
    return
  }
}
