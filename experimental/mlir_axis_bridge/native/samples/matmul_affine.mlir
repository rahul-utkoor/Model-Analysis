module {
  func.func @matmul(%input: memref<1x2x8xf32>, %weight: memref<8x4xf32>, %output: memref<1x2x4xf32>) {
    affine.for %b = 0 to 1 {
      affine.for %s = 0 to 2 {
        affine.for %h = 0 to 4 {
          affine.for %j = 0 to 8 {
            %lhs = affine.load %input[%b, %s, %j] : memref<1x2x8xf32>
            %rhs = affine.load %weight[%j, %h] : memref<8x4xf32>
            %product = arith.mulf %lhs, %rhs : f32
            affine.store %product, %output[%b, %s, %h] : memref<1x2x4xf32>
          }
        }
      }
    }
    return
  }
}
