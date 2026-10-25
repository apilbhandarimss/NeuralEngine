# Architecture decisions

This is a starting boundary map, not a claim that the engine is complete.

## Dependency direction

`nn` composes operations and tensors. `ops` defines mathematical behaviour.
`backend` provides execution and device-specific facilities. `core` holds shared
low-level types. `data` supplies inputs; `optim` updates trainable parameters.

Keep dependencies flowing toward stable low-level contracts. Avoid circular
dependencies between `tensor`, `ops`, and `nn`.

## Decisions still to make deliberately

1. Tensor layout: contiguous row-major first; define strides and views before adding them.
2. Numeric types: begin with `float`; decide how dtype support will work before expanding.
3. Errors: use exceptions for invalid shapes/indices in the initial CPU implementation.
4. Gradients: choose explicit backward methods or a recorded computation graph only
   after defining parameter and operation contracts.
5. Memory: start with RAII and standard containers; introduce custom allocation only
   after profiling demonstrates a need.
6. Backend: keep the initial implementation CPU-only and make kernels correct before
   adding accelerator support.
