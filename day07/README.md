# Day 07: warp divergence

- from first principles: the 32 threads of a warp share one instruction stream, so at any moment the whole warp runs the *same* instruction
- if threads in the same warp take different sides of an `if`, the warp can't do both at once. It runs side A with the other threads switched off, then side B. Time = A + B. That's **warp divergence**
- if whole warps agree (split by `threadIdx.x / 32` instead of `threadIdx.x`), each warp runs only its own path and nothing is wasted

  ![divergence](../images/day07_divergence.png)

- not every `if` is bad:
  - `if (i < n)`: only the last warp is split, basically free
  - tiny branches (ReLU): the compiler computes both sides and picks one (a "select"), no jumping
  - big branches split per thread: these are the ones that hurt

  ![not every if is bad](../images/day07_not_every_if.png)

- [warp_divergence.cu](warp_divergence.cu): LeetGPU format (paste into the Vector Addition challenge, press Run). Times no branch vs branch-per-warp vs branch-per-thread, plus ReLU with `if` vs `fmaxf`
