// Shim for missing symbols in the prebuilt libLiteRtTopKWebGpuSampler.dylib.
// The prebuilt dylib exports Create/Destroy/SampleToIdAndScoreBuffer but is
// missing the 4 input-handling symbols added in a later API revision.
// Returning 0 from CanHandleInput / HandlesInput tells the engine the GPU
// sampler is NOT handling the input in-place, so it downloads logits to the
// CPU before calling the CPU sampler fallback — which can then read them.

int LiteRtTopKWebGpuSampler_UpdateConfig(
    void* sampler, const void* params, int batch_size,
    void* rand_gen, char** error_msg) {
  return 0;
}

int LiteRtTopKWebGpuSampler_CanHandleInput(void* sampler) {
  return 0;
}

int LiteRtTopKWebGpuSampler_HandlesInput(void* sampler) {
  return 0;
}

int LiteRtTopKWebGpuSampler_SetInputTensorsAndInferenceFunc(
    void* sampler, void* ids_tensor, void* prev_input_positions_tensor,
    void* input_positions_tensor, void* prev_mask_tensor, void* mask_tensor,
    int (*run_inference_func)(void*), void* arg, char** error_msg) {
  return 0;
}
