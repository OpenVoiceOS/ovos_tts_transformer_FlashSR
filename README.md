# OVOS FlashSR TTS Transformer

Audio super-resolution for OpenVoiceOS speech synthesis. `FlashSR` upsamples the
audio produced by any TTS plugin from 16 kHz to 48 kHz just before playback,
recovering high-frequency detail that low-sample-rate voices discard. The result
is brighter, less muffled speech without retraining or replacing your existing
voice.

This is a [TTS transformer](https://openvoiceos.github.io/ovos-technical-manual/tts_transformers/)
plugin (`opm.transformer.tts`): it runs after the TTS stage and before playback,
operating on the generated waveform rather than on text.

## How it works

- The generated `.wav` is loaded and resampled to 16 kHz mono.
- A [FlashSR](https://huggingface.co/YatharthS/FlashSR) ONNX model reconstructs a
  48 kHz waveform via [`onnxruntime`](https://onnxruntime.ai/), using the GPU
  (`CUDAExecutionProvider`) when available and falling back to CPU.
- The upsampled audio is written alongside the original and handed back for
  playback.
- If the incoming audio is already 48 kHz, the transform is skipped and the file
  is returned untouched.

The model is fetched from the Hugging Face Hub on first use and cached locally —
no manual download step is required.

## Installation

```bash
pip install ovos-tts-transformer-FlashSR
```

GPU acceleration additionally requires `onnxruntime-gpu` and a working CUDA
runtime; on CPU-only systems the default `onnxruntime` dependency is sufficient.

## Configuration

Enable the transformer in `mycroft.conf` under the `tts_transformers` section,
keyed by the plugin name:

```json
"tts_transformers": {
  "ovos-tts-transformer-FlashSR": {}
}
```

The plugin takes no configuration of its own; once enabled it applies to the
output of whichever TTS plugin is active. Multiple TTS transformers can be
chained — execution order follows each plugin's `priority` (FlashSR defaults to
`50`).

## Requirements

- `onnxruntime` (or `onnxruntime-gpu` for CUDA)
- `huggingface-hub`

## Related

- [`ovos-tts-transformer-NovaSR`](https://github.com/OpenVoiceOS/ovos_tts_transformer_NovaSR) —
  a Torch-based super-resolution transformer covering the same role.
- [`ovos-tts-transformer-sox-plugin`](https://github.com/OpenVoiceOS/ovos-tts-transformer-sox-plugin) —
  general-purpose audio effects (pitch, reverb, EQ, …) for TTS output.

---

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
