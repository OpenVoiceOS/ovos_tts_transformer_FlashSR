from typing import Tuple, Dict, Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from ovos_plugin_manager.templates.transformers import TTSTransformer
from ovos_plugin_manager.templates.tts import TTS
from ovos_plugin_manager.utils.audio import AudioData, AudioFile


class FlashSRTTSTransformer(TTSTransformer):
    """ runs after TTS stage but before playback"""

    def __init__(self, name="ovos-tts-transformer-FlashSR", priority=50, config=None):
        super().__init__(name, priority, config)
        # Download the ONNX model from HF Hub
        model_path = hf_hub_download(repo_id="YatharthS/FlashSR", filename="model.onnx", subfolder="onnx")
        # Create ONNX session and run inference
        self.ort_session = ort.InferenceSession(model_path,
                                                providers= ['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def transform(self, wav_file: str, context: dict | None = None) -> Tuple[str, Dict[str, Any]]:
        """
        Optionally transform passed wav_file and return path to transformed file
        :param wav_file: path to wav file generated in TTS stage
        :returns: path to transformed wav file for playback
        """
        context = context or {}
        sr = context.get("sr", 16000)
        if sr == 48000:
            # skip - already hi-res
            return wav_file, context

        # Load audio file at 16kHz
        y = AudioData.from_file(wav_file)
        #y, sr = librosa.load(wav_file, sr=sr)
        lowres_wav = y.get_np_float32(convert_rate=16000)[np.newaxis, :]  # Add batch dimension

        onnx_output = self.ort_session.run(["reconstruction"],
                                           {"audio_values": lowres_wav})[0]

        # Save output audio at 48kHz
        outpath = wav_file.replace(".wav", "_sr.wav")
        data = AudioData.from_array(onnx_output.squeeze(0), sample_rate=48000, sample_width=2)
        with open(outpath, "wb") as f:
            f.write(data.get_wav_data())

        return outpath, context


if __name__ == "__main__":
    tx = FlashSRTTSTransformer()
    tx.transform("test.wav")
    # listen to test_sr.wav and compare