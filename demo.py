import io

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class AudioWatermarker:
    def __init__(self):
        self.private_key, self.public_key = self.generate_key_pair()

    def generate_key_pair(self):
        """Generate RSA key pair for signing."""
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        return private_key, public_key

    def segment_and_sign_audio(self, audio_data, sr, segment_duration=1.0):
        """
        Transform audio into mel spectrograms, sign each segment, and reconstruct.
        """
        segment_length = int(segment_duration * sr)

        signed_segments = []
        segment_signatures = []

        for start in range(0, len(audio_data), segment_length):
            segment = audio_data[start : start + segment_length]

            # Compute mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)

            # Convert mel spectrogram to bytes for signing
            segment_bytes = mel_spectrogram.tobytes()

            # Sign the segment
            signature = self.private_key.sign(
                segment_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            segment_signatures.append(signature)
            signed_segments.append(segment)

        # Reconstruct full waveform
        signed_waveform = np.concatenate(signed_segments)

        return signed_waveform, segment_signatures

    def verify_audio_segments(
        self, signed_waveform, sr, segment_signatures, segment_duration=1.0
    ):
        """
        Verify the signatures of each audio segment.
        """
        segment_length = int(segment_duration * sr)
        verification_results = []

        for i, start in enumerate(range(0, len(signed_waveform), segment_length)):
            segment = signed_waveform[start : start + segment_length]
            mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)
            segment_bytes = mel_spectrogram.tobytes()

            try:
                self.public_key.verify(
                    segment_signatures[i],
                    segment_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )
                verification_results.append(True)
            except Exception:
                verification_results.append(False)

        return verification_results

    def visualize_mel_spectrogram(self, audio_data, sr):
        """
        Create a mel spectrogram visualization.
        """
        plt.figure(figsize=(10, 4))
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        librosa.display.specshow(
            librosa.power_to_db(mel_spec, ref=np.max),
            sr=sr,
            x_axis="time",
            y_axis="mel",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")

        # Save plot to a buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return buf


def create_gradio_interface():
    watermarker = AudioWatermarker()

    def process_audio(audio_input):
        # Read the audio file
        sr, audio_data = audio_input
        audio_data = audio_data.astype(np.float32)

        # Sign the audio
        signed_waveform, segment_signatures = watermarker.segment_and_sign_audio(
            audio_data, sr
        )

        # Verify the signatures
        verification_results = watermarker.verify_audio_segments(
            signed_waveform, sr, segment_signatures
        )

        # Visualize original and signed mel spectrograms
        orig_spectrogram = watermarker.visualize_mel_spectrogram(audio_data, sr)
        signed_spectrogram = watermarker.visualize_mel_spectrogram(signed_waveform, sr)

        # Prepare signed audio for output
        signed_audio_path = "signed_audio.wav"
        sf.write(signed_audio_path, signed_waveform, sr)

        return (
            signed_audio_path,
            str(verification_results),
            orig_spectrogram,
            signed_spectrogram,
        )

    # Create Gradio interface
    interface = gr.Interface(
        fn=process_audio,
        inputs=gr.Audio(type="numpy", label="Upload Audio"),
        outputs=[
            gr.Audio(label="Signed Audio"),
            gr.Textbox(label="Verification Results"),
            gr.Image(label="Original Mel Spectrogram"),
            gr.Image(label="Signed Mel Spectrogram"),
        ],
        title="Audio Watermarking Demonstration",
        description="Upload an audio file to sign and verify its integrity using mel spectrogram and cryptographic signatures.",
    )

    return interface


# Launch the Gradio app
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
