import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN


def text_to_speech(text, speech_fname):
    # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

    # Running the TTS
    mel_output, mel_length, alignment = tacotron2.encode_text(text)

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_output)

    # Save the waverform
    torchaudio.save(speech_fname, waveforms.squeeze(1), 22050)
    print('done!')

    # from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    # import torch
    # import soundfile as sf
    # from datasets import load_dataset

    # processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    # model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    # vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

    # # load xvector containing speaker's voice characteristics from a dataset
    # embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    # speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # sf.write("speech.wav", speech.numpy(), samplerate=16000)


def speech_to_text(speech_fname, timings_fname_prefix):
    # import whisper
    # model = whisper.load_model("medium.en")
    # stable_whisper gives better word timings
    import stable_whisper
    model = stable_whisper.load_model('base')
    result = model.transcribe(speech_fname)
    result.save_as_json(timings_fname_prefix + '.json')
    result.to_srt_vtt(timings_fname_prefix + '.srt')
    print(result)

if __name__ == '__main__':
    text = "I reached over and secretly undid my seatbelt, and when his foot hit the brake at the red light, I flung open the door and ran. I had no shoes on. I was crying."
    speech_fname = 'speech.wav'
    timings_fname_prefix = 'timings'
    text_to_speech(text, speech_fname=speech_fname)
    speech_to_text(speech_fname=speech_fname, timings_fname_prefix=timings_fname_prefix)