# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset
# import soundfile as sf


# # load model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")

# # load dummy dataset and read audio files
# # with open('filename.flac', 'rb') as f:
#     # data, samplerate = sf.read(f)
# sample, samplerate = sf.read("example_TTS.wav")
# # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# # sample = ds[0]["audio"]
# input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# # generate token ids
# predicted_ids = model.generate(input_features)
# # decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
# ['<|startoftranscript|><|notimestamps|> Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.<|endoftext|>']

# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# [' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']

import whisper
model = whisper.load_model("medium.en")
result = model.transcribe("example_TTS.wav")
print(result)