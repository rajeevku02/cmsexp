
import torch
torch.set_num_threads(1)
SAMPLING_RATE = 16000

from pprint import pprint

import websockets
import asyncio
from websockets.asyncio.server import serve

async def handler(websocket):
    while True:
        message = await websocket.recv()
        print(message)

async def main():
    async with serve(handler, "", 8001):
        await asyncio.get_running_loop().create_future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())


# USE_PIP = True # download model using pip package or torch.hub
# USE_ONNX = False # change this to True if you want to test onnx model
# # ONNX model supports opset_version 15 and 16 (default is 16). 
# # Pass argument opset_version to load_silero_vad (pip) or torch.hub.load (torchhub).
# # !!! ONNX model with opset_version=15 supports only 16000 sampling rate !!!
# 
# #torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')
# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                             model='silero_vad',
#                             force_reload=True,
#                             onnx=USE_ONNX,
#                             opset_version=16)
# 
# (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
# 
# wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
# # get speech timestamps from full audio file
# 
# #speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
# #pprint(speech_timestamps)
# #
# #save_audio('only_speech.wav', collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE) 
# #predicts = model.audio_forward(wav, sr=SAMPLING_RATE)
# #pprint(predicts)
# 
# #vad_iterator = VADIterator(model, sampling_rate=SAMPLING_RATE)
# #
# #window_size_samples = 512 if SAMPLING_RATE == 16000 else 256
# #for i in range(0, len(wav), window_size_samples):
# #    chunk = wav[i: i+ window_size_samples]
# #    if len(chunk) < window_size_samples:
# #      break
# #    speech_dict = vad_iterator(chunk, return_seconds=True)
# #    if speech_dict:
# #        print(speech_dict)
# #vad_iterator.reset_states() # reset model states after each audio
# #
# 
# speech_probs = []
# window_size_samples = 512 if SAMPLING_RATE == 16000 else 256
# for i in range(0, len(wav), window_size_samples):
#     chunk = wav[i: i+window_size_samples]
#     if len(chunk) < window_size_samples:
#         break
#     speech_prob = model(chunk, SAMPLING_RATE).item()
#     speech_probs.append(speech_prob)
# model.reset_states() # reset model states after each audio
# 
# print(speech_probs) # first 10 chunks predicts