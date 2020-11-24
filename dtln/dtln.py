import soundfile as sf
import numpy as np
import tensorflow as tf


# interpreter_dir = 'DTLN/pretrained_model'
def load_interpreters(interpreter_dir):
    interpreter_1_path = f'{interpreter_dir}/model_1.tflite'
    interpreter_2_path = f'{interpreter_dir}/model_2.tflite'
    
    # load models
    interpreter_1 = tflite.Interpreter(model_path=interpreter_1_path)
    interpreter_1.allocate_tensors()
    interpreter_2 = tflite.Interpreter(model_path=interpreter_1_path)
    interpreter_2.allocate_tensors()
    return interpreter_1, interpreter_2


# model_path = './pretrained_model/dtln_saved_model'
def load_model(model_path):
    # load model
    model = tf.saved_model.load(model_path)
    return model


def process_record(model, record, fs):
    ##########################
    # the values are fixed, if you need other values, you have to retrain.
    # The sampling rate of 16k is also fix.
    block_len = 512
    block_shift = 128
    infer = model.signatures["serving_default"]
    # check for sampling rate
    if fs != 16000:
        raise ValueError('This model only supports 16k sampling rate.')
    # preallocate output audio
    out_file = np.zeros((len(record)))
    # create buffer
    in_buffer = np.zeros((block_len))
    out_buffer = np.zeros((block_len))
    # calculate number of blocks
    num_blocks = (record.shape[0] - (block_len-block_shift)) // block_shift
    # iterate over the number of blcoks        
    for idx in range(num_blocks):
        # shift values and write to buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = record[idx*block_shift:(idx*block_shift)+block_shift]
        # create a batch dimension of one
        in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
        # process one block
        out_block= infer(tf.constant(in_block))['conv1d_1']
        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer  += np.squeeze(out_block)
        # write block to output file
        out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
    return out_file
        

def process_audio(model, audio_path, output_path):
    
    record,fs = sf.read(audio_path)

    out_file = process_record(record, fs)
    # write to .wav file 
    sf.write(output_path, out_file, fs)

#     print('Processing finished.')
