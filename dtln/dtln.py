import soundfile as sf
import numpy as np
import tensorflow as tf


class DTLNproc():
    """DTLNproc allows to clean noise and improve speech in audio record.
    """

    def __init__(self, model_path='./dtln/pretrained_model/'):
        """At initialization level DTLNproc loads model into memory.
        """
        #: model_path: path to the weights and config of DTLN model.
        self.model_path = model_path
        #: model: model loaded to the memory
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load model into the memory for further usage.
            
        :param model_path: path to the weights and config of DTLN model.
        :type model_path: str
        
        :return: Model object.
        """
        model = tf.saved_model.load(model_path)
        return model

    def process_record(self, record, fs):
        """Clean record from noise and improve speech quality.
            
        :param record: Record object for improvement.
        :type record: object
        :param fs: Sample rate of record.
        :type fs: int

        :return: Clean and improved record object.
        """
        ##########################
        # the values are fixed, if you need other values, you have to retrain.
        # The sampling rate of 16k is also fix.
        block_len = 512
        block_shift = 128
        infer = self.model.signatures["serving_default"]
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
        
    def process_audio(self, audio_path, output_path):
        """Load record from file and clean record from noise 
        and improve speech quality.
            
        :param audio_path: Path to the audio file.
        :type audio_path: str
        :param output_path: Path ot the processed and saved audio file.
        :type output_path: str
        """


        record, fs = sf.read(audio_path)

        out_file = self.process_record(record, fs)
        # write to .wav file 
        sf.write(output_path, out_file, fs)
