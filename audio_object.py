import math
import time

import numpy as np
import pyaudio
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample_poly


class Audio:
    '''
    A manager for recording and manipulating audio data.

    The main goal is to be able to easily record sound clips for use
    in doing Automatic Speech Recognition. 
    
    ---
    Priorities:
    1. Recording sound.
    2. Quickly analyzing the activity levels
    3. Being able to extract an active sound clip
    from a longer period of time are the priorities.

    Conceptually, there should only be 1 stream open at a time.
    The first thing that you will likely do after initializing this class
    is record some audio.
    '''

    def __init__(self, input_device_index: int=None):
        '''
        input_device_index: The index of the audio device to use for recording. See
            "list_recording_devices.py" to list all devices and pick index. If not set,
            the system's default recording device is used.
        '''

        self.input_device_index = input_device_index

        self.data = None

        self._pyaudio_obj = None
        self._stream = None

        self.rate_hz = None
        # self.channels = None

        self.length_s = None
        self.silence_threshold = None


    def record(
        self,
        time_s: float=3.0,
        rate_hz: int=16000,
        set_data: bool=True) -> np.array:
        '''
        Record an audio sample for X seconds.

        ---
        time_s: time to record sample for in seconds.
        rate_hz: record at this sample rate.
        set_data: whether or not to keep the data. Sometimes recording is for a temp calculation.
        '''

        self._open_stream(rate_hz=rate_hz)
        data = self._read_stream(read_time_s=time_s)
        
        if set_data:
            self.data = data
            self.rate_hz = rate_hz
            self.length_s = time_s
        
        self._close_stream()

        return data


    def record_activity(
        self,
        dwell_s: float=.1,
        silence_cutoff_s: float=.3,
        max_sample_length_s: float=3.0,
        max_run_time_s: float=None,
        rate_hz: int=16000) -> None:
        '''
        Opens an audio stream and tries to collect the next full sample of audio.
        Automatically cuts off the stream if too much time has passed and the sample 
        is too long. Limits the collected audio to the start and stop of activity. 
        This will keep running until activity is detected or max run time is reached.

        ---
        dwell_s: time to collect an individual sample from the stream.
        silence_cutoff_s: stop recording samples if this many of seconds of silence
            has elapsed since you began recording samples. 
        max_sample_length_s: stop recording samples if the cumulative samples is
            over this length. Samples that are too large can cause problems when passed
            to a transcription model.
        max_run_time_s: stop listening or recording if you've been listening for
            this total time. Defaults to None which will run until activity is detected.
        rate_hz: record at this sample rate.
        '''

        # Maximum number of silent dwells before ending collection.
        max_silent_dwells = int(np.round(silence_cutoff_s / dwell_s))

        audio_array = []  # Store recorded sample of audio here.
        num_silent_dwells = 0  # Number of silent dwells.
        is_recording = False

        self._open_stream(rate_hz=rate_hz)
        
        while True:

            sample = self._read_stream(read_time_s=dwell_s)

            rms_sample = self.calc_rms(audio_array=sample)

            if rms_sample > self.silence_threshold:  # Sample is active.
                if not is_recording:  # Start timer if new recording.
                    start_time = time.time()  # Start timer.
                
                is_recording = True

                audio_array.append(sample)
                
            else:  # Sample is NOT Active.
                if is_recording:
                    audio_array.append(sample)
                    num_silent_dwells += 1

            # You are recording, but now its quiet, so stop.
            if num_silent_dwells > max_silent_dwells:
                break
            
            # You are recording, but the active sample is getting too big, so stop.
            if is_recording:
                if (time.time() - start_time) > max_sample_length_s:
                    break

            if max_run_time_s is not None:  # If max run time set, make sure not over this limit.
                if (time.time() - start_time) > max_run_time_s:
                    break
        
        self._close_stream()
        
        self.data = np.concatenate(audio_array)
        self.rate_hz = rate_hz
        self.length_s = len(self.data) / self.rate_hz

        return
    

    def resample_audio(self, rate_hz: int=16000):
        '''
        Resample the current data to the new rate.

        rate_hz: the target rate to resample the data to.
        '''

        if self.data is None:
            raise ValueError('No data to resample!')

        target_rate_hz = rate_hz  # Renaming to make it clear.

        if target_rate_hz == self.rate_hz:
            return  # Data already correctly sampled.

        # Calculate the greatest common divisor (GCD) for efficient resampling.
        gcd = np.gcd(self.rate_hz, target_rate_hz)
        up = target_rate_hz // gcd
        down = self.rate_hz // gcd

        self.data = resample_poly(x=self.data, up=up, down=down).astype(np.float32)
        self.rate_hz = target_rate_hz

        return


    def set_silence_threshold(
        self,
        time_s: float=3.0,
        silence_bump_percent: float=100.0,
        rate_hz: int=16000) -> None:
        '''
        Read in a short audio clip and try to determine the current level of background noise.
        This is the numerical value for a signal where values over this limit are probably signal
        and values under it are probably noise. 

        This is intended to be used by other methods that are trying to determine the activity level
        of a sound clip, specifically whether a sample of audio is considered "active" by comparing
        how much of the audio is above the level of what was considered ambient noise.

        This is meant to be run when there is only "ambient" background noise present and not
        anything active (like someone talking). To this end, it isn't called automatically anywhere
        because that could throw off levels if you're not prepared. Call this manually.

        Technique: Calculate the Root Mean Squared value of the signal. This is a good approximation
        for a signal's "power". Divide by length so the value is independent of recording length.
        Calculate it for the ambient noise. Scale this up by a small amount (X percent) to set a
        level to compare other signal's power to. If a signal has an RMS over the threshold, then
        there's a good chance that it's active.

        ---
        time_s: time to record in seconds to calculate average noise.
        silence_bump_percent: The percent to scale up (or down) the average signal power for a silent
            signal. Example: If your average quiet signal power is 10, then set the line for detecting
            an active signal at 100% (silence_bump_percent = 100) over 10 for a limit of 20. In testing, 
            clearly spoken words seem to be anywhere from 3 - 25x greater than the background noise
            when that noise is a relatively quiet room and the mic is close the source. 
        '''

        data = self.record(time_s=time_s, set_data=False, rate_hz=rate_hz)

        rms_silence = self.calc_rms(audio_array=data)  # Root mean square.
        
        self.silence_threshold = rms_silence * (1 + silence_bump_percent/100)
        
        return


    def plot(self, save_path: str='audio.png') -> None:
        '''
        Plots current data. Since plot objects tend to freeze up the system until they are closed, 
        plot will create a plot and save it as a .png without ever actually showing it to screen.

        TODO: Would be nice to make plot interactive so I could zoom in on sections
        instead of getting the plot all condensed at once.

        ---
        save_path: path to the audio plot file to save.
        '''

        if self.data is None:
            raise ValueError('Must have data to plot!')
        if save_path is not None and not save_path.endswith('.png'):
            raise ValueError('"save_path" must be a .png file!')
        
        time_axis = np.arange(len(self.data)) / self.rate_hz

        plt.figure()
        plt.plot(time_axis, self.data)
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        plt.title('Audio Signal')

        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory.

        return


    def save_as(self, save_path: str='audio.flac') -> None:
        '''
        Saves the current data to a .flac file.

        ---
        save_path: full file path to .flac file in save location.
        '''
        
        if self.data is None:
            raise ValueError('Must have data to save as .wav file!')
        if save_path is not None and not save_path.endswith('.flac'):
            raise ValueError('"save_path" must be a .wav file!')
        
        sf.write(
            file=save_path,
            data=self.data,
            samplerate=self.rate_hz)

        return


    def _open_stream(
        self,
        rate_hz: int=16000,
        channels: int=1,
        frames_per_buffer: int=512, # 1024 is a good value.
        audio_format: int=pyaudio.paFloat32) -> None:  # paInt16
        '''
        Opens an audio stream for recording using pyaudio.

        ---
        rate_hz: Recording rate - 16k is pretty standard for a lot of applications. 
        channels: 1 is mono, 2 is stereo.
        frames_per_buffer: parameter for managing the trade-off between audio processing
            latency and overhead. When audio data is read from or written to the stream, 
            it is done in chunks of this buffer size. A smaller buffer size means that 
            audio data is transferred more frequently with lower latency but at potentially
            higher processing overhead, as the system has to handle more buffers per unit time.
            Conversely, a larger buffer size reduces the processing overhead (since there 
            are fewer buffers to handle) but increases the latency, which might not be 
            suitable for real-time audio processing tasks. Values may need to be a power of 2?
            GPT suggested 512 as an option.
        audio_format: TODO.

        # open_now: bool=True
        '''

        if frames_per_buffer is None:
            frames_per_buffer = self.calc_frames_per_buffer(rate_hz=rate_hz)

        self._pyaudio_obj = pyaudio.PyAudio()

        self._stream = self._pyaudio_obj.open(
            input=True,  # Tells the stream you are opening to record data.
            rate=rate_hz,  # Sample Rate.
            channels=channels,  # If not set, this should default to the recording device's default.
            frames_per_buffer=frames_per_buffer,
            format=audio_format,
            input_device_index=self.input_device_index,
            start=True)
        
        self.rate_hz = rate_hz
        # self.channels = channels

        return
    

    def _close_stream(self) -> None:
        '''
        Cleanly close the stream and the PyAudio object from which the stream originated.
        '''
        
        # Cleanup stream and audio object.
        self._stream.stop_stream()
        self._stream.close()
        self._pyaudio_obj.terminate()

        # TODO: Should I reset the stream and pyaudio params to None?

        return


    def _read_stream(self, read_time_s: float=.5) -> np.array:
        '''
        Reads an open audio stream to record and store sound levels into a numpy array.

        Ideally, the size of numeric data would be exactly equal to the sample rate / record time.
        But since audio is recorded in buffer chunks (frames per buffer), you do an
        approximation of how many buffers of frames_per_buffer do I need to get close to
        my required read_time_s total data. This result is currently ceilinged to avoid the scenario
        where you need less than 1 total buffer. But there's probably an argument for flooring it
        or rounding it instead.

        ---
        read_time_s: the total time to read the stream in seconds.
        '''

        numeric_data = []
        for _ in range(0, int(np.ceil(self.rate_hz / self._stream._frames_per_buffer * read_time_s))):
            raw_buffer = self._stream.read(self._stream._frames_per_buffer)
            numeric_data.extend(np.frombuffer(raw_buffer, dtype=np.float32))  # TODO: does dtype have to match param "format" of sample?

        numeric_data = np.array(numeric_data)  # Convert list to pure numpy array.

        return numeric_data
    

    @staticmethod
    def calc_frames_per_buffer(rate_hz: int, desired_latency_ms: int=50) -> int:
        '''
        If the sample rate is too high, a lower frames_per_buffer will cause
        an overflow error as the system can't buffer data and kick it onwards
        fast enough to keep up with the sampling rate.

        TODO: Actually, a low sample rate is better? 256 worked.
        '''

        MS_TO_SEC = 1/1000

        raw_value = rate_hz * desired_latency_ms * MS_TO_SEC

        # Round to the nearest power of 2.
        result = 2 ** int(math.log2(raw_value)) if raw_value >= 256 else 256

        return result


    @staticmethod
    def calc_rms(audio_array: np.array) -> float:
        '''
        Calculates the RMS of an array. This is a good measure for "signal
        strength", especially when trying to compare the strengths of two signals.

        ---
        audioa_array: numpy array of audio signal data.
        '''

        rms = np.sqrt(np.mean(np.square(audio_array))) # Root mean square.

        return rms
