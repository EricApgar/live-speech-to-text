import numpy as np
import pyaudio
import time


class Audio:
    '''
    Easily record sound for X seconds.
    Handle recorded sound data as numpy array.
    Save data as a .wav file.
    Plot data.
        Save plot as .png at specified location.
        Default location is wherever stuff is being run.

    Conceptually, there should only be 1 stream open at a time.
    '''

    def __init__(self, data: np.array=None):

        self.data = data
        
        self.noise_floor = None

        self._pyaudio_obj = None
        self._stream = None

    def _open_stream(
        self,
        channels: int=1,
        rate_hz: int=16000,
        frame_count: int=1024,
        audio_format: int=pyaudio.paInt16) -> pyaudio.PyAudio:
        '''
        Opens an audio stream for recording using pyaudio.

        channels: 1 is mono, 2 is stereo.
        rate_hz: Recording rate - 16k is pretty standard for a lot of applications. 
        frame_count: TODO. Number of frames per buffer???
        audio_format: TODO. 
        '''

        self._pyaudio_obj = pyaudio.PyAudio()

        self._stream = self._pyaudio_obj.open(
            format=audio_format,
            channels=channels,
            rate=rate_hz,
            input=True,
            frames_per_buffer=frame_count)
        
        return
    
    def _close_stream(self) -> bool:
        '''
        Cleanly close the stream and the PyAudio object from which the stream originated.
        '''
        
        # Cleanup stream and audio object.
        self._stream.stop_stream()
        self._stream.close()
        self._pyaudio_obj.terminate()

        return True

    def _read_stream(self, read_time_s: float=.1, frame_count: int=1024) -> np.array:
        '''
        Reads an open audio stream to record and store sound levels into a numpy array.

        read_time_s: the total time to read the stream in seconds.
        frame_count: The chunk size of how many frames to read at a time. It's possible that this is
            somewhat dependent on the audio stream which is initialized with a "frames_per_buffer"
            parameter. I haven't tested trying to open an audio stream with one type of frame count
            and then reading with a different. That's a TODO.
        '''

        full_data = []

        start_time = time.time()

        # Keep reading blocks of data until the time is up.
        while (time.time() - start_time) < read_time_s:

            raw_sample = self._stream.read(frame_count)  # Time to read frame_count is approx (1/SAMPLE_RATE * frame_count) seconds.
            num_sample = np.double(np.frombuffer(raw_sample, dtype=np.int16))  # Converts some binary buffer obj to numpy array.
            full_data.extend(num_sample)  # Tacks on the data to the accumulating total sample.

        return full_data    
    
    def detect_audio():
        '''
        Opens an audio stream and tries to collect the next full sample of audio.
        Automatically cuts off the stream once too much time has passed.
        Limits the collected audio to the start and stop of activity.
        Does some calculation to only record and keep audio if it counts as activity.
        '''

        return

    def record(self, time_s: float):
        '''
        Record an audio sample for X seconds. Saves recorded sample into self.data.

        time_s: time to record sample for in seconds.
        '''

        stream = self._open_stream()
        self.data = self._read_stream(stream=stream, read_time_s=time_s)
        stream.close()

        return

    def _calc_activity_levels(self, audio_array: np.array):
        '''
        Determine if an audio sample has any activity, and if it ends in a dead zone. 
        '''

        SOUND_THRESHOLD = 1000  # Anything below this is considered background noise. Above is considered "active".
        PERCENT_ACTIVE_REQUIRED = 10  # This percent samples over the SOUND_THRESHOLD is considered an "active" clip. 
        PERCENT_LOOK_BACK = 25  # If there no above-noise-threshold elements in this "back percent" of the sample, its the end of the clip.

        is_active = False  # Initial state for sample has active points.
        ends_dead = True  # Initial for the clip ending in a non-active state.

        sample_length = len(audio_array)

        n_look_back = int(np.round(PERCENT_LOOK_BACK/100 * sample_length))  # Number of frames to look back on.
        num_active_required = int(np.round(PERCENT_ACTIVE_REQUIRED/100 * sample_length))  # Num active frames required.

        # This is a list of the all the indexes of the frames that were considered "active".
        i_active = [i for i, value in enumerate(audio_array) if abs(value) > SOUND_THRESHOLD]

        if len(i_active) >= num_active_required:  # Total number of active frames is above threshold...
            is_active = True
        
            # If the last active frame is later than the percent look back frame...
            if i_active[-1] > (sample_length - n_look_back):  # Last active value.
                ends_dead = False

        return is_active, ends_dead
    
    def calc_noise_floor(self, time_s: float=3.0) -> float:
        '''
        Read in a short audio clip and try to determine the current level of background noise.
        This will be helpful for other methods which are trying to determine the activity level
        of a sound clip.

        This is meant to be run when there is only "ambient" background noise present and not
        anything active (like someone talking).

        time_s: time to record in seconds to calculate average noise.
        '''

        return