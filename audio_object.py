import numpy as np
import pyaudio
import soundfile as sf
import matplotlib.pyplot as plt
import time


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

    def __init__(self):

        self.data = None

        self._pyaudio_obj = None
        self._stream = None

        self.rate_hz = None
        self.channels = None

        self.length_s = None
        self.noise_level = None

    def record(self, time_s: float=3.0, set_data: bool=True) -> np.array:
        '''
        Record an audio sample for X seconds. Saves recorded sample into self.data.

        ---
        time_s: time to record sample for in seconds.
        set_data: whether or not to keep the data. Sometimes recording is for a temp calculation.
        '''

        self._open_stream()
        data = self._read_stream(read_time_s=time_s)
        if set_data:
            self.data = data
        self._close_stream()

        self.length_s = time_s

        return data
    
    def record_activity(self, max_collect_s: float=10.0, dwell_s: float=.2) -> None:
        '''
        Opens an audio stream and tries to collect the next full sample of audio.
        Automatically cuts off the stream once too much time has passed.
        Limits the collected audio to the start and stop of activity.
        Does some calculation to only record and keep audio if it counts as activity.

        ---
        max_collect_s: maximum time to collect data before timing out (even if sample is still active).
            Too long and the ASR model could have difficulty predicting on such a large sample.
            Realistically, the timeout will be determine more by dwell_s (since a period of dwell_s
            that is silent will trigger a break).
        dwell_s: dwell time (in seconds) to collect a micro sample to analyze for activity.
        '''

        ready_for_model = False  # Initialize to no good sample found.
        
        self._open_stream()

        full_sample = []
        
        start_time = time.time()  # Start clock.
        while (time.time() - start_time) < max_collect_s:
            
            sample = self._read_stream(read_time_s=dwell_s)

            is_active, ends_dead = self._is_active(audio_array=sample)

            if is_active:  # Has something that looks like non-background sound.
                full_sample.extend(sample)  # Add snippet to the full sample array.
                ready_for_model = True

            if ready_for_model and ends_dead:  # Means dead zone found after a sample with activity.
                break

        self._close_stream()

        self.data = np.array(full_sample)  # Set the main data as the recorded sample.
        self.length_s = len(self.data) / self.rate_hz

        return
    
    def set_noise_level(self, time_s: float=3.0, percentile: float=99.0) -> None:
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

        Technique: Create a distribution of the absolute value of all values in the sample and 
        draw the line at the Xth percentile. Making a vague assumption that if X% of the noise 
        signal is below this level then its a decent limit.

        ---
        time_s: time to record in seconds to calculate average noise.
        '''

        data = self.record(time_s=time_s, set_data=False)

        self.noise_level = np.percentile(abs(data), percentile)
        
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

        # Save the plot as a PNG file.
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory.
    
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
            samplerate=self.rate_hz)  # subtype='PCM_24'

        return

    def _open_stream(
        self,
        rate_hz: int=16000,
        channels: int=1,
        frames_per_buffer: int=1024,
        audio_format: int=pyaudio.paFloat32) -> None: #paInt16
        '''
        Opens an audio stream for recording using pyaudio.

        ---
        channels: 1 is mono, 2 is stereo.
        rate_hz: Recording rate - 16k is pretty standard for a lot of applications. 
        audio_format: TODO. 
        frames_per_buffer: parameter for managing the trade-off between audio processing
            latency and overhead. When audio data is read from or written to the stream, 
            it is done in chunks of this buffer size. A smaller buffer size means that 
            audio data is transferred more frequently with lower latency but at potentially
            higher processing overhead, as the system has to handle more buffers per unit time.
            Conversely, a larger buffer size reduces the processing overhead (since there 
            are fewer buffers to handle) but increases the latency, which might not be 
            suitable for real-time audio processing tasks.
        '''

        self._pyaudio_obj = pyaudio.PyAudio()

        self._stream = self._pyaudio_obj.open(
            input=True,  # Tells the stream you are opening to record data.
            rate=rate_hz,  # Sample Rate.
            channels=channels,  # If not set, this should default to the recording device's default.
            frames_per_buffer=frames_per_buffer,
            format=audio_format)
        
        self.rate_hz = rate_hz
        self.channels = channels

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
            numeric_data.extend(np.frombuffer(raw_buffer, dtype=np.float32))  # int16 TODO: does dtype have to match param "format" of sample?

        numeric_data = np.array(numeric_data)  # Convert list to pure numpy array.

        return numeric_data
    
    def _is_active(self, audio_array: np.array, active_percent: float=10.0, look_back_percent: float=25.0) -> bool:
        '''
        Determine if an audio sample has any activity, and if it ends in a dead zone. 

        ---
        audio_array: the audio signal to be analyzed.
        active_percent: if at least this percent of the signal is active, then the
            sample is considered an "active" sample.
        look_back_percent: Look at the last X percent of the signal. If there no 
            above-noise-threshold elements in this "back percent" of the sample, its 
            the end of the clip and the sound sample "end_dead".
        '''

        if self.noise_level is None:
            raise ValueError('Must calculate noise level first!')

        SOUND_THRESHOLD = 1000  # Anything below this is considered background noise. Above is considered "active".
        PERCENT_ACTIVE_REQUIRED = 10  # This percent samples over the SOUND_THRESHOLD is considered an "active" clip. 
        PERCENT_LOOK_BACK = 25  # If there no above-noise-threshold elements in this "back percent" of the sample, its the end of the clip.

        is_active = False  # Initial state for sample has active points.
        ends_dead = True  # Initial for the clip ending in a non-active state.

        sample_length = len(audio_array)

        n_look_back = int(np.round(look_back_percent/100 * sample_length))  # Number of frames to look back on.
        num_active_required = int(np.round(active_percent/100 * sample_length))  # Num active frames required.

        # This is a list of the all the indexes of the frames that were considered "active".
        i_active = [i for i, value in enumerate(audio_array) if abs(value) > self.noise_level]

        if len(i_active) >= num_active_required:  # Total number of active frames is above threshold...
            is_active = True
        
            # If the last active frame is later than the percent look back frame...
            if i_active[-1] > (sample_length - n_look_back):  # Last active value.
                ends_dead = False

        return is_active, ends_dead
    